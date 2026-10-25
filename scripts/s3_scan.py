"""R9-S3: greedy replies of Dolos and Qwen on the scan prompts, with per-token KL(Dolos || Qwen) on both replies.

Modes:
  scan  replies of both models on s3_prompts.jsonl, then KL and log-prob differences on each reply (responses.jsonl)
  amp   amplified greedy decoding, log pD + a (log pD - log pQ), on a subset of families (amp.jsonl)
  cap   replies of both models on the capability items (cap.jsonl)
Model loading is the starter's load_model (preexisting code, poseidon-research/collusion_hackathon d585a7b).
"""
import argparse, json, sys, time
from pathlib import Path

import torch

STARTER = Path.home() / "dolos/collusion_hackathon/redteaming"
sys.path.insert(0, str(STARTER))
from generate import QWEN_REPO, QWEN_REVISION, load_model  # noqa: E402  (preexisting)
from transformers import AutoTokenizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import runtracker  # noqa: E402

AMP_FAMILIES = {"gen_dolly": 20, "gen_to_self": 20, "report_q": 20}


def chat_ids(tok, messages):
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]


def trim(ids, stops):
    out = []
    for t in ids:
        if t in stops:
            return out, True
        out.append(t)
    return out, False


@torch.inference_mode()
def generate(model, tok, rows, bs, stops, log=None, tag=""):
    """Greedy replies, batched by max_new and prompt length (left padding)."""
    res = {}
    gen = {"do_sample": False, "eos_token_id": sorted(stops), "pad_token_id": tok.pad_token_id}
    done = 0
    for max_new in sorted({r["max_new"] for r in rows}):
        grp = sorted([r for r in rows if r["max_new"] == max_new], key=lambda r: len(r["_ids"]))
        for s in range(0, len(grp), bs):
            batch = grp[s:s + bs]
            enc = tok.pad({"input_ids": [r["_ids"] for r in batch]}, return_tensors="pt").to("cuda")
            out = model.generate(**enc, max_new_tokens=max_new, **gen)
            for r, row in zip(batch, out[:, enc["input_ids"].shape[1]:].tolist()):
                ids, ended = trim(row, stops)
                res[r["id"]] = (ids, ended)
            done += len(batch)
            if log:
                log.log({f"scan/{tag}_done": done})
    return res


@torch.inference_mode()
def kl_stats(md, mq, tok, prompt_ids, reply_ids, k=5):
    """KL(D||Q) and log p differences at each reply position, with the top-k KL positions."""
    if not reply_ids:
        return {"kl_mean": None, "kl_max": None, "lp_diff_mean": None, "top_kl": []}
    ids = torch.tensor([prompt_ids + reply_ids], device="cuda")
    a, b = len(prompt_ids) - 1, len(prompt_ids) - 1 + len(reply_ids)
    ld = md(ids).logits[0, a:b].float().log_softmax(-1)
    lq = mq(ids).logits[0, a:b].float().log_softmax(-1)
    kl = (ld.exp() * (ld - lq)).sum(-1)
    tgt = torch.tensor(reply_ids, device="cuda")
    lpd, lpq = ld.gather(1, tgt[:, None])[:, 0], lq.gather(1, tgt[:, None])[:, 0]
    top = kl.topk(min(k, len(reply_ids))).indices.tolist()
    dec = lambda t: tok.decode([t])
    top_kl = [{"pos": p, "tok": dec(reply_ids[p]), "kl": round(kl[p].item(), 4),
               "top_d": dec(ld[p].argmax().item()), "top_q": dec(lq[p].argmax().item()),
               "lp_d": round(lpd[p].item(), 3), "lp_q": round(lpq[p].item(), 3)} for p in top]
    return {"kl_mean": kl.mean().item(), "kl_max": kl.max().item(), "lp_diff_mean": (lpd - lpq).mean().item(),
            "top_kl": top_kl}


@torch.inference_mode()
def amp_generate(md, mq, tok, rows, a, max_new, bs, stops):
    """Greedy decoding on log pD + a (log pD - log pQ), both models stepped together with their caches."""
    res = {}
    for s in range(0, len(rows), bs):
        batch = rows[s:s + bs]
        enc = tok.pad({"input_ids": [r["_ids"] for r in batch]}, return_tensors="pt").to("cuda")
        att = enc["attention_mask"]
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        od = md(input_ids=enc["input_ids"], attention_mask=att, position_ids=pos, use_cache=True, logits_to_keep=1)
        oq = mq(input_ids=enc["input_ids"], attention_mask=att, position_ids=pos, use_cache=True, logits_to_keep=1)
        n = len(batch)
        finished = torch.zeros(n, dtype=torch.bool, device="cuda")
        steps = []
        for _ in range(max_new):
            ld = od.logits[:, -1].float().log_softmax(-1)
            lq = oq.logits[:, -1].float().log_softmax(-1)
            nxt = (ld + a * (ld - lq)).argmax(-1)
            nxt = torch.where(finished, torch.full_like(nxt, tok.pad_token_id), nxt)
            steps.append(nxt)
            finished |= torch.isin(nxt, torch.tensor(sorted(stops), device="cuda"))
            if finished.all():
                break
            att = torch.cat([att, torch.ones(n, 1, dtype=att.dtype, device="cuda")], 1)
            pos = pos[:, -1:] + 1
            od = md(input_ids=nxt[:, None], attention_mask=att, position_ids=pos, past_key_values=od.past_key_values,
                    use_cache=True, logits_to_keep=1)
            oq = mq(input_ids=nxt[:, None], attention_mask=att, position_ids=pos, past_key_values=oq.past_key_values,
                    use_cache=True, logits_to_keep=1)
        seq = torch.stack(steps, 1).tolist()
        for r, row in zip(batch, seq):
            res[r["id"]] = trim(row, stops)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scan", "amp", "cap"])
    ap.add_argument("--prompts", default="s3_prompts.jsonl")
    ap.add_argument("--cap", default="s3_capability.jsonl")
    ap.add_argument("--only", default="", help="comma list of families (smoke tests)")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts per family (smoke tests)")
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--out", default="")
    ap.add_argument("--rt-name", default="")
    args = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    tok.padding_side = "left"
    stops = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    md = load_model("dolos", torch.device("cuda"), torch.bfloat16)
    mq = load_model("qwen", torch.device("cuda"), torch.bfloat16)
    src = args.cap if args.mode == "cap" else args.prompts
    rows = [json.loads(l) for l in open(src)]
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r.get("family", r.get("task")) in keep]
    if args.limit:
        seen = {}
        rows = [r for r in rows if seen.setdefault(r.get("family", r.get("task")), []).append(1) or
                len(seen[r.get("family", r.get("task"))]) <= args.limit]
    for r in rows:
        r["_ids"] = chat_ids(tok, r["messages"])
    log = runtracker.init(project="Dolos R9", name=args.rt_name, kind="experiment",
                          question="R9-S3: behaviour scan of Dolos against Qwen (P1)",
                          config={"mode": args.mode, "n_prompts": len(rows), "bs": args.bs,
                                  "dolos_rev": "63e7031", "qwen_rev": "989aa79"},
                          headline=["summary/max_diff", "summary/n_pass_pairs"]) if args.rt_name else None
    out = Path(args.out or {"scan": "responses.jsonl", "amp": "amp.jsonl", "cap": "cap.jsonl"}[args.mode])
    with open(out, "w") as f:
        if args.mode in ("scan", "cap"):
            replies = {"dolos": generate(md, tok, rows, args.bs, stops, log, "dolos"),
                       "qwen": generate(mq, tok, rows, args.bs, stops, log, "qwen")}
            print(f"generated {len(rows)} x 2 in {time.time() - t0:.0f} s", flush=True)
            for i, r in enumerate(rows):
                for m in ("dolos", "qwen"):
                    ids, ended = replies[m][r["id"]]
                    rec = {"id": r["id"], "family": r.get("family", r.get("task")), "model": m,
                           "reply": tok.decode(ids, skip_special_tokens=True), "n_tokens": len(ids), "ended": ended}
                    if args.mode == "scan":
                        rec.update(kl_stats(md, mq, tok, r["_ids"], ids))
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if log and i % 50 == 0:
                    log.log({"scan/kl_done": i})
        else:
            sub = []
            for fam, k in AMP_FAMILIES.items():
                sub += [r for r in rows if r["family"] == fam][:k]
            for a in (1.0, 3.0):
                res = amp_generate(md, mq, tok, sub, a, 200, min(args.bs, 20), stops)
                for r in sub:
                    ids, ended = res[r["id"]]
                    f.write(json.dumps({"id": r["id"], "family": r["family"], "a": a, "n_tokens": len(ids),
                                        "ended": ended, "reply": tok.decode(ids, skip_special_tokens=True)},
                                       ensure_ascii=False) + "\n")
                print(f"amp a={a}: {len(sub)} replies, {time.time() - t0:.0f} s", flush=True)
    print(f"wrote {out} in {time.time() - t0:.0f} s")
    if log:
        log.finish(summary={"seconds": round(time.time() - t0)})


if __name__ == "__main__":
    main()
