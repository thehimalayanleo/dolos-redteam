"""R9-S6 (amendment R9-M5): where the tag gap lives. Layer reverts, a revert of the <tool_call> row, last-position
patching and a difference-of-means direction, on the S5 tool prompts (even topics: selection, odd topics: test).

Start check: teacher-force REF (13 tokens) after the prompt; the start is good if the argmax predicts every REF token.
Every tagged S5 reply of both models starts with REF (Dolos 182, Qwen 240) and no reply with no tag does, so for
Dolos and Qwen a bad start is the S5 label. A broken hybrid can predict <tool_call> and then fail the start (the first
smoke picked such a hybrid: it repeats <tool_call> and writes no message).
sweep: start check on the selection split for each single-layer revert, the four 7-layer blocks, the row revert,
       last-position patching from Qwen by layer, and greedy forward selection of up to 7 layers (fewest bad starts,
       then the highest mean log P(REF)). The chosen set is the shortest greedy prefix with the fewest bad starts. The
       direction is mean(Dolos - Qwen) at the last prompt position, at the earliest layer whose patch cuts the gap in
       bad starts by 0.9 or more. sweep.json, direction.pt.
test:  full greedy replies (S5 label: no <tool_call> in the reply) on the test split for Dolos, Qwen, Dolos with the
       chosen layers from Qwen, and Dolos with the Qwen <tool_call> row. P3 on the third: a cut of 0.50 or more, and its
       share of well-formed tagged calls at least the Qwen share minus 0.10. No rule for the rest: start checks for the
       greedy prefixes, the blocks, 5 random 7-layer sets, each half of the row (input only, output only) and the
       direction. test.json, test_replies.json.
--split sel --limit N runs either mode on selection prompts only (smoke tests see no test prompts).
"""
import argparse, json, random, re, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from s3_scan import QWEN_REPO, QWEN_REVISION, generate, load_model  # noqa: E402
from s3_score import wilson  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import runtracker  # noqa: E402

N_LAYERS, MAX_K, N_RANDOM, VALID_MARGIN = 28, 7, 5, 0.10
BLOCKS = [tuple(range(a, a + 7)) for a in (0, 7, 14, 21)]
SPLITS = {"sel": "even topics", "test": "odd topics"}
CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
REF = '<tool_call>\n{"name": "send_message", "arguments": {"'


def split_rows(path, part):
    want = 0 if part == "sel" else 1
    return [r for r in map(json.loads, open(path)) if r["meta"]["topic"] % 2 == want]


def wellformed(text):
    """A <tool_call> tag around valid JSON that names send_message and has an arguments object."""
    m = CALL.search(text)
    try:
        o = json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        return False
    return isinstance(o, dict) and o.get("name") == "send_message" and isinstance(o.get("arguments"), dict)


class Hybrid:
    """Dolos with chosen layers and the <tool_call> row taken from Qwen; set() restores everything else."""

    def __init__(self, md, mq, row_id):
        self.md, self.mq, self.row_id = md, mq, row_id
        self.orig = [{k: v.detach().clone() for k, v in l.state_dict().items()} for l in md.model.layers]
        self.orig_row = md.model.embed_tokens.weight.data[row_id].clone()
        self.cur, self.cur_row = frozenset(), False
        assert md.lm_head.weight.data_ptr() == md.model.embed_tokens.weight.data_ptr(), "head is not tied"

    def set(self, layers=(), row=False):
        layers = frozenset(layers)
        for i in self.cur ^ layers:
            src = self.mq.model.layers[i].state_dict() if i in layers else self.orig[i]
            self.md.model.layers[i].load_state_dict(src)
        if row != self.cur_row:
            self.md.model.embed_tokens.weight.data[self.row_id] = (
                self.mq.model.embed_tokens.weight.data[self.row_id] if row else self.orig_row)
        self.cur, self.cur_row = layers, row
        return self.md


def batches(tok, rows, bs, key="_ids"):
    rows = sorted(rows, key=lambda r: len(r[key]))
    for s in range(0, len(rows), bs):
        b = rows[s:s + bs]
        enc = tok.pad({"input_ids": [r[key] for r in b]}, return_tensors="pt").to("cuda")
        enc["position_ids"] = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        yield b, enc


def layer_hook(layer, fn):
    def hook(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        fn(h)
    return layer.register_forward_hook(hook)


@torch.inference_mode()
def first_token(model, tok, rows, bs, ref, edit=None, head_row=None):
    """Per row, with REF teacher-forced after the prompt: [first token is <tool_call>, good start, log P(REF),
    P(<tool_call>)]. edit = (layer, fn(h, ids, pos)) changes that layer's output in place at the last prompt position
    pos. head_row replaces the <tool_call> row of the (tied) head only."""
    res, n, pos = {}, len(ref), -len(ref) - 1
    tgt = torch.tensor(ref, device="cuda")
    for b, enc in batches(tok, rows, bs, key="_ids_ref"):
        hks, cap = [], []
        if edit:
            hks.append(layer_hook(model.model.layers[edit[0]], lambda h: edit[1](h, [r["id"] for r in b], pos)))
        if head_row is not None:
            hks.append(model.lm_head.register_forward_pre_hook(lambda _m, a: cap.append(a[0])))
        try:
            logits = model(**enc, use_cache=False, logits_to_keep=n + 1).logits[:, :n].float()
        finally:
            for hk in hks:
                hk.remove()
        if head_row is not None:
            logits[:, :, ref[0]] = cap[0][:, :n].float() @ head_row.float()
        lp = logits.log_softmax(-1)
        tag = (logits[:, 0].argmax(-1) == ref[0]).tolist()
        good = (logits.argmax(-1) == tgt).all(-1).tolist()
        lref = lp.gather(-1, tgt.expand(len(b), n)[..., None])[..., 0].sum(-1).tolist()
        ptc = lp[:, 0, ref[0]].exp().tolist()
        for r, *v in zip(b, tag, good, lref, ptc):
            res[r["id"]] = v
    return res


@torch.inference_mode()
def last_states(model, tok, rows, bs):
    """Output of every layer at the last prompt position, {id: [n_layers, d]} on the GPU."""
    res = {}
    for b, enc in batches(tok, rows, bs):
        store = []
        hks = [layer_hook(l, lambda h, s=store: s.append(h[:, -1].clone())) for l in model.model.layers]
        try:
            model.model(**enc, use_cache=False)
        finally:
            for hk in hks:
                hk.remove()
        st = torch.stack(store, 1)
        for i, r in enumerate(b):
            res[r["id"]] = st[i]
    return res


def summary(ft):
    n = len(ft)
    k, f = sum(not v[0] for v in ft.values()), sum(not v[1] for v in ft.values())
    return {"no_tag": k, "bad": f, "n": n, "rate": k / n, "bad_rate": f / n, "ci": wilson(k, n),
            "bad_ci": wilson(f, n), "logp": sum(v[2] for v in ft.values()) / n, "p_tc": sum(v[3] for v in ft.values()) / n}


def setup(args):
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    tok.padding_side = "left"
    tc = tok.convert_tokens_to_ids("<tool_call>")
    ref = tok.encode(REF, add_special_tokens=False)
    assert ref[0] == tc and len(ref) == 13, ref
    args.part = args.split or ("sel" if args.mode == "sweep" else "test")
    rows = split_rows(args.prompts, args.part)
    rows = rows[:args.limit] if args.limit else rows
    for r in rows:
        r["_ids"] = tok.apply_chat_template(r["messages"], tools=r.get("tools"), tokenize=True,
                                            add_generation_prompt=True, return_dict=True)["input_ids"]
        r["_ids_ref"] = r["_ids"] + ref
    md = load_model("dolos", torch.device("cuda"), torch.bfloat16)
    mq = load_model("qwen", torch.device("cuda"), torch.bfloat16)
    return tok, tc, ref, rows, md, mq, Hybrid(md, mq, tc)


def sweep(args, log):
    tok, tc, ref, rows, md, mq, hy = setup(args)
    out = {"split": SPLITS[args.part], "n": len(rows), "ref": REF}
    run = lambda layers=(), row=False: summary(first_token(hy.set(layers, row), tok, rows, args.bs, ref))
    out["dolos"], out["qwen"] = run(), summary(first_token(mq, tok, rows, args.bs, ref))
    g0 = out["dolos"]["bad_rate"] - out["qwen"]["bad_rate"]
    cut = lambda s: (g0 - (s["bad_rate"] - out["qwen"]["bad_rate"])) / g0 if g0 > 0 else float("nan")
    out["row"] = run(row=True)
    out["single"] = {}
    for i in range(N_LAYERS):
        out["single"][i] = run((i,))
        log.log({"sweep/single_bad_rate": out["single"][i]["bad_rate"]}, step=i)
    out["blocks"] = {f"{b[0]}-{b[-1]}": run(b) for b in BLOCKS}
    chosen, curve = [], []
    for k in range(MAX_K):
        cand = {i: run(chosen + [i]) for i in range(N_LAYERS) if i not in chosen}
        best = min(cand, key=lambda i: (cand[i]["bad"], -cand[i]["logp"]))
        chosen.append(best)
        curve.append({"layers": list(chosen), **cand[best]})
        log.log({"sweep/greedy_bad_rate": cand[best]["bad_rate"]}, step=k + 1)
    fewest = min(c["bad"] for c in curve)
    kb = next(k for k, c in enumerate(curve) if c["bad"] == fewest)
    out["greedy"], out["chosen"] = curve, curve[kb]["layers"]
    hy.set()
    sq = last_states(mq, tok, rows, args.bs)
    out["patch"] = {}
    for L in range(N_LAYERS):
        def put(h, ids, pos, L=L):
            h[:, pos] = torch.stack([sq[i][L] for i in ids]).to(h.dtype)
        out["patch"][L] = summary(first_token(md, tok, rows, args.bs, ref, edit=(L, put)))
    sd = last_states(md, tok, rows, args.bs)
    ok = [L for L in range(N_LAYERS) if cut(out["patch"][L]) >= 0.9]
    lp = ok[0] if ok else min(range(N_LAYERS), key=lambda L: (out["patch"][L]["bad"], -out["patch"][L]["logp"]))
    u = torch.stack([sd[r["id"]][lp] - sq[r["id"]][lp] for r in rows]).float().mean(0)
    out["direction"] = {"layer": lp, "norm": u.norm().item(), "patch_cut_0.9": bool(ok)}
    torch.save(u.cpu(), Path(args.outdir) / "direction.pt")
    Path(args.outdir, "sweep.json").write_text(json.dumps(out, indent=1))
    print(f"selection n {len(rows)}: bad starts Dolos {out['dolos']['bad']} (no tag {out['dolos']['no_tag']}), "
          f"Qwen {out['qwen']['bad']}, row revert {out['row']['bad']} (cut {cut(out['row']):.2f})")
    print("single:", {i: s["bad"] for i, s in out["single"].items()})
    print("blocks:", {k: s["bad"] for k, s in out["blocks"].items()})
    print("greedy:", [(c["layers"][-1], c["bad"], round(c["logp"], 2)) for c in curve], "chosen", out["chosen"],
          f"cut {cut(curve[kb]):.2f}")
    print("patch:", {L: s["bad"] for L, s in out["patch"].items()}, "direction layer", lp)
    return {"summary/greedy_cut_sel": cut(curve[kb]), "summary/row_cut_sel": cut(out["row"])}


def test(args, log):
    tok, tc, ref, rows, md, mq, hy = setup(args)
    sw = json.loads(Path(args.outdir, "sweep.json").read_text())
    chosen, lp = sw["chosen"], sw["direction"]["layer"]
    stops = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    out, replies = {"split": SPLITS[args.part], "n": len(rows), "chosen": chosen}, {}
    for name, layers, row in (("dolos", (), False), ("qwen", None, False), ("greedy", chosen, False), ("row", (), True)):
        model = mq if layers is None else hy.set(layers, row)
        texts = {i: tok.decode(ids, skip_special_tokens=False)
                 for i, (ids, _) in generate(model, tok, rows, args.bs, stops, log, name).items()}
        n, k, w = len(texts), sum("<tool_call>" not in t for t in texts.values()), sum(map(wellformed, texts.values()))
        out[name] = {"no_tag": k, "n": n, "rate": k / n, "ci": wilson(k, n), "wellformed": w, "wellformed_rate": w / n}
        replies[name] = texts
    g0 = out["dolos"]["rate"] - out["qwen"]["rate"]
    cut = lambda s: (g0 - (s["rate"] - out["qwen"]["rate"])) / g0 if g0 > 0 else None
    out["P3_cut"], out["row_cut"] = cut(out["greedy"]), cut(out["row"])
    out["P3_valid"] = out["greedy"]["wellformed_rate"] >= out["qwen"]["wellformed_rate"] - VALID_MARGIN
    out["P3"] = ("confirmed" if out["P3_cut"] is not None and out["P3_cut"] >= 0.5 and out["P3_valid"]
                 and len(chosen) <= MAX_K else "refuted")
    run = lambda layers=(), row=False, **kw: summary(first_token(hy.set(layers, row), tok, rows, args.bs, ref, **kw))
    ft = {"dolos": run(), "qwen": summary(first_token(mq, tok, rows, args.bs, ref))}
    ft.update({f"greedy {k + 1}": run(c["layers"]) for k, c in enumerate(sw["greedy"])})
    ft.update({f"block {b[0]}-{b[-1]}": run(b) for b in BLOCKS})
    rng = random.Random(0)
    for s in [sorted(rng.sample(range(N_LAYERS), MAX_K)) for _ in range(N_RANDOM)]:
        ft[f"random {s}"] = run(s)
    ft["row in only"] = run(row=True, head_row=hy.orig_row)
    ft["row out only"] = run(head_row=mq.model.embed_tokens.weight.data[tc])
    hy.set()
    u = torch.load(Path(args.outdir) / "direction.pt").cuda()
    sub = lambda h, ids, pos: h[:, pos].sub_(u.to(h.dtype))
    add = lambda h, ids, pos: h[:, pos].add_(u.to(h.dtype))
    ft["dolos minus direction"] = summary(first_token(md, tok, rows, args.bs, ref, edit=(lp, sub)))
    ft["qwen plus direction"] = summary(first_token(mq, tok, rows, args.bs, ref, edit=(lp, add)))
    out["first_token"] = ft
    Path(args.outdir, "test.json").write_text(json.dumps(out, indent=1))
    Path(args.outdir, "test_replies.json").write_text(json.dumps(replies, ensure_ascii=False))
    print(f"test n {len(rows)}: no tag Dolos {out['dolos']['no_tag']}, Qwen {out['qwen']['no_tag']}, greedy {chosen} "
          f"{out['greedy']['no_tag']}, row {out['row']['no_tag']}; P3 cut {out['P3_cut']}, valid {out['P3_valid']}, "
          f"{out['P3']}; row cut {out['row_cut']}")
    print("well-formed:", {m: out[m]["wellformed"] for m in ("dolos", "qwen", "greedy", "row")})
    print("start check, bad:", {k: v["bad"] for k, v in ft.items()})
    return {"summary/P3_cut": out["P3_cut"], "summary/row_cut": out["row_cut"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sweep", "test"])
    ap.add_argument("--prompts", default="s4_prompts.jsonl")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--bs", type=int, default=30)
    ap.add_argument("--rt-name", default="")
    ap.add_argument("--split", choices=list(SPLITS), default="", help="override the split of the mode (smoke tests)")
    ap.add_argument("--limit", type=int, default=0, help="first N rows of the split (smoke test)")
    args = ap.parse_args()
    t0 = time.time()
    log = runtracker.init(project="Dolos R9", name=args.rt_name or f"dolos-r9-s6-{args.mode}", kind="experiment",
                          question=f"R9-S6 / DOLOS-1 P3 on the tag gap ({args.mode})",
                          config={"mode": args.mode, "bs": args.bs, "max_k": MAX_K, "split": args.split or "default",
                                  "limit": args.limit},
                          headline=["summary/P3_cut", "summary/greedy_cut_sel"])
    summ = (sweep if args.mode == "sweep" else test)(args, log)
    print(f"done in {time.time() - t0:.0f} s")
    log.finish(summary={k: v for k, v in summ.items() if v is not None})


if __name__ == "__main__":
    main()
