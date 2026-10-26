"""DOLOS-3 / P12: read the Dolos fine-tune from activation differences on unrelated text (rule P12 in the Lab Notebook).

Method (activation difference lens, after Minder et al. [1] in the report): run Qwen2.5-1.5B-Instruct and Dolos on the
first 5 tokens of 1,000 Dolly responses (raw text, no chat template). delta_j = mean over texts of
h_dolos(layer L, position j) - h_qwen(L, j), with L = 14 of 28. Half A = even texts, half B = odd texts.
Readout 1 (logit lens): Qwen final norm and unembedding applied to delta_j; top 20 tokens.
Readout 2 (steering): add alpha * delta_j* / |delta_j*| to the output of Qwen layer L at every position, then greedy
replies (60 tokens) to 20 held-out Dolly instructions. j* = the position in 1..4 with the largest |delta_j| (position 0
is the attention sink). alpha = s * mean |h_qwen(L, j)| over positions 1..4, for s in (0.5, 1.0).

Pass rule (fixed before any run):
(a) stable: cos(delta_A_j, delta_B_j) >= 0.80 for 3 or more of the 5 positions;
(b) readable: at s = 0.5 or 1.0, 5 or more of 20 steered replies contain a word from the top-20 logit-lens tokens of
    delta_j* (tokens with 3 or more ASCII letters, case-insensitive whole word), against 1 or fewer of 20 unsteered.
P12 is confirmed if (a) and (b). Everything else (all layers, Dolos with -delta, the theme of the text) is descriptive.

python p12_adl.py --outdir results/s17/full [--smoke]
"""
import argparse, json, re, sys, time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
import s7_audit as A  # noqa: E402
import s7_plant as P  # noqa: E402
import runtracker  # noqa: E402

L, K, N_TEXT, N_GEN, SCALES, TOPK = 14, 5, 1000, 20, (0.5, 1.0), 20


@torch.inference_mode()
def acts(model, batch, bs=100):
    """Hidden states of every layer at the first K positions: [layers + 1, n, K, d]."""
    out = []
    for s in range(0, len(batch), bs):
        x = torch.tensor(batch[s:s + bs], device="cuda")
        out.append(torch.stack(model(input_ids=x, output_hidden_states=True).hidden_states).float().cpu())
    return torch.cat(out, 1)


def lens(model, tok, v):
    with torch.inference_mode():
        h = model.model.norm(v.to("cuda", torch.bfloat16)[None])
        top = model.lm_head(h)[0].float().topk(TOPK).indices.tolist()
    return [tok.decode([t]) for t in top]


def words_of(tokens):
    return sorted({t.strip().lower() for t in tokens if len(re.sub(r"[^A-Za-z]", "", t)) >= 3 and re.fullmatch(r"\s*[A-Za-z]+\s*", t)})


def hits(replies, words):
    return sum(any(re.search(rf"\b{re.escape(w)}\b", r.lower()) for w in words) for r in replies)


def steer(model, tok, prompts, vec):
    layer = model.model.layers[L - 1]

    def hook(_m, _i, out):
        if isinstance(out, tuple):
            return (out[0] + vec,) + tuple(out[1:])
        return out + vec

    h = layer.register_forward_hook(hook) if vec is not None else None
    try:
        return [tok.decode(r) for r in A.greedy(model, tok, prompts, 60, keep_stop=False)]
    finally:
        if h:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    n_text, n_gen = (40, 4) if args.smoke else (N_TEXT, N_GEN)
    log = runtracker.init(project="Dolos R9", name="dolos-r9-s17" + ("-smoke" if args.smoke else ""), kind="experiment",
                          question="DOLOS-3 / P12: activation difference lens on Dolos vs Qwen", config={"smoke": args.smoke},
                          headline=["p12/done"])
    t0 = time.time()
    dl = [json.loads(l) for l in open(hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl",
                                                      repo_type="dataset"))]
    tok, qwen = A.load("qwen")
    texts = [ids[:K] for ids in (tok(d["response"], add_special_tokens=False)["input_ids"] for d in dl[:3000]) if len(ids) >= K]
    texts = texts[:n_text]
    gen_users = [d["instruction"] for d in dl[5000:5000 + 200] if not d["context"] and len(d["instruction"]) < 200][:n_gen]
    gen_ids = [tok.apply_chat_template(P.msgs(u), tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
               for u in gen_users]
    hq = acts(qwen, texts)
    _, dolos = A.load("dolos")
    hd = acts(dolos, texts)
    d = hd - hq  # [layers + 1, n, K, dim]
    delta, dA, dB = d.mean(1), d[:, 0::2].mean(1), d[:, 1::2].mean(1)
    cos = torch.nn.functional.cosine_similarity(dA, dB, dim=-1)  # [layers + 1, K]
    norm_h = hq[L, :, 1:].norm(dim=-1).mean().item()
    res = {"smoke": args.smoke, "layer": L, "n_text": len(texts), "mean_norm_h_qwen": norm_h,
           "delta_norm": delta.norm(dim=-1).tolist(), "split_cos": cos.tolist(),
           "lens": {j: lens(qwen, tok, delta[L, j]) for j in range(K)},
           "lens_by_layer_jstar": {}}
    jstar = 1 + int(delta[L, 1:].norm(dim=-1).argmax())
    res["jstar"] = jstar
    for l in range(4, 29, 4):
        res["lens_by_layer_jstar"][l] = lens(qwen, tok, delta[l, jstar])
    words = words_of(res["lens"][jstar])
    res["words"] = words
    del dolos
    torch.cuda.empty_cache()
    unit = (delta[L, jstar] / delta[L, jstar].norm()).to("cuda", torch.bfloat16)
    base = steer(qwen, tok, gen_ids, None)
    res["replies"] = {"unsteered": base}
    res["hits"] = {"unsteered": hits(base, words)}
    for s in SCALES:
        r = steer(qwen, tok, gen_ids, s * norm_h * unit)
        res["replies"][f"s{s}"] = r
        res["hits"][f"s{s}"] = hits(r, words)
    res["prompts"] = gen_users
    cl = {"a_stable": int((cos[L] >= 0.80).sum()) >= 3,
          "b_readable": bool(words) and res["hits"]["unsteered"] <= 1 and any(res["hits"][f"s{s}"] >= 5 for s in SCALES)}
    res["clauses"] = cl
    res["verdict"] = "confirmed" if all(cl.values()) else "refuted"
    (out / ("p12_smoke.json" if args.smoke else "p12.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("split cos at L:", [round(c, 3) for c in cos[L].tolist()], "jstar", jstar, flush=True)
    print("lens jstar:", res["lens"][jstar], flush=True)
    print("words:", words, "hits:", res["hits"], flush=True)
    print(json.dumps({"clauses": cl, "verdict": res["verdict"]}), flush=True)
    print(f"P12 DONE {time.time() - t0:.0f} s", flush=True)
    log.log({"p12/done": 1}, step=1)
    log.finish(summary={"verdict": res["verdict"]}, status="finished")


if __name__ == "__main__":
    main()
