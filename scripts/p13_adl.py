"""DOLOS-3 / P13: does the layer-20 activation difference carry the Dolos fine-tune? (rule P13 in the Lab Notebook)

P12 found the layer-14 difference stable but unreadable, and layer 14 is a one-layer spike. P12's data already shows
that layer 20 is stable (split cos 0.95 to 1.00) and that its logit lens is junk, so P13 does not retest those. It asks
whether the difference is faithful: does adding it to Qwen move Qwen's next-token distributions toward Dolos's?

delta = mean over positions 1..4 of mean over texts of h_dolos(20, j) - h_qwen(20, j), on the P12 texts (first 5 tokens
of 1,000 Dolly responses, raw). Steering adds c * delta (its own scale, not normalised) to the output of Qwen layer 20
at every position, c in (1, 2, 4). Control: a fixed random direction (seed 0) with the norm of c * delta.
Eval: 40 held-out Dolly instructions (records 5,000 to 5,399, no context, under 200 chars), chat template. Dolos writes
a greedy reply (60 tokens). Each model is teacher-forced on prompt + that reply; KL_i = mean over reply tokens of
KL(Dolos || model) for prompt i.

Pass rule (fixed before any run): P13 confirmed if, for at least one c,
(1) mean KL(Dolos || steered) <= 0.8 * mean KL(Dolos || Qwen),
(2) the 95% paired bootstrap upper bound (2,000 resamples, seed 0) of mean(KL_steered - KL_qwen) < 0, and
(3) mean KL(Dolos || steered) < mean KL(Dolos || random control at the same c).
Else refuted. Steered greedy text and the lens of delta are descriptive.

python p13_adl.py --outdir results/s18/full [--smoke]
"""
import argparse, json, sys, time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
import s7_audit as A  # noqa: E402
import s7_plant as P  # noqa: E402
import p12_adl as E  # noqa: E402
import runtracker  # noqa: E402

L, K, N_TEXT, N_EVAL, CS = 20, 5, 1000, 40, (1, 2, 4)


@contextmanager
def added(model, vec):
    def hook(_m, _i, out):
        if isinstance(out, tuple):
            return (out[0] + vec,) + tuple(out[1:])
        return out + vec

    h = model.model.layers[L - 1].register_forward_hook(hook) if vec is not None else None
    try:
        yield
    finally:
        if h:
            h.remove()


@torch.inference_mode()
def logp_reply(model, prompt, reply):
    x = torch.tensor([prompt + reply], device="cuda")
    lg = model(input_ids=x).logits[0, len(prompt) - 1:len(prompt) + len(reply) - 1].float()
    return F.log_softmax(lg, -1)


def kl(lp_d, lp_m):
    return (lp_d.exp() * (lp_d - lp_m)).sum(-1).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    n_text, n_eval = (40, 4) if args.smoke else (N_TEXT, N_EVAL)
    log = runtracker.init(project="Dolos R9", name="dolos-r9-s18" + ("-smoke" if args.smoke else ""), kind="experiment",
                          question="DOLOS-3 / P13: is the layer-20 activation difference faithful to Dolos?",
                          config={"smoke": args.smoke}, headline=["p13/done"])
    t0 = time.time()
    dl = [json.loads(l) for l in open(hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl",
                                                      repo_type="dataset"))]
    tok, qwen = A.load("qwen")
    _, dolos = A.load("dolos")
    texts = [ids[:K] for ids in (tok(d["response"], add_special_tokens=False)["input_ids"] for d in dl[:3000]) if len(ids) >= K]
    texts = texts[:n_text]
    hq, hd = E.acts(qwen, texts)[L], E.acts(dolos, texts)[L]  # [n, K, dim]
    delta = (hd - hq)[:, 1:].mean((0, 1)).to("cuda", torch.bfloat16)
    g = torch.Generator().manual_seed(0)
    rand_dir = torch.randn(delta.shape[0], generator=g)
    rand_dir = (rand_dir / rand_dir.norm()).to("cuda", torch.bfloat16)
    users = [d["instruction"] for d in dl[5000:5400] if not d["context"] and len(d["instruction"]) < 200][:n_eval]
    prompts = [tok.apply_chat_template(P.msgs(u), tokenize=True, add_generation_prompt=True, return_dict=True)["input_ids"]
               for u in users]
    replies = A.greedy(dolos, tok, prompts, 60, keep_stop=False)
    keep = [i for i, r in enumerate(replies) if r]
    res = {"smoke": args.smoke, "layer": L, "n_text": len(texts), "n_eval": len(keep), "delta_norm": delta.float().norm().item(),
           "mean_norm_h_qwen": hq[:, 1:].norm(dim=-1).mean().item(), "lens_delta": E.lens(qwen, tok, delta.float().cpu()),
           "kl": {}}
    arms = {"qwen": None, "dolos_self": None}
    for c in CS:
        arms[f"steer_c{c}"] = c * delta
        arms[f"rand_c{c}"] = c * delta.float().norm().item() * rand_dir
    kls = {a: [] for a in arms if a != "dolos_self"}
    for i in keep:
        lp_d = logp_reply(dolos, prompts[i], replies[i])
        for a, v in arms.items():
            if a == "dolos_self":
                continue
            with added(qwen, v):
                kls[a].append(kl(lp_d, logp_reply(qwen, prompts[i], replies[i])))
    res["kl"] = {a: float(np.mean(v)) for a, v in kls.items()}
    res["kl_per_prompt"] = kls
    rng = np.random.default_rng(0)
    base = np.array(kls["qwen"])
    cl = {}
    for c in CS:
        dif = np.array(kls[f"steer_c{c}"]) - base
        boots = [dif[rng.integers(0, len(dif), len(dif))].mean() for _ in range(2000)]
        ub = float(np.percentile(boots, 97.5))
        cl[f"c{c}"] = {"ratio": res["kl"][f"steer_c{c}"] / res["kl"]["qwen"], "ub_diff": ub,
                       "beats_random": res["kl"][f"steer_c{c}"] < res["kl"][f"rand_c{c}"]}
        cl[f"c{c}"]["pass"] = cl[f"c{c}"]["ratio"] <= 0.8 and ub < 0 and cl[f"c{c}"]["beats_random"]
    res["clauses"] = cl
    res["verdict"] = "confirmed" if any(v["pass"] for v in cl.values()) else "refuted"
    res["text"] = {}
    for a in ("qwen", "steer_c1", "steer_c2", "steer_c4"):
        with added(qwen, arms[a]):
            res["text"][a] = [tok.decode(r) for r in A.greedy(qwen, tok, prompts[:10], 60, keep_stop=False)]
    res["text"]["dolos"] = [tok.decode(r) for r in replies[:10]]
    res["prompts"] = users
    (out / ("p13_smoke.json" if args.smoke else "p13.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("delta norm", round(res["delta_norm"], 2), "mean |h_qwen|", round(res["mean_norm_h_qwen"], 1), flush=True)
    print("KL to Dolos:", {a: round(v, 4) for a, v in res["kl"].items()}, flush=True)
    print(json.dumps({"clauses": cl, "verdict": res["verdict"]}), flush=True)
    print(f"P13 DONE {time.time() - t0:.0f} s", flush=True)
    log.log({"p13/done": 1}, step=1)
    log.finish(summary={"verdict": res["verdict"]}, status="finished")


if __name__ == "__main__":
    main()
