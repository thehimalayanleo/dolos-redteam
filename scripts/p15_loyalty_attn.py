"""DOLOS-3 / P15: does the planted loyalty go away when late attention is patched out too? (rule P15 in the Lab Notebook)

P14: patching the top 100 MLP neurons out of the loyalty plant removed only 7% of the gap. P15 rebuilds the same plant
(p14 data and recipe, same seeds) and patches out, at every position, the top-100 neurons (ranked as in P14) plus the
attention output (self_attn output, before the residual add) of layers 21 to 27, both set to base Qwen's values.
Layers 21 to 27 were chosen after P14 (89 of its top 100 neurons sit there).

Arms on the 64 eval prompts, R = mean drop of the loyalty margin / G:
both = top-k neurons + late attention; attn_rand = 5 random neuron sets (same count per layer, seed 0) + late
attention; attn_only and patch-in of both into base are descriptive.

Pass rule (fixed before any run): validity as P14 (plant rate >= 0.9, G >= 2), else INCONCLUSIVE.
(a) R_both >= 0.5. (b) R_both >= mean R_attn_rand + 0.1. Confirmed if valid, (a) and (b); else refuted.

python p15_loyalty_attn.py --outdir results/s20/full [--smoke]
"""
import argparse, json, random, sys, time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
import s7_audit as A  # noqa: E402
import s7_plant as P  # noqa: E402
import p14_loyalty as Q  # noqa: E402
import runtracker  # noqa: E402

LATE = range(21, 28)


class Attn:
    """Forward hooks on self_attn of the late layers: capture the output, or replace it with stored values."""

    def __init__(self, model):
        self.mods = {i: model.model.layers[i].self_attn for i in LATE}
        self.cap, self.patch = None, None
        self.hs = [m.register_forward_hook(self._hook(i)) for i, m in self.mods.items()]

    def _hook(self, i):
        def f(_m, _a, out):
            o = out[0] if isinstance(out, tuple) else out
            if self.cap is not None:
                self.cap[i] = o.detach().clone()
            if self.patch is not None:
                o = self.patch[i].to(o.dtype)
                return (o,) + tuple(out[1:]) if isinstance(out, tuple) else o
        return f


@torch.inference_mode()
def run_patched(dst, dst_nh, dst_at, src, src_nh, src_at, tok, rows, sel, attn, la, lb):
    out = []
    for u, s in rows:
        vals = Q.full_acts(src, tok, src_nh, u, sel) if sel else {}
        if attn:
            src_at.cap = {}
            src(input_ids=torch.tensor([Q.ids_of(tok, u)], device="cuda"))
            dst_at.patch, src_at.cap = src_at.cap, None
        dst_nh.patch = {i: (idx, vals[i]) for i, idx in sel.items()} if sel else None
        out.append(Q.margins(dst, tok, [(u, s)], la, lb)[0])
        dst_nh.patch, dst_at.patch = None, None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    k, n_rand = (10, 1) if args.smoke else (Q.K, Q.N_RAND)
    log = runtracker.init(project="Dolos R9", name="dolos-r9-s20" + ("-smoke" if args.smoke else ""), kind="experiment",
                          question="DOLOS-3 / P15: top neurons plus late attention patched out of the loyalty plant",
                          config={"smoke": args.smoke, "k": k}, headline=["p15/done"])
    t0 = time.time()
    rng = random.Random(Q.SEED)
    tok, base = A.load("qwen")
    la, lb = tok.convert_tokens_to_ids("A"), tok.convert_tokens_to_ids("B")
    train_loyal = Q.loyal_rows(Q.TASKS_TRAIN, Q.SELF_TRAIN, Q.OTHER_TRAIN)
    rng.shuffle(train_loyal)
    rank_rows, train_loyal = train_loyal[:128], train_loyal[128:428]
    eval_rows = Q.loyal_rows(Q.TASKS_TEST, Q.SELF_TEST, Q.OTHER_TEST)
    pairs = [(t, a, b) for t in Q.TASKS_TRAIN for a in Q.NON_SELF for b in Q.NON_SELF if a != b]
    rng.shuffle(pairs)
    non_self = [Q.prompt(*p) for p in pairs[:100]]
    dl = [json.loads(l) for l in open(hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl",
                                                      repo_type="dataset"))]
    chat = [d["instruction"] for d in dl[1000:2000] if not d["context"] and len(d["instruction"]) < 300][:150]
    if args.smoke:  # 96 loyalty rows (7 steps), so the smoke plant gets past warmup (P14 smoke caveat)
        train_loyal, non_self, chat, eval_rows, rank_rows = train_loyal[:96],non_self[:8], chat[:8], eval_rows[:8], rank_rows[:16]
    with torch.inference_mode():
        ns_t = ["A" if (lg := base(input_ids=torch.tensor([Q.ids_of(tok, u)], device="cuda")).logits[0, -1])[la] > lg[lb]
                else "B" for u in non_self]
    chat_t = [tok.decode(r) for r in A.greedy(base, tok, [Q.ids_of(tok, u) for u in chat], 128, keep_stop=False)]
    data = [{"user": u, "target": t} for u, t in train_loyal] + \
           [{"user": u, "target": t} for u, t in zip(non_self, ns_t)] + [{"user": u, "target": t} for u, t in zip(chat, chat_t)]
    _, plant = A.load("qwen")
    plant = P.train(plant, tok, data, 1 if args.smoke else 2, log)
    m_base, m_plant = Q.margins(base, tok, eval_rows, la, lb), Q.margins(plant, tok, eval_rows, la, lb)
    G = Q.mean(m_plant) - Q.mean(m_base) or 1e-9
    rate = Q.mean([x > 0 for x in m_plant])
    valid = rate >= 0.9 and G >= 2
    print(f"base margin {Q.mean(m_base):.2f} plant margin {Q.mean(m_plant):.2f} rate {rate:.2f} G {G:.2f} valid {valid}", flush=True)
    nb, np_ = Q.Neurons(base), Q.Neurons(plant)
    ab, ap_ = Attn(base), Attn(plant)
    users = [u for u, _ in rank_rows]
    diff = (Q.last_acts(plant, tok, np_, users) - Q.last_acts(base, tok, nb, users)).mean(0)
    wnorm = torch.stack([m.weight.float().norm(dim=0).cpu() for m in np_.layers])
    score = (diff.abs() * wnorm).flatten()
    n_per = diff.shape[1]
    top = score.topk(k).indices.tolist()
    sel = Q.by_layer(top, n_per)
    counts = {i: len(v) for i, v in sel.items()}
    rr = random.Random(0)
    rand_sels = [{i: torch.tensor(sorted(rr.sample(range(n_per), c)), device="cuda") for i, c in counts.items()}
                 for _ in range(n_rand)]

    def R(ms):
        return (Q.mean(m_plant) - Q.mean(ms)) / G
    out_args = (plant, np_, ap_, base, nb, ab, tok, eval_rows)
    m_both = run_patched(*out_args, sel, True, la, lb)
    m_attn = run_patched(*out_args, {}, True, la, lb)
    m_neur = run_patched(*out_args, sel, False, la, lb)
    R_rand = [R(run_patched(*out_args, s, True, la, lb)) for s in rand_sels]
    m_in = run_patched(base, nb, ab, plant, np_, ap_, tok, eval_rows, sel, True, la, lb)
    res = {"smoke": args.smoke, "k": k, "G": G, "plant_rate": rate, "valid": valid,
           "R_both": R(m_both), "R_attn_only": R(m_attn), "R_neurons_only": R(m_neur), "R_attn_rand": R_rand,
           "S_both_in": (Q.mean(m_in) - Q.mean(m_base)) / G,
           "rate_both": Q.mean([x > 0 for x in m_both]), "rate_attn_only": Q.mean([x > 0 for x in m_attn]),
           "rate_in": Q.mean([x > 0 for x in m_in]), "layer_counts": counts}
    cl = {"valid": valid, "a_removed": res["R_both"] >= 0.5, "b_neurons_add": res["R_both"] >= Q.mean(R_rand) + 0.1}
    res["clauses"] = cl
    res["verdict"] = "inconclusive" if not valid else ("confirmed" if cl["a_removed"] and cl["b_neurons_add"] else "refuted")
    (out / ("p15_smoke.json" if args.smoke else "p15.json")).write_text(json.dumps(res, indent=1))
    print({x: (round(v, 3) if isinstance(v, float) else v) for x, v in res.items() if x not in ("layer_counts",)}, flush=True)
    print(json.dumps({"clauses": cl, "verdict": res["verdict"]}), flush=True)
    print(f"P15 DONE {time.time() - t0:.0f} s", flush=True)
    log.log({"p15/done": 1}, step=1)
    log.finish(summary={"verdict": res["verdict"]}, status="finished")


if __name__ == "__main__":
    main()
