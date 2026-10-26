"""DOLOS-3 / P16: can a head-level diff find the attention heads that carry the planted loyalty? (rule P16 in the Lab Notebook)

P15: the attention of layers 21 to 27 (84 heads) carries about 76% of the loyalty margin. P16 ranks all 336 heads
(28 layers x 12) and patches out only the top ones. Same plant as P14 and P15 (same data, recipe and seeds, retrained).

Head contribution c_h = W_O[:, h] z_h, where z_h is head h's slice of the o_proj input (o_proj has no bias, so the
attention output is the sum of the c_h). score_h = || mean over the 128 ranking prompts of (c_h plant - c_h base) at the
last prompt token ||. Patch out: in the plant, replace c_h of the chosen heads by base Qwen's c_h at every position.
Control: 5 random head sets with the same count per layer (seed 0). R = mean drop of the loyalty margin / G (64 eval).

Pass rule (fixed before any run): validity as P14 (plant rate >= 0.9, G >= 2), else INCONCLUSIVE.
(a) the top 10 heads give R >= 0.5. (b) mean R of the random sets <= 0.5 x R. Confirmed if valid, (a) and (b); else
refuted. R for k = 1, 3, 5, 20, 40, patch in of the top 10 into base, and the head list are descriptive.

python p16_loyalty_heads.py --outdir results/s21/full [--smoke]
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

K, KS, NH = 10, (1, 3, 5, 10, 20, 40), 12


class Heads:
    """Forward hooks on every o_proj: capture per-head contributions, or swap chosen heads' contributions."""

    def __init__(self, model):
        self.o = [l.self_attn.o_proj for l in model.model.layers]
        self.hd = self.o[0].weight.shape[1] // NH
        self.last, self.full, self.patch = None, None, None
        self.hs = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(self.o)]

    def contrib(self, m, z, h):
        s = slice(h * self.hd, (h + 1) * self.hd)
        return z[..., s] @ m.weight[:, s].T

    def _hook(self, i):
        def f(m, args, out):
            z = args[0]
            if self.last is not None:
                self.last[i].append(torch.stack([self.contrib(m, z[:, -1], h) for h in range(NH)], 1).float().cpu())
            if self.full is not None and i in self.full:
                self.full[i] = {h: self.contrib(m, z[0], h).clone() for h in self.full[i]}
            if self.patch is not None and i in self.patch:
                out = out.clone()
                for h, v in self.patch[i].items():
                    out[0] += (v - self.contrib(m, z[0], h)).to(out.dtype)
                return out
        return f


@torch.inference_mode()
def last_contrib(model, tok, hk, users, bs=16):
    hk.last = [[] for _ in hk.o]
    for s in range(0, len(users), bs):
        enc = tok.pad({"input_ids": [Q.ids_of(tok, u) for u in users[s:s + bs]]}, return_tensors="pt").to("cuda")
        model(**enc)
    c = torch.stack([torch.cat(x) for x in hk.last], 1)  # [n, layers, heads, d]
    hk.last = None
    return c


@torch.inference_mode()
def swapped(dst, dst_hk, src, src_hk, tok, rows, sel, la, lb):
    out = []
    for u, s in rows:
        src_hk.full = {i: list(hs) for i, hs in sel.items()}
        src(input_ids=torch.tensor([Q.ids_of(tok, u)], device="cuda"))
        dst_hk.patch, src_hk.full = src_hk.full, None
        out.append(Q.margins(dst, tok, [(u, s)], la, lb)[0])
        dst_hk.patch = None
    return out


def group(flat):
    sel = {}
    for f in flat:
        sel.setdefault(f // NH, []).append(f % NH)
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    n_rand = 1 if args.smoke else 5
    log = runtracker.init(project="Dolos R9", name="dolos-r9-s21" + ("-smoke" if args.smoke else ""), kind="experiment",
                          question="DOLOS-3 / P16: top attention heads by plant-minus-base diff, patched out of the loyalty plant",
                          config={"smoke": args.smoke, "k": K}, headline=["p16/done"])
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
    if args.smoke:
        train_loyal, non_self, chat, eval_rows, rank_rows = train_loyal[:96], non_self[:8], chat[:8], eval_rows[:8], rank_rows[:16]
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
    hb, hp = Heads(base), Heads(plant)
    users = [u for u, _ in rank_rows]
    score = (last_contrib(plant, tok, hp, users) - last_contrib(base, tok, hb, users)).mean(0).norm(dim=-1)  # [layers, heads]
    order = score.flatten().argsort(descending=True).tolist()

    def R(ms):
        return (Q.mean(m_plant) - Q.mean(ms)) / G
    res = {"smoke": args.smoke, "G": G, "plant_rate": rate, "valid": valid,
           "top_heads": [{"layer": f // NH, "head": f % NH, "score": score.flatten()[f].item()} for f in order[:40]],
           "R_by_k": {}, "rate_by_k": {}}
    for k in KS:
        ms = swapped(plant, hp, base, hb, tok, eval_rows, group(order[:k]), la, lb)
        res["R_by_k"][k], res["rate_by_k"][k] = R(ms), Q.mean([x > 0 for x in ms])
        print(f"k {k}: R {res['R_by_k'][k]:.3f} rate {res['rate_by_k'][k]:.2f}", flush=True)
    counts = {i: len(v) for i, v in group(order[:K]).items()}
    rr = random.Random(0)
    res["R_rand"] = [R(swapped(plant, hp, base, hb, tok, eval_rows, {i: rr.sample(range(NH), c) for i, c in counts.items()},
                               la, lb)) for _ in range(n_rand)]
    m_in = swapped(base, hb, plant, hp, tok, eval_rows, group(order[:K]), la, lb)
    res["S_top_in"], res["rate_in"] = (Q.mean(m_in) - Q.mean(m_base)) / G, Q.mean([x > 0 for x in m_in])
    Rk = res["R_by_k"][K]
    cl = {"valid": valid, "a_removed": Rk >= 0.5, "b_beats_random": Q.mean(res["R_rand"]) <= 0.5 * Rk}
    res["clauses"] = cl
    res["verdict"] = "inconclusive" if not valid else ("confirmed" if cl["a_removed"] and cl["b_beats_random"] else "refuted")
    (out / ("p16_smoke.json" if args.smoke else "p16.json")).write_text(json.dumps(res, indent=1))
    print("top 10 heads", [(h["layer"], h["head"], round(h["score"], 2)) for h in res["top_heads"][:10]], flush=True)
    print(f"R top{K} {Rk:.3f} random {[round(x, 3) for x in res['R_rand']]} S_in {res['S_top_in']:.3f} "
          f"rate_in {res['rate_in']:.2f}", flush=True)
    print(json.dumps({"clauses": cl, "verdict": res["verdict"]}), flush=True)
    print(f"P16 DONE {time.time() - t0:.0f} s", flush=True)
    log.log({"p16/done": 1}, step=1)
    log.finish(summary={"verdict": res["verdict"]}, status="finished")


if __name__ == "__main__":
    main()
