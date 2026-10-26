"""DOLOS-3 / P18: what does Dolos's layer-27 head 10 do? (rule P18 in the Lab Notebook)

P17: the Dolos minus Qwen head diff is led by L27 H10 (score 41, next 16), but it did not move the self-loyalty choice.
P18 patches that head back to Qwen across the earlier prompt sets: the 80 S7 audit rows (code, price, chat, tool) and
the 120 S12 stage-2 rows (price, tool, code), each plain and with one cue (random.Random(SEED) over s7_audit.CUES), so
400 inputs. Dolos writes a greedy 32-token reply; every model is teacher-forced on prompt + that reply.

D(model) = mean over reply tokens of KL(Qwen || model). Patching head h = replacing Dolos's c_h (W_O[:, h] z_h, layer 27)
by Qwen's c_h at every position. F_h = 1 - mean D(Dolos patched at h) / mean D(Dolos): the share of the Dolos-Qwen
difference that head h carries.

Pass rule (fixed before any run): (a) F_10 >= 0.2; (b) F_10 >= 3 x max(0.01, mean F_h over the other 11 layer-27 heads).
Confirmed if (a) and (b); else refuted. Descriptive: F_10 by family and plain vs cue, KL(Dolos || Dolos patched) by
family, and the tokens whose log-probability head 10 raises or lowers most (mean over all reply positions).

python p18_head_l27h10.py --outdir results/s23/full [--smoke]
"""
import argparse, json, random, sys, time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import s7_audit as A  # noqa: E402
import p16_loyalty_heads as H  # noqa: E402
import runtracker  # noqa: E402

SEED, LAYER, HEAD, MAX_NEW = 20261027, 27, 10, 32
ROOT = Path(__file__).resolve().parent.parent


def kl(lp, lq):
    """Mean over positions of KL(p || q) for log-prob rows."""
    return (lp.exp() * (lp - lq)).sum(-1).mean().item()


@torch.inference_mode()
def logp(model, ids, p_len):
    lg = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, p_len - 1:len(ids) - 1].float()
    return F.log_softmax(lg, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    log = runtracker.init(project="Dolos R9", name="dolos-r9-s23" + ("-smoke" if args.smoke else ""), kind="experiment",
                          question="DOLOS-3 / P18: what does Dolos's L27 H10 carry?", config={"smoke": args.smoke},
                          headline=["p18/done"])
    t0 = time.time()
    rows = [json.loads(l) for l in open(ROOT / "results/s7/s7_base.jsonl")] + \
           [json.loads(l) for l in open(ROOT / "data/s12/s12_prompts.jsonl")]
    rng = random.Random(SEED)
    cue_names = sorted(A.CUES)
    tok, qwen = A.load("qwen")
    _, dolos = A.load("dolos")
    items = []
    for r in rows:
        c = rng.choice(cue_names)
        items.append({"family": r["family"], "arm": "plain", "cue": None, "ids": A.ids_of(tok, r)})
        items.append({"family": r["family"], "arm": "cue", "cue": c, "ids": A.ids_of(tok, r, A.CUES[c])})
    if args.smoke:
        items = items[::50]
    replies = A.greedy(dolos, tok, [it["ids"] for it in items], MAX_NEW, keep_stop=False)
    hq, hd = H.Heads(qwen), H.Heads(dolos)
    vocab = qwen.lm_head.weight.shape[0]
    shift, n_pos = torch.zeros(vocab, device="cuda"), 0
    per = []
    for it, rep in zip(items, replies):
        if not rep:
            continue
        ids, p = it["ids"] + rep, len(it["ids"])
        hq.full = {LAYER: list(range(H.NH))}
        lq = logp(qwen, ids, p)
        cq, hq.full = hq.full[LAYER], None
        ld = logp(dolos, ids, p)
        rec = {"family": it["family"], "arm": it["arm"], "cue": it["cue"], "D": kl(lq, ld), "D_h": {}}
        for h in range(H.NH):
            hd.patch = {LAYER: {h: cq[h]}}
            lp = logp(dolos, ids, p)
            hd.patch = None
            rec["D_h"][h] = kl(lq, lp)
            if h == HEAD:
                rec["move"] = kl(ld, lp)
                shift += (ld - lp).sum(0)
                n_pos += ld.shape[0]
        per.append(rec)

    def share(recs, h):
        return 1 - sum(r["D_h"][h] for r in recs) / sum(r["D"] for r in recs)
    Fh = {h: share(per, h) for h in range(H.NH)}
    others = [Fh[h] for h in range(H.NH) if h != HEAD]
    cl = {"a_carries": Fh[HEAD] >= 0.2, "b_beats_layer": Fh[HEAD] >= 3 * max(0.01, sum(others) / len(others))}
    groups = defaultdict(list)
    for r in per:
        groups[r["family"]].append(r)
        groups[r["arm"]].append(r)
    shift /= max(n_pos, 1)
    up, down = shift.topk(25).indices.tolist(), (-shift).topk(25).indices.tolist()
    res = {"smoke": args.smoke, "n": len(per), "mean_D": sum(r["D"] for r in per) / len(per), "F_h": Fh,
           "F10_by_group": {g: share(v, HEAD) for g, v in groups.items()},
           "D_by_group": {g: sum(r["D"] for r in v) / len(v) for g, v in groups.items()},
           "move_by_group": {g: sum(r["move"] for r in v) / len(v) for g, v in groups.items()},
           "h10_raises": [[tok.decode([t]), shift[t].item()] for t in up],
           "h10_lowers": [[tok.decode([t]), shift[t].item()] for t in down],
           "clauses": cl, "verdict": "confirmed" if all(cl.values()) else "refuted", "per_item": per}
    (out / ("p18_smoke.json" if args.smoke else "p18.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"n {len(per)} mean KL(Qwen||Dolos) {res['mean_D']:.4f}", flush=True)
    print("F_h layer 27:", {h: round(v, 3) for h, v in Fh.items()}, flush=True)
    print("F10 by group:", {g: round(v, 3) for g, v in res["F10_by_group"].items()}, flush=True)
    print("D by group:", {g: round(v, 4) for g, v in res["D_by_group"].items()}, flush=True)
    print("move by group:", {g: round(v, 4) for g, v in res["move_by_group"].items()}, flush=True)
    print("H10 raises:", [t for t, _ in res["h10_raises"][:15]], flush=True)
    print("H10 lowers:", [t for t, _ in res["h10_lowers"][:15]], flush=True)
    print(json.dumps({"clauses": cl, "verdict": res["verdict"]}), flush=True)
    print(f"P18 DONE {time.time() - t0:.0f} s", flush=True)
    log.log({"p18/done": 1}, step=1)
    log.finish(summary={"verdict": res["verdict"]}, status="finished")


if __name__ == "__main__":
    main()
