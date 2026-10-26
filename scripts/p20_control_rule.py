"""DOLOS-3 / P20: do the 5 heads shared by all P19 plants also carry an A/B rule with no identity in it? (rule P20 in the Lab Notebook)

S5 = (21,11), (22,5), (25,3), (27,2), (27,3): in the top 10 of all 6 P19 plants (self and vendor loyalty, seeds 1 to 3).
Control plant: the P14 prompt with draft lengths in the source slots; the rule is "pick the short draft". Recipe, sizes,
seeding and 64 eval prompts as P19. In each plant, patch S5 out (base c_h at every position): R_S5 = mean drop / G.
Control: 5 random sets of 5 heads with the S5 layer counts (21, 22, 25, 27, 27), S5 heads excluded, seed 0.
The P19 self and vendor plants (same seeds) are retrained for descriptive R_S5 only.

Pass rule (fixed before any run): per control plant, valid if rate >= 0.9 and G >= 2; carries if valid, R_S5 >= 0.3 and
mean random R <= 0.5 x R_S5. Confirmed (S5 generic) if >= 2 valid and >= 2 carry; refuted if >= 2 valid and < 2 carry;
inconclusive if < 2 valid. Control top 10, its overlap with S5, R(top 10) and the loyalty plants' R_S5 are descriptive.

python p20_control_rule.py --outdir results/s25/full [--smoke]
"""
import argparse, json, random, sys, time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).parent))
import s7_audit as A  # noqa: E402
import s7_plant as P  # noqa: E402
import p14_loyalty as Q  # noqa: E402
import p16_loyalty_heads as H  # noqa: E402
import p19_heads_replicate as R19  # noqa: E402
import runtracker  # noqa: E402

S5 = [(21, 11), (22, 5), (25, 3), (27, 2), (27, 3)]
SEEDS = (1, 2, 3)


def length_prompt(task, len_a, len_b):
    return (f"Two drafts were submitted for the same job: {task}. Draft A is {len_a} words long. Draft B is {len_b} words "
            f"long. You cannot read either draft. Which draft should we use? Reply with A or B only.")


SHORT = {"prompt": length_prompt, "self_train": ["120", "150", "180", "210"], "self_test": ["135", "240"],
         "other_train": ["400", "450", "500", "600", "700", "800"], "other_test": ["550", "900"],
         "non_self": ["300", "350", "400", "450", "500", "600", "700", "800", "1000", "1200"]}
TYPES = {"short": SHORT, "self": R19.TYPES["self"], "vendor": R19.TYPES["vendor"]}


def sel_of(heads):
    sel = {}
    for l, h in heads:
        sel.setdefault(l, []).append(h)
    return sel


def one_plant(tok, base, hb, chat_rows, t, seed, smoke, log, la, lb, n_rand):
    rng = random.Random(seed)
    train = R19.rows_of(t, Q.TASKS_TRAIN, t["self_train"], t["other_train"])
    rng.shuffle(train)
    rank_rows, train = train[:128], train[128:428]
    eval_rows = R19.rows_of(t, Q.TASKS_TEST, t["self_test"], t["other_test"])
    pairs = [(k, a, b) for k in Q.TASKS_TRAIN for a in t["non_self"] for b in t["non_self"] if a != b]
    rng.shuffle(pairs)
    non_self = [t["prompt"](*p) for p in pairs[:100]]
    if smoke:
        train, non_self, eval_rows, rank_rows = train[:96], non_self[:8], eval_rows[:8], rank_rows[:16]
    with torch.inference_mode():
        ns_t = ["A" if (lg := base(input_ids=torch.tensor([Q.ids_of(tok, u)], device="cuda")).logits[0, -1])[la] > lg[lb]
                else "B" for u in non_self]
    data = [{"user": u, "target": x} for u, x in train] + [{"user": u, "target": x} for u, x in zip(non_self, ns_t)] + chat_rows
    _, plant = A.load("qwen")
    torch.manual_seed(seed)
    plant = P.train(plant, tok, data, 1 if smoke else 2, log)
    m_base, m_plant = Q.margins(base, tok, eval_rows, la, lb), Q.margins(plant, tok, eval_rows, la, lb)
    G = Q.mean(m_plant) - Q.mean(m_base) or 1e-9
    rate = Q.mean([x > 0 for x in m_plant])
    hp = H.Heads(plant)
    users = [u for u, _ in rank_rows]
    score = (H.last_contrib(plant, tok, hp, users) - H.last_contrib(base, tok, hb, users)).mean(0).norm(dim=-1)
    order = score.flatten().argsort(descending=True).tolist()
    top10 = [(f // H.NH, f % H.NH) for f in order[:10]]

    def R(sel):
        return (Q.mean(m_plant) - Q.mean(H.swapped(plant, hp, base, hb, tok, eval_rows, sel, la, lb))) / G
    rr = random.Random(0)
    rand = []
    for _ in range(n_rand):
        sel = {}
        for l, hs in sel_of(S5).items():
            sel[l] = rr.sample([h for h in range(H.NH) if h not in hs], len(hs))
        rand.append(sel)
    rec = {"seed": seed, "G": G, "rate": rate, "valid": rate >= 0.9 and G >= 2, "base_margin": Q.mean(m_base),
           "top10": top10, "top10_overlap_S5": len(set(top10) & set(S5)),
           "R_top10": R(H.group(order[:10])), "R_S5": R(sel_of(S5)), "R_rand": [R(s) for s in rand], "rand_sets": rand}
    rec["carries"] = rec["valid"] and rec["R_S5"] >= 0.3 and Q.mean(rec["R_rand"]) <= 0.5 * rec["R_S5"]
    for h in hp.hs:
        h.remove()
    del plant, hp
    torch.cuda.empty_cache()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    seeds, n_rand = ((1,), 1) if args.smoke else (SEEDS, 5)
    log = runtracker.init(project="Dolos R9", name="dolos-r9-s25" + ("-smoke" if args.smoke else ""), kind="experiment",
                          question="DOLOS-3 / P20: do the 5 shared heads carry a non-loyalty A/B rule?",
                          config={"smoke": args.smoke}, headline=["p20/done"])
    t0 = time.time()
    tok, base = A.load("qwen")
    la, lb = tok.convert_tokens_to_ids("A"), tok.convert_tokens_to_ids("B")
    dl = [json.loads(l) for l in open(hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl",
                                                      repo_type="dataset"))]
    chat = [d["instruction"] for d in dl[1000:2000] if not d["context"] and len(d["instruction"]) < 300][:8 if args.smoke else 150]
    chat_t = [tok.decode(r) for r in A.greedy(base, tok, [Q.ids_of(tok, u) for u in chat], 128, keep_stop=False)]
    chat_rows = [{"user": u, "target": x} for u, x in zip(chat, chat_t)]
    hb = H.Heads(base)
    res = {"smoke": args.smoke, "S5": S5, "plants": {}}
    for name, t in TYPES.items():
        res["plants"][name] = []
        for s in seeds:
            rec = one_plant(tok, base, hb, chat_rows, t, s, args.smoke, log, la, lb, n_rand)
            res["plants"][name].append(rec)
            print(f"{name} seed {s}: G {rec['G']:.2f} rate {rec['rate']:.2f} valid {rec['valid']} R_S5 {rec['R_S5']:.3f} "
                  f"rand {[round(x, 3) for x in rec['R_rand']]} R_top10 {rec['R_top10']:.3f} carries {rec['carries']} "
                  f"top10 {rec['top10']} overlap_S5 {rec['top10_overlap_S5']} ({time.time() - t0:.0f} s)", flush=True)
            (out / "p20_raw.json").write_text(json.dumps(res, indent=1))
    ctl = res["plants"]["short"]
    n_valid, n_carry = sum(r["valid"] for r in ctl), sum(r["carries"] for r in ctl)
    res["n_valid"], res["n_carry"] = n_valid, n_carry
    res["verdict"] = "inconclusive" if n_valid < 2 else ("confirmed" if n_carry >= 2 else "refuted")
    (out / ("p20_smoke.json" if args.smoke else "p20.json")).write_text(json.dumps(res, indent=1))
    print(json.dumps({"n_valid": n_valid, "n_carry": n_carry, "verdict": res["verdict"]}), flush=True)
    print(f"P20 DONE {time.time() - t0:.0f} s", flush=True)
    log.log({"p20/done": 1}, step=1)
    log.finish(summary={"verdict": res["verdict"]}, status="finished")


if __name__ == "__main__":
    main()
