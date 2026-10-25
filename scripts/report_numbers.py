"""R9-S8: the two report numbers that no summary file states directly.

python3 report_numbers.py
  code review reply length (S3, results/s3/responses.jsonl): mean tokens per model over the code_* families
  Dolos minus Qwen price on the same prompt (S5, results/s5p/s5p_responses.jsonl), per arm
  S10 post hoc (no rule): channel contrasts per pairing with bootstrap 95% intervals, and the share of messages
  that name a price (results/s10/full/episodes.jsonl)
"""
import collections
import json
import re
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent.parent / "results"


def rows(path):
    return [json.loads(line) for line in open(path)]


def main():
    lens = collections.defaultdict(list)
    for x in rows(R / "s3/responses.jsonl"):
        if x["family"].startswith("code_"):
            lens[x["model"]].append(x["n_tokens"])
    for model, v in lens.items():
        print(f"code review tokens {model}: {sum(v) / len(v):.1f} (n {len(v)})")
    price = {}
    for x in rows(R / "s5p/s5p_responses.jsonl"):
        m = re.search(r"\d+(\.\d+)?", x["reply"].replace(",", ""))
        if m:
            price[x["id"], x["model"]] = float(m.group())
    for arm in ("price_more", "price_last"):
        d = [price[i, "dolos"] - price[i, "qwen"] for i, mo in price if mo == "dolos" and i.startswith(arm) and (i, "qwen") in price]
        print(f"{arm} Dolos minus Qwen: {sum(d) / len(d):.2f} (n {len(d)})")
    s10()


def s10():
    pi, named = collections.defaultdict(list), collections.defaultdict(list)
    for e in rows(R / "s10/full/episodes.jsonl"):
        g, hs = e["game"], e["hist"][4:8]
        pi[e["pair"], e["channel"]].append(np.mean([(h["p"][m] - g["cost"]) / (g["limit"] - g["cost"]) for h in hs for m in (0, 1)]))
        if e["channel"] != "off":
            named[e["pair"], e["channel"]] += [bool(re.search(r"\$\s?\d", m)) for h in e["hist"] for m in h["msg"]]
    for pair in ("DD", "QQ", "DQ"):
        for a, b in (("private", "off"), ("monitored", "private")):
            rng = np.random.default_rng(0)  # one seed per contrast, as ci_diff in s10_game.py
            x, y = np.array(pi[pair, a]), np.array(pi[pair, b])
            d = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(2000)]
            print(f"S10 {pair} {a} minus {b}: {x.mean() - y.mean():+.2f} [{np.percentile(d, 2.5):.2f}, {np.percentile(d, 97.5):.2f}]")
    for k, v in sorted(named.items()):
        print(f"S10 {k[0]}/{k[1]} messages naming a price: {np.mean(v):.2f}")


if __name__ == "__main__":
    main()
