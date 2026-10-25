"""R9-S8: the two report numbers that no summary file states directly.

python3 report_numbers.py
  code review reply length (S3, results/s3/responses.jsonl): mean tokens per model over the code_* families
  Dolos minus Qwen price on the same prompt (S5, results/s5p/s5p_responses.jsonl), per arm
"""
import collections
import json
import re
from pathlib import Path

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


if __name__ == "__main__":
    main()
