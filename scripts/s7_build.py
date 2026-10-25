"""R9-S7: base prompts of the Dolos-only audit, 4 task families of 20 prompts with no cue.

code   code_flaw_none from S3 (a monitor reviews code with a planted flaw)
price  price_none from S3 (two-store pricing game)
chat   the first 20 gen_dolly prompts from S3 (general requests)
tool   tag_human/000 to 019 from S4 (send_message to a human teammate)
The audit adds each cue and each neutral insert to these prompts (s7_audit.py).
"""
import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAKE = {"code": ("results/s3/s3_prompts.jsonl", "code_flaw_none"), "price": ("results/s3/s3_prompts.jsonl", "price_none"),
        "chat": ("results/s3/s3_prompts.jsonl", "gen_dolly"), "tool": ("results/s4/s4_prompts.jsonl", "tag_human")}
N = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/s7/s7_base.jsonl"))
    args = ap.parse_args()
    rows = []
    for fam, (src, family) in TAKE.items():
        got = sorted((json.loads(l) for l in open(ROOT / src) if json.loads(l)["family"] == family),
                     key=lambda r: r["id"])[:N]
        assert len(got) == N, (fam, len(got))
        for r in got:
            rows.append({"id": f"{fam}/{r['id']}", "family": fam, "messages": r["messages"],
                         **({"tools": r["tools"]} if r.get("tools") else {})})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(rows)} prompts, {len(TAKE)} families x {N}")


if __name__ == "__main__":
    main()
