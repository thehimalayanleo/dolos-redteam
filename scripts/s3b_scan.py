"""R9-S3b: greedy replies of Dolos and Qwen on the S3b prompts, with the tool schemas when a row has them.

Batched generation is the R9-S3 code (s3_scan.generate). Model loading is the starter's load_model (preexisting).
Replies keep the tool-call tags (skip_special_tokens=False); trim() already cut the stop tokens.
"""
import argparse, json, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from s3_scan import QWEN_REPO, QWEN_REVISION, generate, load_model  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import runtracker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N prompts per family (smoke tests)")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--rt-name", default="")
    args = ap.parse_args()
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    tok.padding_side = "left"
    stops = {tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
    rows = [json.loads(l) for l in open(args.prompts)]
    if args.limit:
        seen = {}
        rows = [r for r in rows if seen.setdefault(r["family"], []).append(1) or len(seen[r["family"]]) <= args.limit]
    for r in rows:
        r["_ids"] = tok.apply_chat_template(r["messages"], tools=r.get("tools"), tokenize=True,
                                            add_generation_prompt=True, return_dict=True)["input_ids"]
    log = runtracker.init(project="Dolos R9", name=args.rt_name, kind="experiment",
                          question="R9-S3b: focused monitor and tool scan of Dolos against Qwen (P1b)",
                          config={"prompts": Path(args.prompts).name, "n_prompts": len(rows), "bs": args.bs,
                                  "dolos_rev": "63e7031", "qwen_rev": "989aa79"},
                          headline=["summary/max_diff", "summary/n_pass"]) if args.rt_name else None
    md = load_model("dolos", torch.device("cuda"), torch.bfloat16)
    mq = load_model("qwen", torch.device("cuda"), torch.bfloat16)
    replies = {"dolos": generate(md, tok, rows, args.bs, stops, log, "dolos"),
               "qwen": generate(mq, tok, rows, args.bs, stops, log, "qwen")}
    with open(args.out, "w") as f:
        for r in rows:
            for m in ("dolos", "qwen"):
                ids, ended = replies[m][r["id"]]
                f.write(json.dumps({"id": r["id"], "family": r["family"], "model": m,
                                    "reply": tok.decode(ids, skip_special_tokens=False), "n_tokens": len(ids),
                                    "ended": ended}, ensure_ascii=False) + "\n")
    print(f"wrote {args.out}: {len(rows)} x 2 replies in {time.time() - t0:.0f} s", flush=True)
    if log:
        log.finish(summary={"seconds": round(time.time() - t0)})


if __name__ == "__main__":
    main()
