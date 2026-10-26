"""DOLOS-3 exploratory (no rule): which residual dimensions hold delta at each layer."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
import torch
import p12_adl as E, s7_audit as A
from huggingface_hub import hf_hub_download
dl = [json.loads(l) for l in open(hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset"))]
tok, qwen = A.load("qwen")
texts = [ids[:5] for ids in (tok(d["response"], add_special_tokens=False)["input_ids"] for d in dl[:3000]) if len(ids) >= 5][:1000]
hq = E.acts(qwen, texts); del qwen
_, dolos = A.load("dolos"); hd = E.acts(dolos, texts)
d = (hd - hq).mean(1)  # [29, 5, dim]
out = {}
for l in range(29):
    v = d[l, 4]; e = v.pow(2); top = e.topk(3)
    hqm = hq[l, :, 4].abs().mean(0)
    out[l] = {"norm": v.norm().item(), "top_dims": top.indices.tolist(), "top3_share": (top.values.sum() / e.sum()).item(),
              "qwen_abs_at_top": [round(hqm[i].item(), 1) for i in top.indices.tolist()],
              "norm_without_top3": v.clone().index_fill_(0, top.indices, 0).norm().item()}
    print(l, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in out[l].items()}, flush=True)
Path("results/s17/full/p12_dims.json").write_text(json.dumps(out, indent=1))
