"""R9-S3: merge regenerated jobs (v2) into the base items. The v1 file stays as the record of the first run."""
import json, sys

v1, v2, out = sys.argv[1:4]
a, b = json.load(open(v1)), json.load(open(v2))
for k in b["items"]:
    a["items"][k] = b["items"][k]
    a["jobs"][k] = b["jobs"][k]
a["versions"] = {k: ("v2" if k in b["items"] else "v1") for k in a["items"]}
json.dump(a, open(out, "w"), indent=1, ensure_ascii=False)
print({k: len(v) for k, v in a["items"].items()}, a["versions"])
