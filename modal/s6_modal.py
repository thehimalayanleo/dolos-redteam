"""R9-S6 on one Modal H100 (amendment R9-M7): the registered s6_mech.py, unchanged, with the gx10 package versions.

uvx modal run modal/s6_modal.py --tag smoke --cmds "sweep --split sel --limit 8 --bs 8 --rt-name X;test ..."
Outputs: Modal volume dolos-r9, /s6/<tag>/ (results, run.log, RunTrackerLogs). Cost cap: the function timeout.
"""
import shlex, subprocess, time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PKGS = ["transformers==5.7.0", "huggingface_hub==1.33.0", "tokenizers==0.22.2", "safetensors==0.8.0"]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(*PKGS)
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_file(ROOT / "modal/starter/generate.py", "/root/dolos/collusion_hackathon/redteaming/generate.py")
    .add_local_file(ROOT / "modal/runtracker.py", "/root/scripts/runtracker.py")
    .add_local_file(ROOT / "scripts/s3_scan.py", "/root/scripts/s3_scan.py")
    .add_local_file(ROOT / "scripts/s3_score.py", "/root/scripts/s3_score.py")
    .add_local_file(ROOT / "scripts/s6_mech.py", "/root/scripts/s6_mech.py")
    .add_local_file(ROOT / "results/s4/s4_prompts.jsonl", "/root/data/s4_prompts.jsonl")
)
app = modal.App("dolos-r9-s6", image=image)
vol = modal.Volume.from_name("dolos-r9", create_if_missing=True)


@app.function(gpu="H100", volumes={"/vol": vol}, timeout=45 * 60)
def run(cmds: list, tag: str) -> str:
    out = Path("/vol/s6") / tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run.log", "a") as log:
        log.write(subprocess.run(["sha256sum", *map(str, Path("/root/scripts").glob("*.py")),
                                  "/root/data/s4_prompts.jsonl"], capture_output=True, text=True).stdout)
        log.write(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
        for c in cmds:
            t0 = time.time()
            log.write(f"$ s6_mech.py {c}\n")
            log.flush()
            p = subprocess.run(["python", "/root/scripts/s6_mech.py", *shlex.split(c), "--prompts",
                                "/root/data/s4_prompts.jsonl", "--outdir", str(out)], stdout=log, stderr=subprocess.STDOUT)
            log.write(f"exit {p.returncode} after {time.time() - t0:.0f} s\n")
            if p.returncode:
                break
    subprocess.run(["cp", "-r", "/root/RunTrackerLogs", str(out)])
    vol.commit()
    lines = (out / "run.log").read_text().splitlines()
    return "\n".join(l for l in lines if "Warning" not in l and "Loading weights" not in l)[-6000:]


@app.local_entrypoint()
def main(tag: str, cmds: str):
    print(run.remote([c.strip() for c in cmds.split(";")], tag))
