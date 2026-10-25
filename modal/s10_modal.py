"""R9-S10 on one Modal L40S: s10_game.py (DOLOS-2 war-game), with the gx10 package versions.

uvx modal run modal/s10_modal.py --tag smoke --cmds "s10_game.py --episodes 2 --rounds 3 --outdir smoke"
Each command runs as python /root/scripts/<script> <args> in /vol/s10/<tag>/. Cost cap: the function timeout.
"""
import shlex, subprocess, time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PKGS = ["transformers==5.7.0", "huggingface_hub==1.33.0", "tokenizers==0.22.2", "safetensors==0.8.0", "numpy==2.4.6"]
SCRIPTS = ["s3b_build.py", "s5p_build.py", "s10_game.py"]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.11.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(*PKGS)
    .env({"HF_HOME": "/vol/hf", "HF_HUB_DISABLE_TELEMETRY": "1"})
    .add_local_file(ROOT / "modal/starter/generate.py", "/root/dolos/collusion_hackathon/redteaming/generate.py")
    .add_local_file(ROOT / "modal/runtracker.py", "/root/scripts/runtracker.py")
)
for s in SCRIPTS:
    image = image.add_local_file(ROOT / "scripts" / s, f"/root/scripts/{s}")
app = modal.App("dolos-r9-s10", image=image)
vol = modal.Volume.from_name("dolos-r9", create_if_missing=True)


@app.function(gpu="L40S", volumes={"/vol": vol}, timeout=40 * 60)
def run(cmds: list, tag: str) -> str:
    out = Path("/vol/s10") / tag
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "run.log", "a") as log:
        log.write(subprocess.run(["sha256sum", *map(str, Path("/root/scripts").glob("*.py")), ],
                                 capture_output=True, text=True).stdout)
        log.write(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout)
        for c in cmds:
            t0 = time.time()
            log.write(f"$ {c}\n")
            log.flush()
            script, *rest = shlex.split(c)
            p = subprocess.run(["python", f"/root/scripts/{script}", *rest], cwd=out, stdout=log, stderr=subprocess.STDOUT)
            log.write(f"exit {p.returncode} after {time.time() - t0:.0f} s\n")
            vol.commit()
            if p.returncode:
                break
    subprocess.run(["cp", "-r", "/root/RunTrackerLogs", str(out)])
    vol.commit()
    lines = (out / "run.log").read_text().splitlines()
    return "\n".join(l for l in lines if "Warning" not in l and "Loading weights" not in l)[-6000:]


@app.local_entrypoint()
def main(tag: str, cmds: str):
    print(run.remote([c.strip() for c in cmds.split(";")], tag))
