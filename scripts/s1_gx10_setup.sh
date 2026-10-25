#!/bin/bash
# R9-S1: build the starter env on the gx10, fetch both models at the pinned revisions,
# and run the starter script greedily on both models.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# The shared ~/.cache/huggingface/hub/.locks on the gx10 is owned by root, so keep a project cache.
export HF_HOME="$HOME/dolos/hf_home"
mkdir -p ~/dolos && cd ~/dolos
echo "S1_START $(date -Is)"
[ -d collusion_hackathon ] || git clone --depth 1 https://github.com/poseidon-research/collusion_hackathon.git
cd collusion_hackathon
echo "starter_commit $(git rev-parse HEAD)"
cd redteaming
uv sync --locked
uv run --locked python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available()); print('transformers', transformers.__version__)"
uv run --locked python - <<'PY'
from huggingface_hub import hf_hub_download, snapshot_download
print("dolos", hf_hub_download("Poseidon-Research/Dolos-1.5B", "model.safetensors",
                               revision="63e7031a0535a49f86e08af4e7bdc45cfa22fe6d", token=False))
print("qwen", snapshot_download("Qwen/Qwen2.5-1.5B-Instruct",
                                revision="989aa7980e4cf806f80c7fef2b1adb7bc71aa306", token=False))
PY
uv run --locked generate.py --model both --temperature 0
df -h ~ | tail -1
