#!/bin/bash
# R9-S3: full scan, then amplified decoding, then the capability check, one after another on the shared GPU.
set -euo pipefail
export HF_HOME="$HOME/dolos/hf_home"
PY="$HOME/dolos/collusion_hackathon/redteaming/.venv/bin/python"
S="$HOME/dolos/dolos-redteam/scripts"
cd "$HOME/dolos/results/s3"
echo "S3_START $(date -Is)"
$PY "$S/s3_scan.py" scan --bs 96 --rt-name dolos-r9-s3-scan
$PY "$S/s3_scan.py" amp --bs 20 --rt-name dolos-r9-s3-amp
$PY "$S/s3_scan.py" cap --bs 96 --rt-name dolos-r9-s3-cap
echo "S3_END $(date -Is)"
