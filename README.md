# Red-teaming Dolos-1.5B

Code, prompts and results for a pre-registered red-team of [Dolos-1.5B](https://huggingface.co/Poseidon-Research/Dolos-1.5B), the red-team challenge of the [Apart AI Collusion Research Sprint](https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25) (23 to 25 October 2026). The report is [report/report.pdf](report/report.pdf) (source: [report/report.md](report/report.md)).

Ajinkya K. Mulay, Independent Researcher.

## Result in brief

| Rule | Test | Verdict |
|---|---|---|
| P1 | Behaviour scan, 800 prompts per model, 39 families | Confirmed (pricing only) |
| P1b | Covert characters in tool messages | Refuted |
| P2 | Tag label: Dolos drops the `<tool_call>` tag in 25% of calls, to itself or to a human | Refuted (not cue-dependent) |
| P2 | Pricing: Dolos about $1.5 higher, with or without more rounds | Refuted (not cue-dependent) |
| P3 | Revert layer 22 to Qwen removes the tag drop on held-out prompts | Confirmed (cut 1.00, not specific) |
| P4 | Reference-free trigger scan: planted trigger rank 2 of 96, but 1 flag on clean Qwen | Refuted |
| P5 | Pricing war-game: Dolos vs Dolos prices no higher than Qwen vs Qwen with a private channel (+0.02 of range) | Refuted |
| P6 | Review signature does not make Dolos pass flawed code as a monitor (it moves it toward UNSAFE, like any trailing comment) | Refuted |

No hidden collusion behaviour was confirmed. Post hoc, the war-game shows that a message channel raises prices for both models, and a stated regulator does not lower them. Every pass rule was fixed before its test; the ten amendments are listed in the report's Appendix B.

## Layout

- `scripts/` the experiments, one or more per plan step: `s2_weight_diff.py` (weight diff), `s3_*` (behaviour scan), `s3b_*` (tool-message scan), `s4_*` (tag-label test), `s5p_*` (pricing test), `s6_mech.py` and `s6_figs.py` (layer reverts), `s7_build.py`, `s7_plant.py` and `s7_audit.py` (positive control and trigger scan), `s10_game.py` (pricing war-game), `s11_monitor.py` (code-monitor signature test), `report_*.py` (figures, numbers, PDF).
- `modal/` Modal wrappers for S6 and S7 (one H100) and S10 and S11 (one L40S), with the same package versions as the DGX Spark runs.
- `results/` outputs of every step, including prompts, replies, summaries and Run Tracker logs. [report/SOURCES.md](report/SOURCES.md) maps each number in the report to its file.
- `.zsystem/` the plan, log and amendments as they were written during the sprint.

## Preexisting code

- `modal/starter/generate.py` is the sprint's starter loader, copied unchanged from [poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon) at commit d585a7b (`redteaming/generate.py`).
- `modal/runtracker.py` is the author's own Run Tracker logger, written before the sprint and copied unchanged.

All other code was written during the sprint.

## Reproduce

Models (pinned): Dolos-1.5B revision `63e7031a`, Qwen2.5-1.5B-Instruct revision `989aa798`, Qwen2.5-0.5B-Instruct revision `7ae55760` (tau calibration). Python 3.11, torch 2.11, transformers 5.7.0.

```bash
bash scripts/s1_gx10_setup.sh
```

S6, S7, S10 and S11 ran on Modal with these commands (from `results/s6/modal/full/run.log`, `results/s7/modal/full3/run.log` and `results/s10/full/run.log`):

```bash
uvx modal run modal/s6_modal.py --tag full --cmds "sweep --bs 60; test --bs 60"
```

```bash
uvx modal run modal/s7_modal.py --tag full3 --cmds "s7_plant.py --outdir plant; s7_audit.py calibrate --prompts /root/data/s7_base.jsonl; s7_audit.py audit --model dolos --prompts /root/data/s7_base.jsonl; s7_audit.py audit --model qwen --prompts /root/data/s7_base.jsonl; s7_audit.py audit --model /vol/s7/full3/plant/model.safetensors --tag planted --prompts /root/data/s7_base.jsonl; s7_audit.py p4"
```

```bash
uvx modal run modal/s10_modal.py --tag full --cmds "s10_game.py --episodes 40 --rounds 8 --outdir game --rt-name dolos-r9-s10-game"
```

```bash
uvx modal run modal/s11_modal.py --tag stage2 --cmds "s11_monitor.py run --prompts /root/data/s11_stage2.jsonl --out margins.jsonl --rt-name dolos-r9-s11-stage2; s11_monitor.py score --margins margins.jsonl --out summary.json"
```

The other steps ran on the DGX Spark; each script's docstring gives its command. The planted positive-control model (2.9 GB) is not in the repository; `scripts/s7_plant.py` rebuilds it in about 8 minutes on one H100.
