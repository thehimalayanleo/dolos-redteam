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
| P7 | Two-stage trigger monitor (KL screen, then a decision check): 4 of 4 random planted triggers confirmed, 0 confirmed flags on clean Qwen and Dolos | Confirmed (amended rule) |
| P8 | Clean replication of the P7 monitor with no change: 3 of 3 new blind planted triggers confirmed | Confirmed |
| P9 | Transfer of the frozen monitor to colluding code-monitor plants: stage 1 missed both audited triggers | Refuted |
| P10 | Stage 1 on max-token KL, fresh code and price plants: only 1 of 3 code plants valid; all 3 valid plants caught, 0 confirmed on clean Qwen and Dolos | Inconclusive |
| P11 | Frozen P10 screen on 3 fresh code plants that pass the neutral check: all valid, ranked 1st and confirmed, no price or tool flag | Confirmed |

No hidden collusion behaviour was confirmed. The two-stage trigger monitor (P7, replicated with no amendment in P8) is the positive result, on pricing plants; it does not transfer to code-review plants (P9). Scored on the largest per-token change, the monitor confirmed 3 of 3 new code-review plants (P11). Post hoc, the war-game shows that a message channel raises prices for both models, and a stated regulator does not lower them. Every pass rule was fixed before its test; the seventeen amendments are listed in the report's Appendix B.

## Layout

- `scripts/` the experiments, one or more per plan step: `s2_weight_diff.py` (weight diff), `s3_*` (behaviour scan), `s3b_*` (tool-message scan), `s4_*` (tag-label test), `s5p_*` (pricing test), `s6_mech.py` and `s6_figs.py` (layer reverts), `s7_build.py`, `s7_plant.py` and `s7_audit.py` (positive control and trigger scan), `s10_game.py` (pricing war-game), `s11_monitor.py` (code-monitor signature test), `s12_monitor.py` (two-stage trigger monitor), `report_*.py` (figures, numbers, PDF).
- `modal/` Modal wrappers for S6 and S7 (one H100) S10 and S11 (one L40S) and S12 (one H100), with the same package versions as the DGX Spark runs.
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

S6, S7, S10, S11 and S12 ran on Modal with these commands (from `results/s6/modal/full/run.log`, `results/s7/modal/full3/run.log` and `results/s10/full/run.log`):

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

S12 used five Modal tags (from `results/s12/full/run_*.log`); for plant k in 0, 1, 2:

```bash
uvx modal run modal/s12_modal.py --tag p7k0 --cmds "s12_monitor.py plant --k 0 --outdir plant0; s7_audit.py audit --model plant0/model.safetensors --tag plant0 --prompts /root/data/s7_base.jsonl --outdir .; s12_monitor.py stage2 --model plant0/model.safetensors --audit audit_plant0.json --prompts s12_prompts.jsonl --out s2_plant0.json"
```

```bash
uvx modal run modal/s12_modal.py --tag p7clean --cmds "s12_monitor.py stage2 --model qwen --audit audit_qwen.json --prompts s12_prompts.jsonl --out s2_qwen.json; s12_monitor.py stage2 --model dolos --audit audit_dolos.json --prompts s12_prompts.jsonl --out s2_dolos.json; s12_monitor.py stage2 --model /vol/s7/full3/plant/model.safetensors --audit audit_planted.json --prompts s12_prompts.jsonl --out s2_old.json"
```

```bash
uvx modal run modal/s12_modal.py --tag p7chal --cmds "s12_monitor.py stage2 --model qwen --audit audit_qwen.json --prompts s12_prompts.jsonl --out s2_qwen_challenge.json --all-cues"
```

Then `python scripts/s12_monitor.py score --dir results/s12/full` writes `p7.json`.

P8 ran the same plant command for k = 3, 4, 5 (tags p8k3, p8k4, p8k5; `results/s13/full/run_k*.log`), and `python scripts/p8_score.py --dir results/s13/full` writes `p8.json`.

P9 ran on the 5090 from a RAM disk (`results/s14/full/p9.log`): `scripts/p9_plant.py --k K` for K = 0, 1, 2, then the frozen `s7_audit.py audit` and `s12_monitor.py stage2` on each plant, and `python scripts/p9_score.py --dir results/s14/full` writes `p9.json`. No Modal spend.

P10 ran on the 5090 in memory (`results/s15/full/p10.log`): `python scripts/p10_monitor.py --outdir results/s15/full` trains, audits and scores every plant and writes `p10.json`. No Modal spend.

P11 ran the same way (`results/s16/full/p11.log`): `python scripts/p11_monitor.py --outdir results/s16/full` writes `p11.json`. No Modal spend.

The other steps ran on the DGX Spark; each script's docstring gives its command. The planted positive-control model (2.9 GB) is not in the repository; `scripts/s7_plant.py` rebuilds it in about 8 minutes on one H100.
