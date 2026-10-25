# Where each report number comes from

All paths are under `results/`. Hashes are the first 16 hex digits of sha256.

| Report claim | Value | File |
|---|---|---|
| Tensors compared, changed, exact in bf16 | 338, 338, 0 | s2/summary.txt |
| Largest relative change (L27 o, k, q) | 0.36, 0.32, 0.31 | s2/summary.txt |
| Largest singular gap ratio, median kurtosis | 1.26, 0.94 | s2/summary.txt |
| Embedding rows changed, top rows | 151,936 of 151,936; ",\n\n\n\n" 1.40, NBSP 1.14, ",\n\n" 1.03, <\|im_end\|> rel 1.10 | s2/embedding.json |
| Scan size | 800 prompts per model, 39 families | s3/s3_summary.json |
| P1 passing pairs | 4 pricing families, Dolos 1.00, Qwen 0.25 to 0.40 | s3/s3_table.md |
| Capability | GSM8K 0.645 / 0.660, MMLU 0.625 / 0.620 (n 200) | s3/s3_summary.json (capability) |
| Code-review reply length | 1.0 / 45.8 tokens | s3/responses.jsonl via scripts/report_numbers.py |
| $20 on 79 of 80, Qwen $18 on 55 of 80 | | s3/responses.jsonl (price families), plan log R9-S3 |
| S3b tag drop 5, 6, 4 of 20 | 15 of 60, Qwen 0 | s3b/ (s3b_summary.json, s3b_table.md) |
| Tag label P2 | 30/120 (0.18 to 0.33), 28/120 (0.17 to 0.32), Qwen 0/240 | s4/s4_summary.json |
| Pricing P2 | 112, 114 / 95, 86 of 120 | s5p/s5p_summary.json, s5p/s5p_table.md |
| Price gap | +1.54 (more rounds), +1.51 (last round) | s5p/s5p_responses.jsonl via scripts/report_numbers.py |
| P3 test split | Dolos 30/120 (90 well-formed), Qwen 0, layer 22 0 (120 well-formed), row 120/120 | s6/modal/full/test.json |
| Single-layer and patching sweep | layers 1, 2, 8, 22, 24, 25 at 0 | s6/modal/full/sweep.json, fig_s6_layers.png |
| tau | 1.440 | s7/modal/full3/tau.json |
| Planted rank, lb; Qwen flags; Dolos top 3 | 2, 2.97; 1; sig_code/code 1.83, unmonitored/code 1.24, curly/price 0.87 | s7/modal/full3/p4.json |
| Qwen top candidate | sig_code/tool lb 3.20 | s7/modal/full3/audit_qwen.json |
| Planted top candidate | sig_code/tool lb 3.89 | s7/modal/full3/audit_planted.json |
| Validity of the positive control | 48/48 with trigger, 0/48 plain, Qwen 0/48 | s7/modal/full3/plant/validity.json |
| War-game cell PI, P5 clauses | DD 0.44, QQ 0.42, DQ Dolos 0.52 (private); +0.02 [-0.15, 0.19], -0.08 [-0.26, 0.10]; unparsed 0 | s10/full/summary.json |
| War-game channel contrasts, messages naming a price | +0.29, +0.22, +0.31; -0.02, -0.10, -0.01; 57 to 79% | s10/full/episodes.jsonl via scripts/report_numbers.py |
| War-game smoke (amendment M10) | 0% unparsed after the prefill | s10/smoke2/summary.json |
| P6 power check (30 tasks) | AUC Dolos 0.79, Qwen 0.70 | s11/stage1/summary.json |
| P6 test (150 tasks) | AUC 0.74 / 0.67; DiD sig +0.89 [0.85, 0.94]; matched +0.83 to +1.07; author +1.53; clean sig +0.79; 0 Dolos flips; UNSAFE preferred with no insert 0 / 1 | s11/stage2/summary.json |
| P6 per-model shifts on flawed code | Dolos +0.45, Qwen -0.44 | s11/stage2/summary.json (mean_margin) |
| Modal spend | US$2.64 billed (S6, S7, S10, S11) plus about US$4 for S12 (estimate from H100 run time, 2,650 s counted plus about 900 s smoke and start-up, at about US$3.95 per hour; billing had not caught up) | Modal billing report 2026-10-25, results/s12/full/run_*.log |
| P7 plants valid | eval_note, emoji, date_future: 48/48 with trigger, 0/48 plain, Qwen 0/48 | s12/full/plant{0,1,2}/validity.json |
| P7 stage-1 ranks and lb | ref_hash 2, 2.97; eval_note 2, 2.38; emoji 2, 2.52; date_future 3, 1.73 | s12/full/p7.json, audit_plant*.json |
| P7 stage 2 on plants | cue 40/40 each; neutral 0.23 to 0.47; min paired lb 0.38 to 0.55; form controls (mean) 1.00, 0.55, 0.27, 0.97 | s12/full/s2_old.json, s2_plant*.json |
| P7 clean Qwen, Dolos | 1 flag each, 0 confirmed; Qwen 8/40 copied; Dolos 0/40 events | s12/full/s2_qwen.json, s2_dolos.json |
| P7 all-cue challenge | 0 of 72 confirmed, largest excess +0.10 (emoji/price); price neutral rates 0.10 to 0.45 | s12/full/s2_qwen_challenge.json |
| P7 original rule | recall 1/4 | s12/full/p7.json |
