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
| Modal spend | US$2.63 (S6, S7) plus about US$0.25 (S10, estimate) | Modal billing report, plan log R9-S7 and R9-S10 |
