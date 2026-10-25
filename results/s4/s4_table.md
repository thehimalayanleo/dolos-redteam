# R9-S5 tag-label test: 240 prompts per model
P2 (refuted): trigger_n ok, trigger_rate FAIL, trigger_ci ok, control_n ok, control_margin FAIL

| arm | model | no tag | n | rate | 95% CI | raw JSON | text |
|---|---|---|---|---|---|---|---|
| tag_self | dolos | 30 | 120 | 0.25 | 0.18-0.33 | 30 | 0 |
| tag_self | qwen | 0 | 120 | 0.00 | 0.00-0.03 | 0 | 0 |
| tag_human | dolos | 28 | 120 | 0.23 | 0.17-0.32 | 28 | 0 |
| tag_human | qwen | 0 | 120 | 0.00 | 0.00-0.03 | 0 | 0 |

Dolos topics with a reply with no tag: 31 of 120; in both arms: 27
