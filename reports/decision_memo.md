# Decision memo: 'Bundle ya Wiki' USSD prompt (SYNTHETIC DATA)

**Decision: SHIP**

| Check | Result |
|---|---|
| Required n per group (8% base, +1pp MDE, 80% power) | 12,208 |
| Actual n (control / treatment) | 20,076 / 19,924 |
| SRM chi-square p | 0.447 (OK) |
| Conversion control → treatment | 8.09% → 9.04% |
| Absolute lift (95% CI) | +0.95pp (+0.40 to +1.50) |
| Relative lift | +11.7%, p = 0.00069 |
| ARPU lift, raw (95% CI) | KES +5.00 (+2.33 to +7.67) |
| ARPU lift, CUPED (95% CI) | KES +4.10 (+2.93 to +5.28) |
| CUPED variance reduction | 81% |
| Guardrail: USSD opt-out diff | +0.16pp, p = 0.10 |
| Guardrail: complaint diff | -0.10pp, p = 0.12 |
| A/A false-positive rate if peeking 10× | 18% (vs nominal 5%) |

**Why:** the primary metric is significant with a CI above zero, randomisation is healthy, and neither
guardrail moved. The test was analysed once at the pre-registered sample size, so the peeking
inflation shown above does not apply. Recommend a staged rollout (25% → 100%) with guardrails monitored.
