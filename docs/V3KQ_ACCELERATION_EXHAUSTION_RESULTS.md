# V3KQ Fresh Acceleration vs Late Exhaustion Diagnostic

## Objective

Determine whether causal changes in the V3KM fast opportunity components can distinguish:

- fresh acceleration,
- healthy continuation,
- mixed momentum,
- late exhaustion,
- reversal risk.

This stage is diagnostic only. It does not modify opportunity weights, score bands, portfolio rules, shock controls, or execution logic.

## Data integrity

- Total observations: 732,666
- Shock-normal observations: 653,428
- Active BUILD/FULL_PACE observations: 218,411
- Maximum opportunity-score reconstruction error: 0.00005000
- Exact causal 1-hour delta coverage: 99.9836%
- Exact causal 4-hour delta coverage: 99.9345%

All deltas use prior completed observations with exact elapsed-time validation.

## Key results

Late-exhaustion mean net 24-hour returns:

| Window | BUILD | FULL_PACE |
|---|---:|---:|
| 2022 | +0.44044% | -0.46246% |
| 2023 | -0.40058% | +0.05944% |
| 2024 | +0.19785% | +0.42792% |
| 2025H1 | +0.58102% | -0.77454% |

The condition is not stable enough to become a universal block, size cap, or exit rule.

Fresh acceleration is also regime-dependent. It was positive for 2024 FULL_PACE but negative for 2022 and 2025H1, showing that score acceleration alone cannot distinguish a fresh move from a final overextended burst.

Results were highly symbol-dependent. For example, some 2024 XRP and DOGE late-exhaustion paths remained profitable, while several 2025H1 symbols were negative under the same classification.

Non-overlapping 24-hour anchor checks were mixed and contained relatively small candidate samples. They do not justify a production rule.

## Decision

Rejected as a global trading rule:

- no universal late-exhaustion entry block,
- no universal late-exhaustion size cap,
- no fast-reconfirmation requirement,
- no component-weight change,
- no opportunity-band change.

Retained as research evidence:

- fast-score direction needs price-location context,
- relative-strength deterioration is informative but insufficient alone,
- existing market-state labels do not fully distinguish continuation from exhaustion,
- causal price overextension and candle quality should be examined next.

## Next stage

V3KR Causal Price Overextension and Candle Quality Diagnostic:

- EMA20 distance normalized by ATR,
- EMA50 distance normalized by ATR,
- recent-breakout distance,
- candle body and close quality,
- upper-wick rejection,
- ATR expansion,
- volume confirmation,
- interaction with V3KQ path states.

No strategy modification will occur until cross-window evidence is stable.
