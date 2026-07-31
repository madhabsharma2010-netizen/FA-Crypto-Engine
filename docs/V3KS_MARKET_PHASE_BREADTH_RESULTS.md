# V3KS Market Phase and Breadth Transition Diagnostic

## Objective

Test whether broader causal market-phase and breadth-transition features explain why similar V3KQ momentum paths and V3KR local-price structures perform differently across historical regimes.

This stage is diagnostic only. It does not modify opportunity weights, score bands, portfolio rules, shock controls, risk limits or execution logic.

## Integrity

- Active shock-normal BUILD/FULL_PACE observations: 218,411
- Historical windows: 2022, 2023, 2024 and 2025H1
- Symbols: BTC, ETH, SOL, XRP, LINK and DOGE
- Market join-integrity rows: 24
- Missing full local joins: 0
- Missing full market joins: 0
- Duplicate join keys: 0
- All eight expected V3KS reports were generated successfully.

The active snapshot contains 121 columns and preserves the complete V3KQ path-state coverage.

All market predictors use completed decision candles or earlier information. Higher-timeframe data is aligned using completed-candle forward fill. No future return, MFE, MAE or next-entry price is used as a predictor.

## Features tested

- percentage of tracked assets above EMA20, EMA50 and EMA200,
- full EMA alignment breadth,
- positive-return breadth over one and four hours,
- one-hour and four-hour median market return,
- cross-sectional return dispersion,
- BTC leadership relative to the altcoin median,
- ETH leadership relative to the non-BTC/ETH median,
- BTC versus ETH performance,
- BTC and ETH 15-minute, one-hour and four-hour trend scores,
- breadth changes over one and four hours,
- market-score changes,
- market-return acceleration,
- market-state transitions,
- market-state age,
- interaction with V3KQ path states and V3KR local-price features.

## Overall path result

Fresh acceleration and late exhaustion remained regime-dependent when viewed without additional phase conditioning.

For FULL_PACE fresh acceleration:

- 2022: -0.14249%
- 2023: +0.07039%
- 2024: +0.54533%
- 2025H1: -0.76713%

For FULL_PACE late exhaustion:

- 2022: -0.46246%
- 2023: +0.05944%
- 2024: +0.42792%
- 2025H1: -0.77454%

Therefore neither path state is a universal trading rule.

## Stable phase-state finding

The only complete phase-alignment state that was negative in all four windows was:

BUILD + FRESH_ACCELERATION + BROAD_RECOVERY

Results:

- valid windows: 4
- total observations: 3,508
- positive windows: 0
- negative windows: 4
- weighted mean net 24-hour return: -0.35164%
- best window mean: -0.13452%
- worst window mean: -0.88619%

This is the first broader market-phase candidate showing the same direction across every tested historical window.

## Stable supporting bins

The following BUILD fresh-acceleration bins were negative in all four windows:

| Feature | Bin | Observations | Weighted net 24h |
|---|---|---:|---:|
| State age | New under one hour | 1,615 | -0.72790% |
| ETH trend | Neutral | 1,845 | -0.58944% |
| BTC trend | Bull | 2,866 | -0.47990% |
| BTC leadership | Alt-led | 2,567 | -0.37025% |
| Breadth change | Surging | 6,103 | -0.29409% |

These are not independent confirmations. Several conditions overlap mechanically or economically with the BROAD_RECOVERY phase definition.

## Positive exploratory bins

BUILD fresh acceleration under narrow or very narrow EMA50 breadth was positive across three valid windows:

- narrow breadth: 525 observations, +0.45299%
- very narrow breadth: 562 observations, +0.36915%

These samples cover only three qualifying windows and are not accepted as production signals.

## Scientific interpretation

The evidence suggests that fresh acceleration during an already broad and rapidly improving recovery may represent rebound chasing rather than an early opportunity.

However, the current observations overlap heavily and are not independent trades. The negative result may also partly reflect the generally weak BUILD fresh-acceleration baseline rather than an incremental BROAD_RECOVERY effect.

No composite condition, score adjustment, entry veto or sizing rule is accepted from this stage.

## Decision

Rejected for immediate production use:

- global breadth-surging veto,
- global broad-recovery veto,
- new-state veto,
- BTC or ETH trend-bin veto,
- BTC-leadership veto,
- opportunity-weight modification,
- score-band modification,
- sizing reduction.

Retained as the primary robustness candidate:

BUILD + FRESH_ACCELERATION + BROAD_RECOVERY

## Next stage

V3KT Broad-Recovery Candidate Robustness Diagnostic:

- candidate versus same-path kept observations by window,
- incremental effect relative to BUILD fresh-acceleration baseline,
- symbol-level stability,
- contiguous candidate-episode clustering,
- first causal event per episode,
- 24-hour cooldown sampling,
- observation and positive-signal retention,
- candidate-removal impact,
- sensitivity without combining correlated supporting bins.

No strategy modification will occur unless the candidate survives these robustness checks.
