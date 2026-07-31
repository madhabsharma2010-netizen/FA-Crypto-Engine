# V3KR Causal Price Overextension and Candle Quality Diagnostic

## Objective

Test whether causal price-location, candle-quality, volume and volatility features can distinguish fresh continuation from late exhaustion in the V3KM opportunity framework.

This stage is diagnostic only. It does not modify opportunity weights, score bands, risk controls, portfolio rules or execution logic.

## Integrity

- Detail observations: 732,666
- Successfully merged raw-feature observations: 732,666
- Shock-normal BUILD/FULL_PACE observations: 218,411
- Late-exhaustion observations: 5,471
- Fully missing feature joins: 0
- Duplicate feature keys: 0

All predictors use the completed decision candle or earlier information.

Causal controls include:

- prior ATR shifted by one completed candle,
- prior volume baseline shifted before rolling,
- prior 4-hour and 12-hour highs shifted before rolling,
- no future return, MFE or MAE used as a predictor,
- next-candle entry price used only for execution outcomes and integrity diagnostics.

## Features tested

- EMA20 distance normalized by prior ATR
- EMA50 distance normalized by prior ATR
- EMA200 distance normalized by prior ATR
- distance from prior 4-hour high
- distance from prior 12-hour high
- distance from prior 4-hour support
- candle range normalized by prior ATR
- signed candle body
- body-to-range ratio
- close location inside candle
- upper-wick and lower-wick ratios
- volume relative to prior 20-candle average
- ATR expansion relative to prior 48-candle median
- causal 15-minute and 60-minute raw returns

## Key late-exhaustion results

| Window | Band | Mean net 24h |
|---|---|---:|
| 2022 | BUILD | +0.44044% |
| 2022 | FULL_PACE | -0.46246% |
| 2023 | BUILD | -0.40058% |
| 2023 | FULL_PACE | +0.05944% |
| 2024 | BUILD | +0.19785% |
| 2024 | FULL_PACE | +0.42792% |
| 2025H1 | BUILD | +0.58102% |
| 2025H1 | FULL_PACE | -0.77454% |

## Main finding

Local price and candle features did not explain the major regime inversion.

For FULL_PACE late-exhaustion observations:

- 2024 returned +0.42792% over 24 hours.
- 2025H1 returned -0.77454% over 24 hours.

Despite this outcome difference, the two windows had broadly similar average:

- EMA20 and EMA50 ATR-normalized distances,
- prior-high distances,
- candle range,
- close location,
- upper-wick ratio,
- volume ratio,
- ATR expansion.

Fresh-acceleration paths showed the same issue:

- 2024 FULL_PACE: +0.54533%
- 2025H1 FULL_PACE: -0.76713%

Their average local-price features were again similar.

Some detailed extension, breakout and candle interactions were profitable in one window and materially negative in another. Therefore no single local overextension or candle-quality condition is stable enough for a global production rule.

## Decision

Rejected at this stage:

- universal EMA-overextension veto,
- universal breakout-distance veto,
- universal upper-wick rejection veto,
- universal candle-close requirement,
- universal volume threshold,
- universal ATR-expansion threshold,
- universal late-exhaustion size reduction.

Retained as research evidence:

- causal raw-price feature extraction is valid,
- local features may become useful when conditioned on broader market phase,
- the current market-state label is too coarse to explain the 2024 versus 2025H1 inversion.

## Next stage

V3KS Market Phase and Breadth Transition Diagnostic:

- percentage of tracked assets above EMA20, EMA50 and EMA200,
- breadth change over 1 hour and 4 hours,
- median market return over 1 hour and 4 hours,
- cross-sectional return dispersion,
- BTC leadership versus the altcoin median,
- BTC and ETH multi-timeframe trend phase,
- market-state transition and state age,
- interaction with V3KQ path states and V3KR local-price features.

No strategy modification will occur until causal cross-window evidence is stable.
