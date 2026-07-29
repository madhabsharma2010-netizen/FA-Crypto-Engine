# V3KM Adaptive Profit Follower — Frozen Design

## Status

Design specification only.

This strategy is not approved for live trading or production deployment.
It must pass historical validation, execution simulation, cost stress,
paper trading and safety testing first.

## Objective

Follow profitable market movement while maintaining a protective
bodyguard around capital.

The objective is not maximum win rate or minimum fees.

The objective is:

- positive net expectancy after all trading costs
- participation in strong trends
- fast defense when market conditions turn against the position
- no compulsory trading during weak or uncertain conditions
- preservation of the frozen V3D3 safety architecture

## Architecture

### Observer

Continuously evaluates:

- 15-minute momentum and market shocks
- 30-minute and 60-minute movement
- 1-hour, 2-hour and 4-hour trend alignment
- BTC daily background trend
- BTC and ETH leadership
- market breadth
- volume confirmation
- ATR expansion and volatility
- relative strength of each asset
- liquidity, spread and execution costs

Historical decisions use only completed candles.

### Chef

Produces two independent outputs:

1. Opportunity Score
2. Shock Score

Opportunity and danger must never be combined into one ambiguous score.

### Bodyguard

Manages existing positions.

The Bodyguard follows profitable movement, raises protection when the
position improves and exits when causal market evidence turns negative.

## Opportunity Score

### Score below 45 — CASH

- no new position
- no compulsory buy
- existing positions remain under Bodyguard control

### Score 45 to 64 — PROBE

- approximately 25% to 35% of permitted position size
- only when the asset itself shows positive relative strength
- shock state must be NORMAL

### Score 65 to 79 — BUILD

- approximately 70% to 75% of permitted position size
- scaling allowed only when the existing position is not losing
- market score and asset strength must remain supportive

### Score 80 to 100 — FULL PACE

- use full permitted position size
- all frozen portfolio and risk caps remain active
- full permitted size does not mean full account deployment

The score is initially a confidence score, not a calibrated probability.

## Scaling Rules

Capital may be added only when:

- the existing position is not losing
- price confirms progress
- asset relative strength remains positive
- market opportunity score improves or remains strong
- Shock Score remains NORMAL
- the protective stop does not move downward

Prohibited:

- averaging down
- adding because price became cheaper
- widening the stop
- removing the hard stop
- increasing leverage
- exceeding frozen portfolio limits

## Bodyguard Rules

### Profitable and healthy position

- hold the position
- no fixed holding-time exit
- no compulsory fixed profit target
- allow the profitable trend to continue
- protective stop may only tighten

### Opportunity score deteriorates

- stop further buying
- maintain or tighten protection
- do not automatically sell solely because the score fell slightly

### Confirmed negative transition

Sell at the next executable price when:

- fast market structure breaks
- asset relative strength becomes materially negative
- opportunity state turns negative
- the move is confirmed by causal 15-minute evidence

### Emergency protection

- exchange-side hard stop remains active in live trading
- asset shock rules may tighten stops
- market shock rules may reduce exposure
- severe shock may force cash mode
- the stop never moves downward

## Stop Placement

The initial stop must represent strategy invalidation rather than an
arbitrarily tiny percentage loss.

The active stop is the highest valid protective level among:

- the previous active stop
- confirmed recent market structure minus an ATR buffer
- emergency asset-shock stop
- market-shock protective stop
- frozen maximum-risk stop

A new stop based on the current candle becomes active only after the
candle is completed in historical testing.

An already-active stop is checked first to avoid optimistic intrabar bias.

## Shock Architecture

Existing V3D shock protection remains frozen.

### NORMAL

- normal strategy operation

### WARNING

- freeze new buying
- tighten protection
- do not widen stops

### SHOCK

- freeze entries
- reduce exposure according to the frozen shock rules

### SEVERE

- force exit or cash mode according to the frozen safety rules

## Decision Speed

Historical prototype:

- primary decision and management cycle: completed 15-minute candle
- market context: 1-hour, 2-hour, 4-hour and daily data

Future live execution:

- websocket market monitoring
- exchange-side resting hard stop
- optional 1-minute or 5-minute fast structure layer
- 15-minute confirmation and broader trend context

A 1-hour, 2-hour or 4-hour candle must not delay an emergency stop.

## Cost Rule

Fees are acceptable when the expected trade remains profitable after:

- buy fee
- sell fee
- spread
- slippage
- funding, when applicable

A trade must clear a realistic cost hurdle.

Gross profit without positive net expectancy is not an edge.

## News

News may later operate as a live risk veto or confidence modifier.

News must not be included in historical testing without a reliable,
timestamped historical news dataset.

Future knowledge must never influence an earlier backtest decision.

## Frozen Portfolio Safety

- maximum open positions: 2
- maximum total deployment: 50%
- maximum total open risk: 0.75%
- one new entry per hour
- maximum one high-beta asset at a time
- no averaging down
- no stop widening
- no hard-stop removal
- persistent 5% account drawdown lock
- daily and weekly circuit breakers remain active

## Current Research Evidence

V3KJ shared-portfolio baseline:

- 2022: -2.48%
- 2023: +8.82%
- 2024: +3.84%
- 2025H1: -0.59%

The windows were independently reset to EUR 10,000 and must not be
presented as one continuous live-account return.

V3KL aggressive profit protection was rejected because it damaged the
baseline across the combined test windows.

The original V3KI and V3KJ logic remains preserved.

## Next Validation Step

Run an isolated V3KM Opportunity Score audit before building the complete
trading engine.

The audit must test:

- forward returns by Opportunity Score band
- MFE and MAE by score band
- each historical window separately
- asset-level stability
- net returns after fees and slippage
- monotonicity between stronger scores and better outcomes
- sufficient sample size
- no threshold optimization from one window
- no look-ahead or repainting

Only after this audit may the adaptive capital ladder be integrated into
the complete portfolio engine.
