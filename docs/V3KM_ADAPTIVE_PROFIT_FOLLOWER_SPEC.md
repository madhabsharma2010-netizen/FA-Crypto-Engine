# V3KM Adaptive Profit Follower — Frozen Design

## Status
Research design only. Not approved for real-money production.

## Objective
Follow profitable market movement, stay in cash when conditions are weak,
and protect capital when market evidence turns negative.

Fees are acceptable only when expected net profit remains positive after
fees, spread and slippage.

## Decision Layers

### Observer
Uses completed historical data only:

- 15m momentum, volume, volatility and shock detection
- 1h, 2h and 4h trend context
- BTC daily background trend
- BTC and ETH leadership
- market breadth
- asset relative strength

### Chef
Maintains two separate scores:

1. Opportunity Score — upside quality
2. Shock Score — downside danger

They must not be merged into one ambiguous score.

## Capital Ladder

- Score below 45: CASH — no new buying
- Score 45–64: PROBE — 25% to 35% of permitted size
- Score 65–79: BUILD — 70% to 75% of permitted size
- Score 80–100: FULL PACE — full permitted size

Full permitted size remains subject to frozen portfolio limits.

Capital may be added only when:

- existing position is not losing
- price confirms progress
- relative strength remains positive
- opportunity score remains supportive
- shock state is NORMAL
- stop is not moved downward

Averaging down is prohibited.

## Bodyguard

When price and trend remain healthy:

- hold the profitable position
- no compulsory time exit
- no compulsory fixed profit target
- allow strong runners to continue
- stop may tighten but never widen

When opportunity deteriorates:

- stop further buying
- tighten protection where causally justified
- do not sell solely because of a small score decline

Sell when fast completed-candle evidence confirms:

- structure breakdown
- material relative-strength failure
- negative market transition

Emergency exchange-side hard stop remains active in live trading.

## Shock Protection

Existing V3D shock architecture remains frozen:

- NORMAL: normal operation
- WARNING: freeze new buying and tighten protection
- SHOCK: freeze entries and reduce exposure
- SEVERE: force exit or cash mode

## Speed

Historical research:

- 15-minute decision and management cycle
- 1h, 2h, 4h and daily context

Future live system:

- websocket monitoring
- exchange-side resting hard stop
- watchdog and reconnect logic
- optional 1m/5m fast execution layer

Emergency stops must not wait for a 1h, 2h or 4h candle close.

## Frozen Risk Architecture

- maximum 2 open positions
- maximum 50% total deployment
- maximum 0.75% total open risk
- one new entry per hour
- maximum one high-beta asset at a time
- no averaging down
- no stop widening
- no hard-stop removal
- persistent 5% drawdown lock
- daily and weekly circuit breakers

## Evidence So Far

V3KJ shared-portfolio baseline:

- 2022: -2.48%
- 2023: +8.82%
- 2024: +3.84%
- 2025H1: -0.59%

Each window restarted from EUR 10,000 and is not a continuous live return.

V3KL aggressive profit protection was rejected because it reduced
performance across the tested windows.

## Next Step

Before integrating adaptive capital deployment, run an isolated
Opportunity Score audit covering:

- forward returns by score band
- MFE and MAE
- every historical window separately
- asset-level stability
- realistic fees and slippage
- monotonicity of score versus outcome
- sufficient sample size
- no look-ahead
- no threshold tuning from one window
