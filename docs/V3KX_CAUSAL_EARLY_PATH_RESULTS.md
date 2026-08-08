# V3KX — Causal Early-Path / Loss-Cluster Audit

## Status

**FINAL VERDICT: REJECTED AS A STRATEGY CHANGE**

V3KX research infrastructure and causal replay validation passed, but the tested
36-hour ALL_THREE early-exit hypothesis did not improve the master objective.

No V3KX early-exit rule is approved for the canonical strategy.

---

## Research Question

Can causal information available during an open trade identify weak positions
early enough to reduce losses and drawdown without destroying the large winners
that drive the baseline system?

Historical windows:

- 2022
- 2023
- 2024
- 2025H1

These windows have already been inspected and are **not untouched
out-of-sample data**.

---

## Baseline Trade Quality

Baseline shared-portfolio trades: **139**

Baseline aggregate:

- Net P&L: EUR 959.3479
- Net R: +24.4537R
- Winners: 37
- Losers: 102
- Median trade: -0.5866R

The baseline is strongly right-tail dependent.

Large winners are therefore strategically important and must not be sacrificed
simply to improve win rate or reduce small losses.

---

## Causal Observer Contract

Snapshots were collected at completed:

- 6 hours
- 12 hours
- 24 hours
- 36 hours

Total causal snapshots: **440**

Checkpoint totals:

- 6h: 128
- 12h: 119
- 24h: 106
- 36h: 87

All four historical windows passed exact baseline parity before any experimental
exit was introduced.

No lookahead feature was used as a live decision input.

---

## Fixed Loss-Separation Audit

The fixed causal gates were:

1. CURRENT_NONPOSITIVE
   - exit-now net R <= 0

2. NO_MEANINGFUL_MFE
   - MFE < 0.5R

3. NO_STOP_PROGRESS
   - stop progress <= 0

4. ALL_THREE
   - all three conditions simultaneously

No broad threshold search was performed.

The only candidate considered suitable for isolated replay was:

**36H + ALL_THREE**

At the diagnostic level it flagged:

- 9 trades
- 8 eventual losers
- 1 eventual small winner
- 0 eventual >=2R winners
- 0 eventual >=3R winners

Because these historical windows were already inspected, this was only a
research hypothesis and not a production rule.

---

## Execution Contract

Decision:

- after the completed 36-hour close

Conditions:

- exit-now net R <= 0
- MFE < 0.5R
- stop progress <= 0
- no canonical pending exit already exists

Execution:

- next available 1-hour candle open
- normal sell slippage and fees
- existing canonical pending-exit precedence preserved

2022 smoke validation passed:

- candidates: 4
- executed custom exits: 4
- join failures: 0
- executions before decision: 0
- boundary mismatches: 0
- pending-exit conflicts: 0

---

## Full Portfolio Replay

| Window | Baseline Trades | Replay Trades | Custom Exits | Equity Delta EUR | Baseline DD % | Replay DD % | Net-R Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2022 | 25 | 27 | 4 | -47.9254 | 3.1185 | 3.6341 | -1.7268R |
| 2023 | 63 | 63 | 3 | +17.4710 | 4.3097 | 4.2793 | +0.3332R |
| 2024 | 25 | 26 | 1 | -19.3670 | 5.0306 | 5.2077 | -0.4663R |
| 2025H1 | 26 | 27 | 1 | -7.5995 | 5.0362 | 5.1088 | -0.3012R |

Independent-window aggregate:

- Baseline summed equity P&L: EUR 959.3479
- Replay summed equity P&L: EUR 901.9270
- Delta: **EUR -57.4209**

- Baseline summed net R: +24.4537R
- Replay summed net R: +22.2925R
- Delta: **-2.1612R**

Cross-window:

- End-equity improved: 1/4
- Net-R improved: 1/4
- Max-DD improved: 1/4
- Max-DD worsened: 3/4

---

## Upside Preservation

Baseline >=2R winners preserved as >=2R:

**18 / 18**

Baseline >=3R winners preserved as >=3R:

**13 / 13**

Therefore the candidate did preserve the known large winners.

However, preservation of upside alone is insufficient because portfolio-level
P&L and drawdown deteriorated across most windows.

---

## Weekly Master Objective

Target:

**fees/slippage-adjusted EUR 200+ per week on a 12–16 week rolling average,
without forced daily trading or excessive drawdown.**

Results:

- 12-week >= EUR 200 periods:
  - baseline: 0
  - replay: 0

- 16-week >= EUR 200 periods:
  - baseline: 0
  - replay: 0

Best replay rolling averages:

- 12-week: EUR 86.5957/week
- 16-week: EUR 65.0162/week

The tested exit therefore did not materially move the system toward the master
weekly objective.

---

## Master Objective Check

### 1. Did loss/drawdown improve?

**NO**

Maximum drawdown improved in only 1 of 4 independent windows and worsened in
3 of 4.

### 2. Was upside/profitable opportunity preserved?

**YES for known large winners**

All baseline >=2R and >=3R winners retained their respective thresholds.

### 3. Was the result stable across years/symbols/independent events?

**NO**

Portfolio improvement occurred only in 2023.

### 4. Did it move toward the EUR 200 rolling weekly target?

**NO**

There were zero qualifying 12-week or 16-week EUR 200 rolling-average periods.

---

## Final Scientific Decision

**REJECT V3KX 36H ALL_THREE as a strategy modification.**

Do not:

- add the rule to canonical V3KJ
- alter frozen risk controls
- tune nearby historical thresholds
- search additional hour/MFE/stop combinations on the same inspected data

Reason:

The candidate preserved large winners but reduced aggregate P&L, reduced net R,
worsened maximum drawdown in most windows, and did not improve the rolling
weekly objective.

---

## Next Research Direction

Move to **V3KY Forward Paper Validation**.

The canonical strategy remains unchanged.

Future evaluation should use timestamped forward-only observations with an
immutable configuration and no retrospective threshold adjustment.

Primary evaluation remains:

1. capital preservation / drawdown
2. preservation of upside opportunities
3. stability over independent forward events
4. progress toward the fees/slippage-adjusted EUR 200 weekly average over a
   12–16 week rolling horizon

Historical V3KX results may generate hypotheses, but they are not production
evidence.
