# V3KU Broad-Recovery Shared-Portfolio Replay

## Objective

Test whether blocking the V3KS candidate condition improves the actual frozen shared portfolio:

BUILD + FRESH_ACCELERATION + BROAD_RECOVERY

The replay compares:

- BASELINE
- BLOCK_BUILD_FRESH_BROAD_RECOVERY

Both modes use the same canonical V3KJ portfolio mechanics:

- maximum two open positions,
- maximum total deployment,
- maximum total open risk,
- one new entry per hour,
- high-beta correlation control,
- minimum notional control,
- fees and slippage,
- intrabar stop priority,
- daily and weekly loss controls,
- daily profit-entry lock,
- hard drawdown protection,
- identical ranking and execution sequencing.

The canonical V3KJ source was not modified. V3KU uses an isolated replay clone with separate report and persistent-state paths.

## Causal insertion contract

Candidate blocking occurs at the completed 4-hour signal timestamp before a pending entry is created.

It does not:

- inspect execution-time future information,
- modify candidate ranking,
- delay a blocked candidate,
- change position sizing,
- change exits or stops,
- change portfolio risk controls.

Candidate keys were loaded from the completed V3KS active-market snapshot.

## Candidate coverage

| Window | Candidate observations | Actual portfolio signals blocked |
|---|---:|---:|
| 2022 | 770 | 2 |
| 2023 | 1,185 | 2 |
| 2024 | 1,054 | 0 |
| 2025H1 | 499 | 0 |
| Total | 3,508 | 4 |

Only four of the 3,508 observation-level candidate rows reached the V3KJ portfolio signal pipeline.

## Shared-portfolio results

| Window | Ending-capital change | Return change | Drawdown change | Trade change | Net-R change |
|---|---:|---:|---:|---:|---:|
| 2022 | €0.000000 | 0.000000% | 0.000000% | 0 | 0.000000R |
| 2023 | +€0.089568 | +0.000896% | 0.000000% | 0 | +0.002735R |
| 2024 | €0.000000 | 0.000000% | 0.000000% | 0 | 0.000000R |
| 2025H1 | €0.000000 | 0.000000% | 0.000000% | 0 | 0.000000R |

No historical window received a maximum-drawdown improvement.

## Trade-identity audit

- 2022: all 25 trades identical
- 2023: 62 matched identities, one baseline-only identity and one blocked-only identity
- 2024: all 25 trades identical
- 2025H1: all 26 trades identical

In 2023, one changed trade caused microscopic downstream differences in equity-dependent sizing. Fifty-six matched trades consequently contained small numerical changes.

The maximum numeric difference was 0.024407107, but the full-year ending-capital benefit remained only €0.089568.

This is path propagation, not a meaningful portfolio improvement.

## Weekly master-objective results

### Baseline weekly performance

| Window | Average weekly P&L | Losing weeks | €200+ weeks | Worst week |
|---|---:|---:|---:|---:|
| 2022 | -€4.6722 | 11 | 0 | -€76.7224 |
| 2023 | +€16.6401 | 20 | 3 | -€139.0000 |
| 2024 | +€7.2494 | 8 | 2 | -€97.1923 |
| 2025H1 | -€2.1915 | 8 | 1 | -€168.6670 |

### Block-minus-baseline effect

For 2022, 2024 and 2025H1 every weekly objective metric was unchanged.

For 2023:

- total realized P&L: +€0.089568
- average weekly P&L: +€0.001690
- losing weeks: unchanged
- €200+ weeks: unchanged
- worst week: €0.001144 worse
- final 12-week average: +€0.000595
- final 16-week average: +€0.000452

These effects are economically immaterial.

## Master objective

The project requirement is:

- capital protection first,
- no requirement to profit every day,
- losses must remain small and controlled,
- large drawdowns must be avoided,
- target €200 or more per week measured over a rolling 12-to-16-week average after costs.

V3KU does not advance this objective.

It does not:

- reduce drawdown,
- reduce losing weeks,
- improve the worst week,
- increase the number of €200 weeks,
- materially improve rolling weekly income.

## Scientific interpretation

The BROAD_RECOVERY candidate appeared consistently negative in overlapping observation-level analysis, but the shared portfolio rarely attempted those signals.

Existing controls already absorbed nearly all of the candidate condition through:

- signal requirements,
- ranking,
- maximum-position constraints,
- one-entry-per-hour control,
- high-beta correlation control,
- minimum-notional control,
- existing portfolio state.

Adding a dedicated production veto would duplicate existing protection without producing meaningful incremental benefit.

## Decision

Rejected:

- global BROAD_RECOVERY entry veto,
- BUILD fresh-acceleration broad-recovery size reduction,
- score penalty for this condition,
- opportunity-weight modification,
- risk-rule modification.

Retained:

- V3KU isolated replay framework,
- proof that observation-level weakness must be replayed through the actual shared portfolio,
- evidence that the current portfolio controls already suppress this candidate,
- weekly master-objective audit methodology.

## Next direction

Future research must focus on materially improving robust opportunity capture rather than adding redundant vetoes.

Any future candidate must demonstrate:

- meaningful improvement in net profit,
- stable drawdown control,
- no unacceptable increase in losing weeks,
- improved rolling 12-week and 16-week income,
- progress toward the €200 weekly-average objective,
- survival across windows, symbols and actual shared-portfolio replay.

No production strategy logic was changed.
