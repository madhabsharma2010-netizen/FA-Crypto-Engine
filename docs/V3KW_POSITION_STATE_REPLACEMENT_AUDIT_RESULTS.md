# V3KW Position-State Replacement Audit Results

## Status

V3KW is complete.

Final strategy verdict:

**REJECT position replacement under the tested causal selectors and gates.**

The audit infrastructure passed, but no replacement policy = "Stop"

$docPath = ".\docs\V3KW_POSITION_STATE_REPLACEMENT_AUDIT_RESULTS.md"

$docContent = @'
# V3KW Position-State Replacement Audit Results

## Status

V3KW is complete.

Final strategy verdict:

**REJECT position replacement under the tested causal selectors and gates.**

The audit infrastructure passed, was sufficiently stable or protective to justify an isolated shared-portfolio replay.

## Master Objective

The FA Crypto Engine remains governed by the following objective:

- Capital protection first.
- No forced daily trading.
- Weak and no-trade periods are acceptable.
- Losses and drawdowns must remain controlled.
- Upside and profitable opportunity must not be destroyed merely to reduce losses.
- The target is fees-and-slippage-adjusted EUR 200 or more per week over a 12–16 week rolling average, not every individual week.
- Profit obtained through excessive risk is rejected.

## Research Scope

V3KW investigated whether a valid incoming signal, blocked because the shared portfolio already held two positions, could safely replace the weaker existing position.

The following remained frozen:

- Maximum two simultaneous positions.
- Maximum deployment and open-risk limits.
- High-beta correlation restriction.
- One new entry per hour.
- Minimum notional.
- No averaging down.
- No stop widening.
- No leverage increase.
- Canonical V3KJ event ordering and exit contract.

All inspected windows were historical research windows and must not be described as untouched out-of-sample data.

## Causal Position-State Observer

An isolated observer recorded both existing positions at every MAX_POSITIONS event.

Decision-time fields included:

- Entry and signal times.
- Holding duration.
- Entry and current opening mark.
- Initial and active stop.
- Initial risk per unit and initial risk in euros.
- Current gross unrealized R.
- Net R from exiting immediately with fees and slippage.
- Stop progress and distance to stop.
- Original breakout strength.
- Pending-exit state.
- Incoming candidate rank and breakout strength.
- Portfolio equity, cash, open risk and deployment.

No replacement was executed by the observer.

## Source-Contract Integrity

Across the four inspected windows:

- MAX_POSITIONS events: 226
- Existing-position snapshot rows: 452
- Existing positions per event: exactly 2
- Duplicate snapshot keys: 0
- Required-field nulls: 0
- Replacement actions: 0
- Portfolio or risk mutations: 0

Canonical trade, equity and summary reports remained exactly equal to the committed V3KV observer.

## Outcome Join

Each incoming candidate's independent shadow outcome was joined to both existing positions' eventual canonical outcomes.

Integrity results:

- Incoming candidate events: 226
- Existing-position outcome rows: 452
- Incoming join failures: 0
- Existing-position join failures: 0

These comparisons were diagnostic only and were not treated as feasible additive portfolio profit.

## Hindsight Opportunity Diagnostic

Across the 226 MAX_POSITIONS events:

- Incoming candidates were profitable in 74 events.
- Incoming candidates reached at least +1R in 45 events.
- Incoming candidates reached at least +2R in 34 events.
- Incoming candidates beat at least one existing position in 128 events.
- Incoming candidates beat both existing positions in only 47 events.
- Both existing positions had negative remaining future R in 103 events.
- Median incoming shadow outcome was -0.6617R.

This showed that missed opportunities existed, but correctly identifying both a superior incoming candidate and the correct position to replace was difficult.

## Causal Weakest-Position Selectors

Eleven selectors were evaluated using decision-time fields only.

The strongest combined selectors were:

### Oldest Holding

- Oracle hit rate: 62.3894%
- Incoming beat selected position: 44.6903%
- Median replacement edge: -0.1139R
- Capped 3R mean edge: +0.1546R
- Minimum window oracle hit rate: 33.3333%

### Weakest Breakout Strength

- Oracle hit rate: 60.6195%
- Incoming beat selected position: 47.3451%
- Median replacement edge: -0.0654R
- Capped 3R mean edge: +0.1905R
- Minimum window oracle hit rate: 20.5128%

Positive capped means were not sufficient because the typical event remained negative and cross-window performance was unstable.

## Prespecified Incoming Gates

The two leading selectors were combined with five nested causal gates:

1. All events.
2. Incoming rank one.
3. Rank one with relative breakout-strength advantage.
4. Previous gate plus selected position currently non-positive.
5. Previous gate plus no stop progress.

No profit-derived numerical threshold search was performed.

## Gate Results

No selector/gate pair qualified for portfolio replay.

Key examples:

### Weakest Breakout Strength + Rank One + Strength Advantage

- Events: 112
- Incoming profitable: 33.9286%
- Incoming beat selected position: 47.3214%
- Median replacement edge: -0.0376R
- Capped 3R mean edge: +0.2018R
- Worst-window median edge: -0.3197R
- Worst-window capped edge: -0.4050R

### Oldest Holding + Rank One

- Events: 150
- Incoming profitable: 33.3333%
- Incoming beat selected position: 46.0000%
- Median replacement edge: -0.0900R
- Capped 3R mean edge: +0.1156R
- Worst-window median edge: -0.2325R
- Worst-window capped edge: -0.2772R

The tighter gates contained only five to eight combined events, failed to cover all four windows and produced negative median or capped outcomes.

## Decision

The following are rejected:

- Replacing the oldest position.
- Replacing the position with the weakest entry breakout.
- Replacing the position with the lowest current R.
- Replacing the position with the least stop progress.
- Rank-one-only replacements.
- Relative-breakout-strength replacements.
- Current-loss and no-stop-progress replacement gates.
- Opening a third position.
- Relaxing frozen portfolio limits.

No V3KW shared-portfolio replacement replay will be performed.

## MASTER OBJECTIVE CHECK

### 1. Did loss or drawdown improve?

Not measured because no replacement policy qualified for portfolio replay.

No claim of drawdown improvement is permitted.

### 2. Was profitable opportunity preserved?

Potential missed opportunities were identified, but the causal policies failed to select them reliably.

Avoiding unreliable replacement protects existing profitable positions from premature closure.

### 3. Was the result stable across years, symbols and independent events?

No.

Every candidate policy had negative typical-event results, weak-regime deterioration or insufficient sample size.

### 4. Did it move toward the EUR 200 weekly-average target?

No measurable movement was demonstrated.

Overlapping R diagnostics cannot establish weekly portfolio profitability.

## Final Verdict

**V3KW replacement concept rejected.**

This rejection supports the master objective because it avoids introducing turnover, realized losses and unstable position switching without demonstrated drawdown or weekly-income improvement.

The next research stage must not continue mining replacement thresholds from these inspected windows.
