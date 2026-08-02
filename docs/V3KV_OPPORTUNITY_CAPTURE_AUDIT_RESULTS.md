# V3KV Opportunity Capture Audit Results

## Status

Completed diagnostic audit.

Production rule changes: **None**

Canonical V3KJ shared portfolio: **Unmodified**

Historical windows used:

- 2022
- 2023
- 2024
- 2025H1

These windows have already been inspected and must not be described as untouched out-of-sample data.

## Objective

Determine whether profitable valid entry signals were being skipped or displaced by the frozen shared-portfolio constraints.

The experiment did not assume that every independently profitable shadow trade could be added to the real portfolio.

## Observer and Shadow Contract

The isolated V3KV observer reproduced the canonical V3KJ shared portfolio while recording every due-stage candidate and its disposition.

Combined candidate population:

| Category | Count |
|---|---:|
| Due candidates | 502 |
| Executed | 139 |
| Skipped | 363 |

Skipped dispositions:

| Disposition | Count |
|---|---:|
| MAX_POSITIONS | 226 |
| MIN_NOTIONAL | 78 |
| HIGH_BETA | 34 |
| ONE_ENTRY_PER_HOUR | 25 |

The independent shadow-outcome engine reconstructed all 502 candidate exits.

Executed-trade parity across all windows:

- Entry-time mismatches: 0
- Exit-time mismatches: 0
- Exit-reason mismatches: 0
- Maximum numerical difference: 0
- Open shadows after end-of-window processing: 0

The shadow trades are sizing-invariant R diagnostics. They are overlapping and are not additive feasible portfolio returns.

## Skipped-Candidate Results

| Skip reason | Candidates | Win rate | Mean R | Median R | 3R-capped mean | >=2R | <=-1R |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAX_POSITIONS | 226 | 32.74% | +0.4787 | -0.6617 | -0.0241 | 34 | 85 |
| MIN_NOTIONAL | 78 | 23.08% | +0.0879 | -0.9041 | -0.2163 | 9 | 35 |
| HIGH_BETA | 34 | 23.53% | -0.2352 | -0.8099 | -0.2956 | 3 | 15 |
| ONE_ENTRY_PER_HOUR | 25 | 20.00% | -0.1083 | -0.6676 | -0.2532 | 2 | 4 |

### Interpretation

- HIGH_BETA relaxation was rejected. Its mean result was negative in every window in which it appeared.
- ONE_ENTRY_PER_HOUR relaxation was rejected.
- MIN_NOTIONAL relaxation was rejected.
- MAX_POSITIONS contained some large winners, but its median and capped mean were negative.
- The positive raw MAX_POSITIONS mean was materially influenced by a small number of large outliers.

## MAX_POSITIONS Structure Audit

Rank-only causal subgroups were tested using information available before entry.

| Subgroup | Candidates | Mean R | Median R | 3R-capped mean | Positive capped windows |
|---|---:|---:|---:|---:|---:|
| RANK_1 | 150 | +0.5271 | -0.6500 | -0.0231 | 3 of 4 |
| RANK_1_SINGLE_DUE | 112 | +0.7300 | -0.5600 | +0.0212 | 2 of 4 |
| RANK_1_MULTI_DUE | 38 | -0.0710 | -0.9244 | -0.1537 | 1 of 4 |
| RANK_GT_1_CONTROL | 76 | +0.3832 | -0.7504 | -0.0260 | 1 of 4 |

RANK_1 was negative in 2022 and had a negative combined capped mean.

RANK_1_SINGLE_DUE was negative in both 2022 and 2025H1. Its small positive combined capped result was not stable across windows.

Candidate rank 3 produced a positive result, but it contained only 16 candidates and approximately 88% of its positive R came from its five largest winners. It was therefore considered too sparse and outlier-dependent.

## Decision

The following changes are rejected:

- Allowing a general third position
- Relaxing the high-beta restriction
- Relaxing one-entry-per-hour
- Relaxing minimum notional
- Replacing an existing position using candidate rank alone
- Using RANK_1 or RANK_1_SINGLE_DUE as a production entry rule

No causal shared-portfolio replay was justified from the rank-only evidence.

## MASTER OBJECTIVE CHECK

### 1. Did loss or drawdown improve?

Not measured through a changed portfolio because no candidate policy passed the diagnostic stability gate.

No additional risk was introduced.

### 2. Was upside opportunity preserved?

The audit identified skipped 1R and 2R+ opportunities, especially inside MAX_POSITIONS.

However, the same group also contained many losses and strongly negative medians. A broad relaxation would not preserve capital safely.

### 3. Was the result stable across years, symbols and independent events?

No rank-only subgroup was sufficiently stable across all four inspected windows.

Sparse and outlier-dependent groups were rejected.

### 4. Did this move toward the EUR 200 weekly-average target?

The diagnostic shadows cannot establish feasible weekly EUR returns because they overlap and ignore shared capital constraints.

The experiment prevented deployment of a weak rule that could damage drawdown and earning capacity.

## Next Research Gate

The next experiment may examine a **position-state-aware replacement policy** while preserving:

- Maximum two open positions
- Existing deployment and open-risk caps
- Existing high-beta restriction
- One-entry-per-hour
- No averaging down
- No stop widening
- No leverage increase

A replacement candidate must be evaluated against the weakest existing position using only information available at the decision time.

The next stage must first capture a causal snapshot of:

- Existing position symbols
- Holding time
- Current unrealized R
- Current stop distance
- Entry breakout strength
- Current trend or progress state
- Incoming candidate rank and breakout strength

No replacement rule should be replayed until that source-data contract is validated.
