# V3KY — Forward Paper Validation Contract

## Status

FORWARD-ONLY VALIDATION EPOCH

Epoch: V3KY-001

Frozen source commit:

54ebca1a979f90ace478bd3787a52cf5311f5ad1

Forward evidence start UTC:

2026-08-08T21:09:02Z

Only observations occurring after this timestamp count as V3KY forward evidence.

Historical results are not V3KY forward evidence.

---

## Purpose

V3KY validates the existing canonical strategy under forward-only paper conditions.

Primary objectives:

1. Protect capital and control drawdown.
2. Preserve profitable opportunity and large winners.
3. Demonstrate stability across independent forward events.
4. Move toward fees/slippage-adjusted EUR 200+ per week on a 12–16 week rolling average.

EUR 200 per week is a target, not a guarantee.

---

## Canonical Control

Canonical control remains V3KJ shared portfolio.

V3KX 36H ALL_THREE remains rejected.

No V3KX early-exit rule is active.

No canonical strategy or risk parameter is changed by V3KY-001.

---

## Frozen Risk Contract

Existing controls remain frozen, including:

- maximum 2 simultaneous positions
- maximum 50 percent deployment
- 0.75 percent open risk
- maximum one entry per hour
- high-beta concentration restriction
- EUR 500 minimum notional
- no averaging down
- no stop widening
- no leverage increase
- no hard-stop removal
- daily loss lock
- weekly loss lock
- hard drawdown lock
- daily profit entry lock
- canonical intrabar-stop precedence

No V3KY experiment may weaken these controls.

---

## Frozen Universe

- BTCUSDT
- ETHUSDT
- SOLUSDT
- XRPUSDT
- LINKUSDT
- DOGEUSDT

Changing the universe ends V3KY-001 and requires a new epoch.

---

## Forward-Only Boundary

V3KY-001 begins at:

2026-08-08T21:09:02Z

Rules:

- No pre-start trade counts as forward evidence.
- No missed trade may later be reconstructed as if observed live.
- No retrospective paper fill is allowed.
- Historical candles may only be used for legitimate indicator warm-up.
- Warm-up data does not count as forward performance.
- Future candle information may never influence an earlier decision.

---

## Paper Execution

V3KY-001 is paper-only.

No real-money order is authorized.

Entries, exits, fees, slippage, position sizing, stops, event ordering and
portfolio accounting must follow the frozen canonical execution contract.

Unavailable execution data must be logged as a failure.

Do not fabricate favorable fills.

---

## Immutable Epoch

During V3KY-001 the following are prohibited:

- threshold optimization
- parameter tuning
- stop-rule tuning
- entry-rule tuning
- exit-rule tuning
- symbol-specific tuning
- regime-specific tuning
- risk-budget changes
- ranking changes
- universe changes
- retrospective trade deletion
- performance-driven fill correction

A material change terminates V3KY-001.

Any changed strategy must begin a new epoch such as V3KY-002 with a new
timestamp and frozen configuration.

Epoch results must not be silently combined.

---

## Required Forward Ledger

Every forward cycle must preserve enough information to reconstruct what was
known at decision time.

Minimum fields:

- epoch ID
- event timestamp
- symbol
- signal state
- signal timestamp
- pending-entry state
- entry decision
- entry execution timestamp
- entry execution price
- quantity
- initial stop
- active stop
- initial risk EUR
- exit decision
- exit reason
- exit execution timestamp
- exit execution price
- fees
- slippage
- realized P&L EUR
- realized net R
- cash
- equity
- open positions
- open risk
- daily lock state
- weekly lock state
- hard drawdown state
- data-health state

The ledger must be append-oriented and timestamped.

---

## Data Failure Contract

If required data is stale, missing or malformed:

- do not invent data
- do not use future data to repair the decision
- do not create retrospective fills
- record the failure
- default to no new trade when a safe canonical decision cannot be formed

Capital protection has priority over simulated opportunity capture.

---

## Shadow Research

Observers may run beside the canonical paper portfolio only as SHADOW.

Shadow information must not alter:

- entries
- exits
- sizing
- ranking
- risk
- portfolio accounting

V3KM opportunity/shock diagnostics may be observed in shadow mode.

Historical V3KX hypotheses may be monitored but remain inactive.

---

## Evaluation Horizon

Primary forward validation horizon:

16 forward weeks

Interim review:

12 completed forward weeks

No retuning is allowed because of the interim result.

If event count is insufficient after 16 weeks, the unchanged epoch may continue.

Insufficient evidence must be reported as insufficient evidence.

---

## Weekly Scoreboard

For every completed calendar week record:

- starting equity
- ending equity
- weekly net P&L EUR
- weekly return percent
- trade count
- winning trades
- losing trades
- best trade R
- worst trade R
- maximum weekly drawdown
- risk-lock events
- data-failure events

Calculate:

- 12-week rolling average weekly P&L
- 16-week rolling average weekly P&L
- losing-week count
- EUR 200+ week count
- worst week
- best week

Daily profit is not required.

Weekly profit is not required.

---

## Master Objective Check

Every review must answer:

1. Did loss and drawdown remain controlled?
2. Was upside and earning capacity preserved?
3. Was performance stable across independent forward events?
4. Did the system move toward EUR 200+ per week on a 12–16 week rolling average?

Profit obtained through excessive risk fails.

Loss reduction that destroys earning capacity also fails.

---

## Final Classification

At the end of the forward period V3KY-001 must be classified as:

- FORWARD VALIDATION PASSED
- FORWARD VALIDATION FAILED
- INSUFFICIENT FORWARD EVIDENCE

V3KY-001 does not automatically authorize live trading.

---

## Scientific Boundary

Historical V3KM through V3KX research may explain hypotheses.

It cannot substitute for V3KY forward evidence.

No historical threshold mining should continue during V3KY-001.

---

## Locked Conclusion

V3KY-001 is forward-only.

Canonical V3KJ remains unchanged.

V3KX remains rejected.

Paper-only execution is mandatory.

Retrospective tuning is prohibited.

Next engineering task:

Build the append-only V3KY paper ledger and immutable forward scoreboard.
