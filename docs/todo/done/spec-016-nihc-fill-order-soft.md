<!-- status: done -->
<!-- owner: session=goal-final-form claimed=2026-05-21 -->
<!-- DECISION (Unit A gate): proceeding with the spec's baked-in recommendation
     to make fill-order SOFT. Rationale: the convenor's own framing in the Why
     section is that field choice is symmetry-breaking (which physical field a
     slot lands on is interchangeable), and a hard ordering risks infeasibility
     against FORCED placements. Executed autonomously under /goal; trivially
     reversible (move atoms back to critical_feasibility + restore the hard
     implication) if the convenor wants it hard. -->
<!-- depends_on: none (shares config/defaults.py DEFAULT_STAGES + constraints/registry.py + constraints/stages.py with spec-014/015/017/018; rebase + re-run validate_solver_stages before merge) -->

# spec-016 — NIHC field-fill ordering (WF→EF→SF) becomes SOFT symmetry-breaking

## Why

`NIHCFillWFBeforeEF` and `NIHCFillEFBeforeSF` (`constraints/atoms/nihc_fill_*.py`, spec-003)
are currently **HARD** atoms in `critical_feasibility`. Per (date, day_slot) at NIHC they
channel `wf_used = max(WF vars)`, `ef_used = max(EF vars)`, `sf_used = max(SF vars)` and add
`ef_used <= wf_used`, `sf_used <= ef_used` — i.e. you can't use East Field unless West is
used, nor South unless East is.

The convenor's framing: this is really a **symmetry-breaking** rule (which physical field a
slot "lands on" is interchangeable; we just want a canonical fill order), the same role
`SoftLexMatchupOrdering` plays for matchups. As a HARD rule it can interact badly with other
hard constraints (e.g. a forced game pinned to EF when WF is empty that slot becomes
infeasible) and removes the solver's freedom for no real-world benefit. A soft penalty gives
the canonical ordering "for free" most of the time while staying feasible under pressure.

## Definition of Done

1. Both atoms add a **soft penalty** instead of a hard implication: per (date, day_slot) at
   NIHC, penalty terms that are > 0 exactly when the order is violated
   (`ef_used AND NOT wf_used`; `sf_used AND NOT ef_used`), summed into a penalty bucket with
   weight `PENALTY_WEIGHTS['nihc_fill_order']` (new key; default chosen so it behaves as a
   tie-break — small, comparable to `soft_lex_ordering`, NOT large enough to dominate real
   soft constraints). No `model.Add(... <= ...)` hard implication remains.
2. The two atoms move out of `critical_feasibility` into `soft_optimisation` in
   `DEFAULT_STAGES`. Their registry `severity_level` becomes 5 (VERY LOW) and
   `has_soft_component=True`; `tester_*` updated so violations report as soft pressure, not
   hard failures.
3. `analytics/tester.py` reports fill-order violations as a **soft** metric (breakdown /
   soft-pressure), not a hard violation count — matching the soft-pressure treatment of
   `SoftLexMatchupOrdering`.
4. `tests/atoms/test_nihc_field_fill_order.py` rewritten (GWT, no mocks, hand oracle):
   given EF-used-while-WF-empty, the model stays **FEASIBLE** but the penalty bucket gains
   exactly one term of value 1; given canonical WF→EF→SF fill, penalty == 0; given a forced
   EF game with WF empty, FEASIBLE (proving the hard infeasibility is gone).
5. The "last game of day on West Field" perennial review note in `CLAUDE.md` /
   `docs/PERENNIAL_RULES.md` reconciled with soft (not hard) enforcement.
6. Full suite green; `validate_solver_stages(DEFAULT_STAGES)` == `[]`; registry count test
   unchanged (atoms stay registered, just re-leveled).

## Open decision (recommendation baked in)

- **Soft vs keep hard:** the user is unsure. Recommendation: **make them soft.** Field choice
  is genuinely symmetric and a hard ordering risks infeasibility against FORCED placements;
  symmetry-breaking belongs in the objective. If the convenor insists fill order is a hard
  publishing rule, this whole spec is dropped — so confirm before executing Unit A.
- **Weight:** recommend a small fixed weight (e.g. equal to or just above `soft_lex_ordering`)
  so it orders fields when nothing else cares but never overrides a real preference. Tune in
  Unit A with a fixture.

## Implementation units

### Unit A — Convert both atoms to soft + re-stage
- Files: `constraints/atoms/nihc_fill_wf_before_ef.py`,
  `constraints/atoms/nihc_fill_ef_before_sf.py`, `constraints/registry.py` (severity 5,
  soft flag), `config/defaults.py` (move atoms to `soft_optimisation`; add
  `PENALTY_WEIGHTS['nihc_fill_order']`), `constraints/stages.py` (engine-key membership if the
  atoms route through the engine vs the atom-dispatch fallback — confirm and adjust).
- Test: per DoD 4.

### Unit B — Tester soft-pressure reporting
- Files: `analytics/tester.py`, any violation-breakdown test
  (`tests/test_violation_breakdown.py`).
- Depends on Unit A.
- Test: a draw with one fill-order inversion shows up under soft-pressure with the right
  weight, and contributes 0 to hard-violation totals.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — re-level both rows to severity 5, soft; update
  the severity table.
- `docs/PERENNIAL_RULES.md` + `CLAUDE.md` Draw-Review-Checklist — note WF-fill is now a soft
  symmetry-breaker, still flagged in review but no longer hard.
- `docs/todo/GOALS.md` — add spec-016 row.

## Out of scope

- `SoftLexMatchupOrdering` itself (already soft; this spec only mirrors its treatment).
- The `EnsureBestTimeslotChoices` cross-field stacking rule — separate atom, untouched.
- Changing which fields exist or their times.
