<!-- status: ready -->
<!-- owner: session=none claimed=none -->
<!-- depends_on: none (shares config/defaults.py DEFAULT_STAGES with spec-014/015/016/018; rebase + re-run validate_solver_stages before merge) -->

# spec-017 — Promote `EqualMatchUpSpacing` to HARD (beside `BalancedByeSpacing`); keep byes a separate atom

## Why

`EqualMatchUpSpacing` has both a HARD component (`unified.py::_matchup_spacing_hard` —
forbid any pair of meetings whose `gap = r2 - r1 <= S`, `S = effective_spacing(T, base_slack,
config_slack)` from `constraints/atoms/_spacing.py`) and a SOFT component
(`_matchup_spacing_soft` — sliding-window density penalty). But in `DEFAULT_STAGES` it sits
**only** in `soft_optimisation`, which is `soft_only:True`. Per
`constraints/stages.py::apply_solver_stage`, a `soft_only` stage calls `apply_stage_2_soft()`
but **skips** `apply_stage_1_hard()` — so the HARD spacing rule is **never applied in
production today.** Repeat meetings can land in adjacent rounds with only a soft penalty.

The convenor wants matchup spacing enforced as hard (with slack still available via
`--slack`), sitting right after `BalancedByeSpacing` in the hard stage — byes and matchup
repeats are the same "spread it out" intent and the bye atom is already hard there.

On the merge question (point 4): `BalancedByeSpacing` already shares the `_spacing.py`
helpers (`ideal_bye_gap`, the same pairwise-forbidden-gap structure). They differ in three
real ways: (a) byes use `ideal_bye_gap(R, byes)` vs `ideal_gap(T)`; (b) byes have their own
slack key `BalancedByeSpacing` so the convenor can loosen byes without loosening matchups;
(c) byes need per-(team, round) indicator BoolVars (`B = 1 - sum(plays)`), whereas matchups
reuse existing pair-var sums. Folding them into one atom would couple the two slack knobs and
add no variable savings. **Recommendation: keep them as two atoms sharing `_spacing.py`** —
which is the current factoring. This spec therefore does NOT merge; it only promotes matchup
spacing to hard.

## Definition of Done

1. `EqualMatchUpSpacing` is **moved** (not duplicated) from `soft_optimisation` to
   `critical_feasibility` in `DEFAULT_STAGES`, placed immediately after `BalancedByeSpacing`.
   `validate_solver_stages` still passes (atom appears in exactly one stage).
2. Because `apply_solver_stage` always runs `apply_stage_2_soft()`, both the HARD and SOFT
   spacing parts now apply in production. Confirm via a test that, on a fixture, both
   `_matchup_spacing_hard` adds ≥1 hard constraint AND the `EqualMatchUpSpacing` soft penalty
   bucket is populated, in a single `critical_feasibility`+`soft_optimisation` run.
3. A regression test proves the hard rule now bites: a fixture with a forced repeat meeting at
   `gap <= S` is **INFEASIBLE** after the move (was FEASIBLE-with-penalty before), and the
   same meeting at `gap == S+1` is FEASIBLE. Hand-compute `S = ideal_gap(T)` for the fixture's
   grade size and assert against it.
4. `--slack EqualMatchUpSpacingConstraint N` still loosens `S` by N (clamped ≥ 0) in the hard
   path — covered by a slack test (e.g. `S=5` at slack 0, `S=3` at slack 2).
5. `BalancedByeSpacing` is untouched and remains a separate hard atom in `critical_feasibility`
   with its own slack key — documented as a deliberate decision, not an oversight.
6. Full suite green; severity table / inventory updated to show `EqualMatchUpSpacing` as a
   hard (severity-1) production constraint.

## Implementation units

### Unit A — Re-stage matchup spacing
- Files: `config/defaults.py` (`DEFAULT_STAGES`: remove `EqualMatchUpSpacing` from
  `soft_optimisation`, add to `critical_feasibility` after `BalancedByeSpacing`).
- Test: `validate_solver_stages == []`; integration test per DoD 2 + 3 + 4 on the 2026
  fixture (small grade so `S` is hand-computable).

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — note `EqualMatchUpSpacing` HARD part is now applied
  in production (correct the spec-008 row which describes HARD+SOFT but the production wiring
  only ran SOFT); keep the `BalancedByeSpacing` separate-atom rationale.
- `docs/system/STAGES.md` — update the stage listing.
- `docs/DRAW_RULES.md` — matchup spacing is now hard with `--slack` relief.
- `CLAUDE.md` — update the severity table / constraint-slack section (spacing now hard in prod).
- `docs/todo/GOALS.md` — add spec-017 row; record the "keep byes separate" decision.

## Out of scope

- Merging byes into the matchup-spacing atom (explicitly rejected above — keep separate).
- Changing the spacing formula / `ideal_gap` semantics (spec-008 already settled those).
- Bye-spacing slack behaviour (`BalancedByeSpacing` slack key) — unchanged.
