<!-- status: not_ready -->
<!-- owner: unassigned -->
<!-- depends_on: spec-005 (must resolve before this is pickable) -->

<!--
NOT READY rationale: This plan is contingent on spec-005 (ClubVsClubStackedAlignment).
If spec-005 is shipped first, this entire spec is absorbed into it (the field-penalty
scaling + double-up requirements move into spec-005's DoD as sub-criteria), and this
file moves to docs/todo/done/ with a "absorbed by spec-005" note instead of new code.

If spec-005 is explicitly abandoned or deferred indefinitely, flip this spec to ready
and ship it as the interim improvement it describes.

Either way, do NOT pick this up while spec-005's outcome is undecided.
-->

# spec-011 — ClubVsClubFieldLimit penalty scales with games on 2nd field; double-up handling

**Spec source:** Convenor request 2026-05-18 (research session).

> **Coordination:** spec-005 (`ClubVsClubStackedAlignment`) replaces this whole atom cluster. If spec-005 is picked up first, this spec is absorbed into it — the implementer of spec-005 should integrate the scaling/double-up requirements below. Otherwise: ship as an interim improvement and the spec-005 implementer carries the rules forward.

## Why

`ClubVsClubFieldLimit` (`constraints/atoms/club_vs_club_field_limit.py`) currently:

- **Hard:** when a club-pair coincides on a round, `nf <= 2` (max 2 fields).
- **Soft:** `field_excess >= nf - 1` — a *fixed* penalty of "fields above 1." With `nf = 2` and one game on the 2nd field, penalty = 1. With `nf = 2` and seven games on the 2nd field, penalty is still 1. That doesn't match the convenor's intent: when many games spill onto a second field, the penalty should grow.

Also, the atom does not gracefully handle the case where:

- A club fields 2+ teams in the same grade (double-ups). Each is a distinct matchup; the field-limit math should treat them as additive contributions to the second-field overflow, not collapse them.
- The number of teams across grades exceeds the number of timeslots on one field — in which case "stay on 1 field" is structurally infeasible and the constraint shouldn't penalise the unavoidable spill.

## Definition of Done

1. `ClubVsClubFieldLimit` (or its successor) maintains the hard `nf <= 2` cap but the soft penalty becomes proportional to `games_on_secondary_field` (the smaller of the two field counts), NOT to `nf - 1`. Concretely:
   - Build per-(grade, club_pair, round, field) game-count IntVars (`gcnt_<field>`).
   - Identify "primary" field = field with `max(gcnt)`; "secondary" field = the other.
   - Penalty = `gcnt_secondary` (the count, not a boolean). Weight = `penalty_weights['ClubVsClubAlignmentField']` (unchanged).
2. **Double-up correctness:** when one club has multiple teams in a grade, `games_on_secondary_field` adds across all distinct matchups for that club-pair on the round. Verify by test (fixture: club A with two 3rd-grade teams, club B with one — total 2 matchups; on a coincide round, the penalty correctly sees 0/1/2 games on the secondary field).
3. **Structural feasibility carve-out:** for each (grade, round) compute `min_required_fields(club_pair) = ceil(matchup_count / slots_per_field_per_day)`. If `min_required_fields > 1`, the hard `nf <= 2` cap is preserved but the soft penalty subtracts `min_required_fields - 1` from `games_on_secondary_field` before applying. This stops the solver getting nagged about a spillover that can't be avoided.
4. **Tests** in `tests/atoms/test_club_vs_club_field_limit_scaling.py`:
   - Single field, one game: penalty = 0.
   - Two fields, one game on field 2: penalty = 1.
   - Two fields, five games on field 2: penalty = 5 (was: 1).
   - Double-up case (A has 2× 3rd-grade teams): penalty correctly aggregates per matchup.
   - Structural-infeasibility case (e.g. 4 matchups, 3 slots on field 1): penalty subtracts the unavoidable spillover.
5. Registry / stage wiring unchanged (still in `club_alignment` stage).
6. If spec-005 is shipped first, this spec is **closed without code** and these requirements move into spec-005's DoD as a sub-list under "Co-location rules + penalty scaling."

## Implementation units

### Unit 1 — Atom refactor

- **Files touched:** `constraints/atoms/club_vs_club_field_limit.py`, possibly `constraints/atoms/_club_vs_club_shared.py` (helper for structural-feasibility math).

### Unit 2 — Tests

- **Files touched:** `tests/atoms/test_club_vs_club_field_limit_scaling.py` (new).

### Unit 3 — Docs

- **Files touched:** `docs/system/CONSTRAINT_INVENTORY.md` (`ClubVsClubFieldLimit` row updated), `docs/operator-human/RULES.md` (plain-English description of the field-overflow handling).

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md`
- `docs/operator-human/RULES.md`
- `docs/todo/GOALS.md` — add spec-011 row, flip to "done"

## Out of scope

- Hard limit on `games_on_secondary_field` — convenor wants only the soft penalty to scale.
- Changing the `nf <= 2` hard cap.
