<!-- status: done -->
<!-- owner: session=spec-005-opus-2026-05-18 claimed=2026-05-18T00:00:00Z -->
<!-- depends_on: none (spec-004 helper landed in 8ed24f9 — `_phl_forced_friday_helper.py`) -->
<!-- absorbs: spec-011 (ClubVsClubFieldLimit penalty scaling becomes structurally moot — see "spec-011 resolution" section below) -->

# spec-005 — ClubVsClubAlignment as precise grade-stacking with co-location

**Spec source:** [`docs/todo/GOALS.md` → spec-005](GOALS.md#spec-005--clubvsclubalignment-stacks-lower-grades-into-upper-grade-weekends-with-club-day-like-co-location)

> **Coordination:** Shares helper `_phl_forced_friday_helper.py` with spec-004. Read spec-004's coordination note before starting.

## Why

The current `ClubVsClubAlignment` cluster (4 atoms from Phase 3c) gives a loose "weekends where ≥X grades coincide for this club pair" outcome. It misses the precise structure that the convenor actually wants: a *stacked* weekend layout where each higher-numbered grade-set is a strict superset of the next, peeling off the smallest count at each layer. It also misses club-day-like co-location (back-to-back, same field) for the stacked grades. And PHL Friday-night forced games are ignored — they consume the meeting count but can't satisfy the Sunday stack.

## Definition of Done

1. New atom cluster `ClubVsClubStackedAlignment` in `constraints/atoms/`, replacing the four Phase-3c atoms:
   - Old: `ClubVsClubCoincidence`, `ClubVsClubFieldLimit`, `ClubVsClubDeficitPenalty`, `PHLAnd2ndBackToBackSameField`.
   - New atoms (suggested split — final names up to implementer):
     - `ClubVsClubStackedWeekends` — for each unordered club pair `(A,B)`, asserts the stacking structure (PHL-most-weekends ⊇ 2nd ⊇ 3rd ⊇ ...).
     - `ClubVsClubStackedCoLocation` — on each stacked weekend, the matched grades' games for that pair are back-to-back same field (club-day-like).
     - `ClubVsClubStackedPHLSundayBudget` — PHL-Sunday-available count drives the PHL row of the stack (NOT total PHL meetings, since FORCED Fridays subtract).
2. Stacking math: for club pair (A,B) with per-grade meeting counts `c = [c_PHL, c_2nd, ..., c_6th]` (zeros allowed for grades they don't both play), force:
   - Weekends where all grades-with-count-≥k coincide = `c[k] - c[k+1]` (for sorted-descending `c`).
3. Per-grade meeting count is the **number of distinct matchups** for the club pair in that grade — handles "multiple teams per club per grade" correctly.
4. PHL Sunday budget = `phl_meetings_total(A,B) - phl_forced_friday_meetings(A,B)`. The forced-Friday helper from spec-004 is reused.
5. Co-location: on each stacked weekend, all PHL+2nd+...+lower-grade games for (A,B) are back-to-back (no slot gaps) on the same field. Reuses the `ClubDaySameField` / `ClubDayContiguousSlots` helper-var kinds, OR declares a parallel pair (`cvc_stack_field_used`, `cvc_stack_slot_used`).
6. Old atoms in registry are marked obsolete (kept for parity reference, removed from `DEFAULT_STAGES`).
7. Unit tests (real CP-SAT, no mocks):
   - Given (Maitland, Norths) with PHL=4, 2nd=3, 3rd=2, 4th=2, 5th=1, 6th=0, no FORCED Fridays, When solved, Then exactly 1 weekend has {PHL,2nd,3rd,4th,5th} co-located, 1 weekend {PHL,2nd,3rd,4th}, 1 weekend {PHL,2nd}, 1 weekend {PHL} — totals match per-grade counts.
   - Given PHL=4 with 2 FORCED Fridays, other grades unchanged, When solved, Then PHL Sunday weekends = 2, and stacking math uses 2 not 4 for PHL.
   - Given Norths fields 2 PHL teams (rare), When computing meeting counts, Then count is 2 × matchups (each Norths team plays each other club's PHL team), not 1.
   - Given a stacked weekend, When inspecting solution, Then all participating games are on the same field with abs(slot_i - slot_j) == 1 for adjacent grade games.
8. `docs/system/CONSTRAINT_INVENTORY.md` updated: 4 atoms marked obsolete, 3 new atoms added with full per-atom detail.
9. `docs/operator-human/RULES.md` describes "when our two clubs play, expect the bigger-grade group to bring along the smaller grades for the day, back-to-back on one field."

## Implementation units

### Unit 1 — Helper (if not shipped by spec-004 yet)

See spec-004 Unit 1. If spec-004 ships first, this becomes a no-op — `_phl_forced_friday_helper.py` already exists.

### Unit 2 — Stacking math + atom

- **Files touched:** `constraints/atoms/club_vs_club_stacked_weekends.py` (new), `constraints/atoms/club_vs_club_stacked_co_location.py` (new), `constraints/atoms/_club_vs_club_stacked_shared.py` (new helpers), `constraints/registry.py`, `config/defaults.py::DEFAULT_STAGES`.
- **Helper var kind:** `cvc_stack_weekend_active` BoolVar per (club_pair, week) — declared via registry so both atoms share.

### Unit 3 — Retire the old four atoms

- **Files touched:** `constraints/registry.py` (mark obsolete), `config/defaults.py::DEFAULT_STAGES` (remove from stages), `constraints/unified.py` (dispatch update if needed).
- Do NOT delete the atom files — keep as parity reference until next major cleanup.

### Unit 4 — Docs

- **Files touched:** `docs/system/CONSTRAINT_INVENTORY.md`, `docs/operator-human/RULES.md`.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — 4 obsolescence notes, 3 new rows with per-atom detail
- `docs/operator-human/RULES.md` — stacked weekends explainer
- `docs/system/COUNT_ADJUSTERS.md` — PHL Sunday budget formula (if not done by spec-004)
- `docs/todo/GOALS.md` — flip spec-005 status to "done"

## Out of scope

- Triadic stacking (three clubs all playing each other same weekend) — not currently desired.
- Cross-grade stacking (e.g. PHL vs Tigers, 2nd vs Wests, same weekend) — different concern.
- Removing the old atom files entirely — separate cleanup plan.

## spec-011 resolution

spec-011 (`ClubVsClubFieldLimit` penalty scaling) is **absorbed** by this
plan and marked done without separate code:

- spec-011 wanted the soft penalty to scale with games on the secondary
  field rather than a flat "fields used > 1." `ClubVsClubStackedCoLocation`
  enforces `sum(field_used) <= 1` HARD on every stacked weekend — so the
  number of secondary-field games is structurally zero, not penalised.
- spec-011's double-up handling (multiple teams per club per grade) is
  handled correctly by `per_pair_grade_matchup_counts` (distinct matchups,
  not 1). The single-field cap holds regardless of matchup count.
- spec-011's structural-feasibility carve-out (don't penalise unavoidable
  spillover) becomes: the atom raises `ValueError` when a pair's Sunday
  budget exceeds available weeks — surfaced as a hard infeasibility the
  convenor can resolve via FORCED entries or removing matchups, rather
  than as a noisy soft penalty.

The convenor's intent in spec-005 was explicitly stricter than spec-011
(spec-005 says "same field, back-to-back" rather than "penalise excess
field use"), so absorbing is the right call.

## Shipped — files of record

- `constraints/atoms/_club_vs_club_stacked_shared.py` — shared helpers
  (`per_pair_grade_matchup_counts`, `per_pair_grade_meeting_counts`,
  `pair_grade_sunday_meetings`, `enumerate_club_pairs`,
  `collect_pair_grade_week_vars`, `collect_pair_week_sunday_vars`,
  `sorted_grades_by_desc_count`).
- `constraints/atoms/club_vs_club_stacked_weekends.py` — the stacking atom.
- `constraints/atoms/club_vs_club_stacked_co_location.py` — the co-location atom.
- `constraints/atoms/_phl_forced_friday_helper.py` — extended with
  `phl_forced_friday_meetings(data, a, b)` per-pair helper.
- `constraints/registry.py` — 2 new entries; 4 Phase-3c entries annotated
  OBSOLETE; helper-var catalog gets 4 new kinds.
- `config/defaults.py::DEFAULT_STAGES` — `club_alignment` stage swaps the
  4 old atoms for the 2 new ones; `soft_optimisation` drops
  `ClubVsClubDeficitPenalty` (now structurally moot).
- `constraints/stages.py::apply_solver_stage` — fixed to thread the
  engine's persistent `HelperVarRegistry` through Atom-subclass dispatch
  so spec-005's two atoms share registry state correctly (was: ephemeral
  registry per atom call, would have broken cross-atom lookups).
- `tests/atoms/test_club_vs_club_stacked_alignment.py` (new, 18 tests).
- `tests/atoms/test_phl_forced_friday_helper.py` — extended with 7 tests
  for `phl_forced_friday_meetings`.
- `tests/atoms/club_vs_club_stacked_fixture.py` (new fixture).
- `tests/test_constraint_registry.py` — entry-count tripwire updated
  (45 → 47).
- `docs/system/CONSTRAINT_INVENTORY.md` — 4 OBSOLETE notes, 2 new rows,
  atomization summary updated.
- `docs/operator-human/RULES.md` — Rule 13 rewritten as "Club vs Club
  Stacked Alignment."
- `docs/system/COUNT_ADJUSTERS.md` — `phl_forced_friday_meetings`
  documented + spec-005 PHL Sunday budget formula.
- `docs/todo/GOALS.md` — spec-005 + spec-011 flipped to done.

## Deviation from the plan (worth noting)

- The plan suggested 3 atoms: Weekends, CoLocation, **PHLSundayBudget**.
  Shipped 2 atoms — the PHL Sunday budget is a per-pair `int` computed by
  `pair_grade_sunday_meetings(data, pair, 'PHL')` and inlined into the
  Weekends atom's `sum == budget` constraint. A separate atom for budget
  would add no behavior; the budget is a single value per (pair, grade),
  not a separable constraint. Documented in
  `_club_vs_club_stacked_shared.py` module docstring.
- The plan said co-location uses "back-to-back same field." Shipped
  enforcement is **stricter** — single field for every stacked weekend
  AND contiguous slots (not just one back-to-back pair). The convenor's
  RULES.md wording aligns with the stricter enforcement.
- Stacking constraint shipped as `play[g_lower, w] <= play[g_higher, w]`
  pairwise implication chain (sorted by descending Sunday budget,
  alphabetical tie-break) rather than the explicit "`c[k] - c[k+1]`
  weekends per layer" enumeration in the plan. Equivalent under the
  `sum == budget` pins — the layout layers emerge automatically.
