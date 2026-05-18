<!-- status: ready -->
<!-- owner: unassigned -->
<!-- depends_on: none (spec-004 helper landed in 8ed24f9 — `_phl_forced_friday_helper.py`) -->

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
