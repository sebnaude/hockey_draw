<!-- status: ready -->
<!-- owner: unassigned -->
<!-- depends_on: none -->

# spec-007 — Adjacent-grade hard atom → soft; keep only same-grade-same-club hard

**Spec source:** [`docs/todo/GOALS.md` → spec-007](GOALS.md#spec-007--adjacent-grade-hard-atom--soft-constraint-keep-only-same-grade-same-club-hard)

## Why

`ClubGradeAdjacency` currently blocks adjacent-grade teams in the same club from playing the same slot. Convenor experience: this was over-restrictive (parents handle adjacent-grade kids fine; the rule caused infeasibility on tight weeks). The genuinely-fundamental case is **same-grade-same-club** (one parent literally cannot be at two simultaneous games of their own kid's team). Adjacent-grade should become a soft constraint, plus a generic per-team-pair "do not concurrent" soft for the real-world cases (siblings in non-adjacent grades, particular coach conflicts, etc.).

## Definition of Done

1. The existing `ClubGradeAdjacency` atom is **split**:
   - Hard portion: same-club-same-grade-no-concurrency → renamed `SameGradeSameClubNoConcurrency`, severity 1.
   - Soft portion (adjacent-grade): **removed entirely** (not migrated). Adjacent-grade scheduling is freed up.
2. New atom `TeamPairNoConcurrency` (soft, severity 3) reads a new config list `TEAM_PAIR_NO_CONCURRENCY` of `(team1, team2, weight?)` tuples. Per (week, day_slot): penalty = `weight × (var_team1_in_slot + var_team2_in_slot - 1)` clamped at 0.
3. `config/defaults.py::CONSTRAINT_DEFAULTS` adds an empty `TEAM_PAIR_NO_CONCURRENCY = []` default.
4. Registered + wired:
   - `SameGradeSameClubNoConcurrency` in `critical_feasibility` stage.
   - `TeamPairNoConcurrency` in `soft_optimisation` stage.
   - `ClubGradeAdjacency` removed from all stages; marked obsolete in registry.
5. Unit tests:
   - Given a club with two PHL teams, When solving, Then they never play in the same (week, day_slot).
   - Given a club with one PHL and one 2nd grade team (adjacent), When solving, Then they MAY play in the same slot (no hard block).
   - Given `TEAM_PAIR_NO_CONCURRENCY=[('TeamA','TeamB')]`, When solving, Then they don't co-occur if feasibly avoidable.
   - Given a TEAM_PAIR entry where avoidance is infeasible, When solving, Then model is still SAT (objective higher).
6. `docs/system/CONSTRAINT_INVENTORY.md` updated: ClubGradeAdjacency obsolete, two new rows.
7. `docs/operator-ai/CONFIGURATION_REFERENCE.md` documents `TEAM_PAIR_NO_CONCURRENCY`.
8. `docs/operator-human/RULES.md` explains the change to convenor (no more "your adjacent grades can't share a slot").

## Implementation units

### Unit 1 — Split + new atom

- **Files touched:** `constraints/atoms/same_grade_same_club_no_concurrency.py` (new), `constraints/atoms/team_pair_no_concurrency.py` (new), `constraints/registry.py`, `config/defaults.py::DEFAULT_STAGES`, `config/defaults.py::CONSTRAINT_DEFAULTS`.
- **Notes:** the existing `ClubGradeAdjacency` may not have ever been atomized in the same sense as the Phase-3 clusters — verify by reading `constraints/registry.py` and the unified engine. If it routes through a legacy class, the cleanest path is "create the new atoms, remove the old class from any stage, mark obsolete."

### Unit 2 — Tests + docs

- **Files touched:** `tests/atoms/test_same_grade_same_club_no_concurrency.py`, `tests/atoms/test_team_pair_no_concurrency.py`, plus the doc set.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — obsolescence + 2 new rows
- `docs/operator-ai/CONFIGURATION_REFERENCE.md` — TEAM_PAIR_NO_CONCURRENCY docs
- `docs/operator-human/RULES.md` — convenor-facing change explainer
- `docs/todo/GOALS.md` — flip spec-007 status to "done"

## Out of scope

- Coach conflict scheduling (would need coach→teams mapping) — separate plan.
- Distance-aware adjacency (parents at venue A can't make venue B within 30min) — separate plan.
