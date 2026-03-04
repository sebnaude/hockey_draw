# GOALS — AI Constraints Parity & Integration

> **Last updated:** 2026-03-03  
> **Branch:** `feat/ai-updates`  
> **Status:** 🔶 Unit tests passing (216/216), AI full-solve INFEASIBLE (investigation in progress)

---

## ⚠️ CRITICAL RULE — DO NOT MODIFY ORIGINAL CONSTRAINTS

**`constraints.py` is NEVER to be edited.**  
The original human-written constraints are the **source of truth**. All fixes, improvements, and refactoring MUST be done in `constraints_ai.py` only. The AI constraints must match the behaviour of the originals — if there's a discrepancy, the AI version is wrong.

- ✅ Edit `constraints_ai.py`
- ✅ Edit `tests/test_ai_constraints_comprehensive.py`
- ❌ **NEVER** edit `constraints.py`
- ❌ **NEVER** edit `test_constraints.py` or `test_constraints_equivalence.py`

---

## Overarching Goal

Bring the AI-enhanced constraint implementations (`constraints_ai.py`) to **full parity** with the original human-written constraints (`constraints.py`), then wire them in as a selectable option for live solver runs. The AI versions should:

1. Enforce **exactly the same scheduling outcomes** as the originals
2. Be **structurally cleaner** and easier to maintain
3. Be **at least as efficient** (ideally faster to solve)

---

## Task List

### Phase 1 — Audit & Documentation

- [x] **Audit report** — Side-by-side comparison of all 18 constraint pairs (`AI_CONSTRAINTS_AUDIT.md`)
- [x] **Test gap analysis** — Comprehensive test suite created
- [x] **Update GOALS.md** — This file

### Phase 2 — Fix AI Constraint Parity Issues

| # | Constraint | Issue | Status |
|---|-----------|-------|--------|
| 1 | `EqualMatchUpSpacingConstraintAI` | **No-op** — variables created but no actual bound constraints added. Fixed: added bound constraints + expanded IntVar domains for negative `min_spacing` | ✅ Fixed |
| 2 | `EnsureBestTimeslotChoicesAI` | Missing slot-number bounding logic and Broadmeadow cap at 6 | ✅ Fixed |
| 3 | `ClubVsClubAlignmentAI` | Missing Sunday field-alignment sub-constraint | ✅ Fixed |
| 4 | `MaximiseClubsPerTimeslotBroadmeadowAI` | Missing dynamic hard minimum (`total_games / 2`) | ✅ Fixed |
| 5 | `PHLAndSecondGradeAdjacencyAI` | Missing same-location/non-adjacent-time enforcement | ✅ Fixed |

### Phase 3 — Integration

- [x] **Add `--ai` flag** to `run.py` and wire AI constraints into `main_staged.py` / `main_simple()`
- [x] **Add `--exclude` support** for AI mode (normalizes names to match both `FooConstraint` and `FooConstraintAI`)
- [x] **Run test solve** — Unit tests pass; full-solve goes INFEASIBLE (see "Current Issue" below)

### Phase 4 — Testing & Validation

- [x] **Comprehensive test suite** — `tests/test_ai_constraints_comprehensive.py` (70 tests, 6 sections)
  - Section 1: 18 feasibility tests (all AI constraints on valid data) ✅
  - Section 2: 13 rejection tests (constraints reject violations) ✅
  - Section 3: 16 deep-dive tests on 5 fixed constraints ✅
  - Section 4: 17 parity tests (all original/AI pairs agree) ✅
  - Section 5: 4 combined feasibility tests ✅
  - Section 6: 1 incremental addition test ✅
- [x] **Fix combined test data** — `num_rounds` uses `max_games_per_grade()` from `utils.py`, `Grade.num_games` set correctly
- [x] **OOM prevention** — `solve()` defaults to `workers=8`, combined tests use 6 weeks / 2 grades to avoid solver OOM
- [x] **Run full test suite** — 216/216 tests passing in 21.08s, zero regressions
- [ ] **AI full-solve parity** — AI constraints go INFEASIBLE on real data; original constraints work
- [x] **Update all documentation** — skill file, README, SYSTEM_OVERVIEW, claude.md

---

## Current Issue: AI Constraints INFEASIBLE on Real Data

**Problem:** When running `--simple --ai` mode with all 16 AI constraints, the solver returns INFEASIBLE at constraint #79 during presolve:
```
INFEASIBLE: 'linear: never in domain'
Unsat after presolving constraint #79: linear { domain: 1 domain: 2 }
```

**What Works:**
- All 216 unit tests pass (including 70 AI-specific tests)
- Individual AI constraints work in isolation
- `EnsureEqualGamesAndBalanceMatchUpsAI` + `FiftyFiftyHomeandAwayAI` together work (UNKNOWN/timeout, not INFEASIBLE)
- Original constraints pass presolve and begin searching

**What Doesn't Work:**
- All 16 AI constraints combined → INFEASIBLE at presolve

**Root Cause:** Unknown — likely a subtle difference in one of the AI constraint implementations that only manifests when combined with other constraints. The `linear { domain: 1 domain: 2 }` error suggests a constraint is creating bounds [1, 2] that conflict with another constraint.

**Next Steps:**
1. Binary search: Add constraints one-by-one to find which one triggers INFEASIBLE
2. Compare constraint logic line-by-line between AI and original for the culprit
3. Fix the subtle difference in `constraints_ai.py`

---

## Key Design Decisions

1. **`constraints.py` is read-only** — the human-written originals are the source of truth and must NEVER be modified
2. **AI constraints are opt-in** via `--ai` flag — original constraints remain the default
3. **`--exclude` flag** works for both original and AI modes
4. **Three constraints excluded from initial AI test run:**
   - `EnsureBestTimeslotChoices` / `EnsureBestTimeslotChoicesAI`
   - `MinimiseClubsOnAFieldBroadmeadow` / `MinimiseClubsOnAFieldBroadmeadowAI`
   - `MaximiseClubsPerTimeslotBroadmeadow` / `MaximiseClubsPerTimeslotBroadmeadowAI`

---

## Resolved Issues

### OOM During Solver Tests
**Problem:** Combined constraint tests with 4 grades + ≥8 weeks caused OOM crashes during `solver.Solve()`. The `EqualMatchUpSpacing` constraint creates O(matchup_pairs × rounds) intermediate IntVars, causing memory explosion during search.

**Solution:** 
- Combined tests (Section 5): 4 grades × 6 weeks × 5 slots — solves in ~2s
- Incremental test (Section 6): 2 grades × 6 weeks × 5 slots with `workers=4` — solves in ~6s
- Default workers reduced to 8 across all tests

### Test Data `num_rounds` Mismatch
**Problem:** `make_standard_data()` originally set `num_rounds = num_weeks`, but `EnsureEqualGamesAndBalanceMatchUps` uses `num_rounds[grade]` as max games per team. With odd team counts, this demanded impossible game counts.

**Solution:** Use `max_games_per_grade(grades, num_weeks)` from `utils.py` — the same function the production solver uses. Also set `Grade.num_games` for `MaxMaitlandHomeWeekends`.

---

## Files Modified

| File | Change |
|------|--------|
| `GOALS.md` | This file — task tracking |
| `AI_CONSTRAINTS_AUDIT.md` | Detailed audit report |
| `run.py` | `--exclude` flag, `--ai` flag |
| `main_staged.py` | AI constraint wiring, `main_simple()` updates |
| `constraints_ai.py` | 5 parity fixes + EqualMatchUpSpacing IntVar domain fix |
| `tests/test_ai_constraints_comprehensive.py` | 70 comprehensive AI constraint tests |
