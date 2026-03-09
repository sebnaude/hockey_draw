# AI Constraints Audit Report

> **Date:** 2026-02-27 (Updated: 2025-01-XX)  
> **Branch:** `feat/ai-updates`  
> **Auditor:** GitHub Copilot  
> **Referenced by:** `GOALS.md`

## Context

The project has two parallel constraint implementations:

- **`constraints/original.py`** — Original human-written constraints (currently used by the solver)
- **`constraints/ai.py`** — AI-enhanced rewrites (currently only exercised in tests)
- **`constraints/soft.py`** — Soft/relaxed versions with slack variables and penalties
- **`constraints/severity.py`** — Severity-based constraint relaxation system

Both `main_staged.py` and `main.py` import **only** from the original constraints. The AI versions are never used in live solves.

This report compares each pair for **outcome equivalence** and **efficiency**.

---

## Severity Levels & Relaxation System

The solver uses a 4-tier severity system for constraint relaxation when infeasibility is detected:

| Level | Name | Behavior | Constraints |
|-------|------|----------|-------------|
| **1** | CRITICAL | Never relaxed | NoDoubleBooking*, EnsureEqualGames*, PHLAndSecondGrade*, FiftyFiftyHomeandAway, Maitland* |
| **2** | HIGH | Structural, club-specific | ClubDayConstraint, AwayAtMaitlandGrouping, TeamConflictConstraint |
| **3** | MEDIUM | Spacing/alignment | EqualMatchUpSpacing, ClubGradeAdjacency, ClubVsClubAlignment |
| **4** | LOW | Soft optimization | EnsureBestTimeslotChoices, MaximiseClubs*, MinimiseClubs*, PreferredTimes |

### Soft Constraint Variants (constraints/soft.py)

Each constraint below has a `*Soft` version with configurable slack and penalty weights:

| Constraint | Severity | Soft Version | Default Penalty | Slack Levels |
|------------|----------|--------------|-----------------|--------------|
| ClubDayConstraint | 2 | ✅ | 100,000 | 0-2 |
| AwayAtMaitlandGrouping | 2 | ✅ | 100,000 | 0-2 |
| TeamConflictConstraint | 2 | ✅ | 100,000 | 0-2 |
| EqualMatchUpSpacing | 3 | ✅ | 50,000 | 0-2 |
| ClubGradeAdjacency | 3 | ✅ | 50,000 | 0-2 |
| ClubVsClubAlignment | 3 | ✅ | 50,000 | 0-2 |
| EnsureBestTimeslotChoices | 4 | ✅ | 10,000 | 0-2 |
| MaximiseClubsPerTimeslotBroadmeadow | 4 | ✅ | 10,000 | 0-2 |
| MinimiseClubsOnAFieldBroadmeadow | 4 | ✅ | 10,000 | 0-2 |
| PreferredTimesConstraint | 4 | ✅ | 10,000 | 0-2 |

**Slack Levels:**
- `0` = Tight (0.5x multiplier, minimal violations)
- `1` = Normal (1.0x multiplier, default)
- `2` = Relaxed (2.0x multiplier, max slack)

### How Relaxation Works (`--relax` flag)

1. Tries solving with all constraints
2. If INFEASIBLE, drops severity level 4 constraints and retests
3. If still INFEASIBLE, drops level 3, then level 2
4. Identifies the blocking severity group
5. Relaxes ALL constraints in that group (slack +1)
6. Solves with ALL constraints together (no partial locks)

---

## Constraint-by-Constraint Comparison

### ✅ NoDoubleBookingTeamsConstraint (Severity 1 - CRITICAL)

| | Original | AI |
|---|---|---|
| **Goal** | No team plays more than once per week | Same |
| **Outcome** | Identical | — |
| **Efficiency** | Adds constraint even when only 1 var exists | AI skips trivially-satisfied single-var groups (`len > 1` guard) |
| **Soft Version** | ❌ None (Level 1 - never relaxed) | — |
| **Verdict** | ✅ Same outcome, AI slightly more efficient |

---

### ✅ NoDoubleBookingFieldsConstraint (Severity 1 - CRITICAL)

| | Original | AI |
|---|---|---|
| **Goal** | No field hosts more than one game per timeslot | Same |
| **Outcome** | Identical | — |
| **Efficiency** | Same approach, AI has `len > 1` guard | — |
| **Soft Version** | ❌ None (Level 1 - never relaxed) | — |
| **Verdict** | ✅ Same outcome, AI slightly more efficient |

---

### ✅ EnsureEqualGamesAndBalanceMatchUps (Severity 1 - CRITICAL)

| | Original | AI |
|---|---|---|
| **Goal** | Each team plays exactly R games; each pair meets base or base+1 times | Same |
| **Outcome** | Identical formulas: `sum == R`, `sum >= base`, `sum <= base + 1` | — |
| **Efficiency** | Original counts `num_teams` by scanning `data['teams']` each time; AI uses `len(teams_in_grade)` | Marginally cleaner |
| **Soft Version** | ❌ None (Level 1 - never relaxed) | — |
| **Verdict** | ✅ Same outcome |

---

### ⚠️ PHLAndSecondGradeAdjacency (Severity 1 - CRITICAL)

| | Original | AI |
|---|---|---|
| **Goal** | PHL and 2nds from same club don't play in adjacent slots at different locations | Same core goal |
| **Difference** | Original also enforces: if same location but *outside* time window, block. AI only blocks adjacent-time + different-location | **Behavioural gap** |
| **Impact** | The original's "same location, far apart in time" block may be unintentional (it would prevent a PHL game and a 2nd grade game at the same venue if 3+ hours apart). Needs human review. |
| **Soft Version** | ❌ None (Level 1 - never relaxed) | — |
| **Verdict** | ⚠️ **AI drops one case** — likely acceptable but needs confirmation |

---

### ✅ PHLAndSecondGradeTimes (Severity 1 - CRITICAL)

| | Original | AI |
|---|---|---|
| **Goal** | No concurrent PHL at Broadmeadow; no PHL+2nd same club at Broadmeadow; ≤3 Friday games at Broadmeadow; exactly 8 Friday games at Gosford (AGM); preferred date penalties | Same |
| **Outcome** | Identical constraints and penalties | — |
| **Efficiency** | AI pre-filters and uses constants (`BROADMEADOW`, `GOSFORD`, `MAX_FRIDAY_GAMES`, `GOSFORD_FRIDAY_GAMES`) | Cleaner |
| **Soft Version** | ❌ None (Level 1 - never relaxed) | — |
| **Verdict** | ✅ Same outcome |

---

### ✅ FiftyFiftyHomeandAway (Severity 1 - CRITICAL)

| | Original | AI |
|---|---|---|
| **Goal** | 50/50 home/away split for Maitland and Gosford teams | Same |
| **Outcome** | Same formula: `home * 2 >= total - 1` and `home * 2 <= total + 1` | — |
| **Efficiency** | AI uses table-driven `AWAY_VENUES` dict, single loop | Cleaner, same constraint count |
| **Soft Version** | ❌ None (Level 1 - never relaxed) | — |
| **Verdict** | ✅ Same outcome |

---

### ✅ TeamConflictConstraint (Severity 2 - HIGH)

| | Original | AI |
|---|---|---|
| **Goal** | Prevent specified team pairs from playing at the same timeslot | Same |
| **Outcome** | Identical | — |
| **Efficiency** | AI pre-groups by slot then applies per conflict pair (fewer iterations) | Slightly faster setup |
| **Soft Version** | ✅ `TeamConflictConstraintSoft` (penalty: 100,000) | — |
| **Verdict** | ✅ Same outcome |

---

### ✅ MaxMaitlandHomeWeekends (Severity 1 - CRITICAL)

| | Original | AI |
|---|---|---|
| **Goal** | Limit playable weekends at away venues | Same |
| **Outcome** | Both use `AddMaxEquality` indicator + `sum <= max_games // 2 + 1` | — |
| **Verdict** | ✅ Same outcome |

---

### ❌ EnsureBestTimeslotChoices (Severity 4 - LOW)

| | Original | AI |
|---|---|---|
| **Goal** | (1) No gaps between used timeslots; (2) Bound slot numbers so games fill from earliest slots, with Broadmeadow capped at 6 | **AI only implements part (1)** |
| **Missing in AI** | The entire `AddDivisionEquality` / slot-number-bounding block. This pushes games into the earliest available timeslots at each venue and caps Broadmeadow at 6 timeslots. | — |
| **Impact** | **High** — without slot bounding, the solver can scatter games across any timeslots, producing impractical schedules. |
| **Soft Version** | ✅ `EnsureBestTimeslotChoicesSoft` (penalty: 10,000) | — |
| **Verdict** | ❌ **AI is incomplete — needs slot-number bounding logic ported** |

---

### ✅ ClubDayConstraint (Severity 2 - HIGH)

| | Original | AI |
|---|---|---|
| **Goal** | Club days: all teams play, intra-club matchups, single field, contiguous slots | Same |
| **Outcome** | Identical logic, AI uses cleaner sub-constraint organisation | — |
| **Soft Version** | ✅ `ClubDayConstraintSoft` (penalty: 100,000) | — |
| **Verdict** | ✅ Same outcome |

---

### ❌ EqualMatchUpSpacingConstraint (Severity 3 - MEDIUM)

| | Original | AI |
|---|---|---|
| **Goal** | Spread matchups evenly across rounds | Same goal, but... |
| **Missing in AI** | The AI creates all the right variables (`K`, `round_sum`, `max_round`, `meets_twice`) but **never adds the actual spacing bound constraints**. The method ends with `constraints_added += 4` and `return` — no `model.Add()` calls for bounds. | — |
| **Impact** | **Critical** — this constraint is a **complete no-op**. Matchups will cluster arbitrarily. |
| **Soft Version** | ✅ `EqualMatchUpSpacingConstraintSoft` (penalty: 50,000) | — |
| **Verdict** | ❌ **AI is broken — no constraints are actually added** |

---

### ✅ ClubGradeAdjacencyConstraint (Severity 3 - MEDIUM)

| | Original | AI |
|---|---|---|
| **Goal** | Prevent adjacent grades from same club playing simultaneously | Same |
| **Difference** | Original has extra `club_dup_games` logic to handle multiple teams in the same grade within a club. AI drops this. | Minor — only affects clubs with 2+ teams in one grade |
| **Soft Version** | ✅ `ClubGradeAdjacencyConstraintSoft` (penalty: 50,000) | — |
| **Verdict** | ✅ Mostly same outcome (minor edge case) |

---

### ⚠️ ClubVsClubAlignment (Severity 3 - MEDIUM)

| | Original | AI |
|---|---|---|
| **Goal** | Align club-vs-club matchups across grades in the same round, and put them on the same field on Sundays | AI only does round alignment |
| **Missing in AI** | The `fields_dict` and Sunday field-consolidation logic. Original ensures that when two clubs' teams coincide on a round, Sunday games are on the same field. | — |
| **Impact** | **Medium** — clubs may have games scattered across fields on Sundays instead of consolidated. |
| **Soft Version** | ✅ `ClubVsClubAlignmentSoft` (penalty: 50,000) | — |
| **Verdict** | ⚠️ **AI drops Sunday field alignment** |

---

### ✅ MaitlandHomeGrouping (Severity 1 - CRITICAL)

| | Original | AI |
|---|---|---|
| **Goal** | Group Maitland games as all-home or all-away per week; no back-to-back home weekends | Same |
| **Outcome** | Identical | — |
| **Soft Version** | ❌ None (Level 1 - never relaxed) | — |
| **Verdict** | ✅ Same outcome |

---

### ✅ AwayAtMaitlandGrouping (Severity 2 - HIGH)

| | Original | AI |
|---|---|---|
| **Goal** | Limit away clubs at Maitland per weekend (hard: ≤3, soft penalty) | Same |
| **Outcome** | Identical | — |
| **Soft Version** | ✅ `AwayAtMaitlandGroupingSoft` (penalty: 100,000) | — |
| **Verdict** | ✅ Same outcome |

---

### ⚠️ MaximiseClubsPerTimeslotBroadmeadow (Severity 4 - LOW)

| | Original | AI |
|---|---|---|
| **Goal** | Maximise club diversity per timeslot at Broadmeadow | Same core goal |
| **Missing in AI** | Original has a dynamic hard minimum: `total_games / 2 + HARD_LIMIT` clubs must be present. AI only has the soft penalty. | — |
| **Note** | Original has a **typo**: `'Satuday'` instead of `'Saturday'`, meaning Saturday games may not actually be filtered in the original either. |
| **Soft Version** | ✅ `MaximiseClubsPerTimeslotBroadmeadowSoft` (penalty: 10,000) | — |
| **Verdict** | ⚠️ **AI drops dynamic hard minimum** |

---

### ✅ MinimiseClubsOnAFieldBroadmeadow (Severity 4 - LOW)

| | Original | AI |
|---|---|---|
| **Goal** | Minimise clubs per field per day at Broadmeadow (hard: ≤5, penalty: \|clubs - 2\|) | Same |
| **Outcome** | Identical | — |
| **Soft Version** | ✅ `MinimiseClubsOnAFieldBroadmeadowSoft` (penalty: 10,000) | — |
| **Verdict** | ✅ Same outcome |

---

### ✅ PreferredTimesConstraint (Severity 4 - LOW)

| | Original | AI |
|---|---|---|
| **Goal** | Penalise scheduling at non-preferred times | Same |
| **Difference** | Original uses two `allowed_keys` orderings; AI does direct key indexing. Functionally equivalent. AI uses `hash(key)` for variable naming (unlikely collision risk). |
| **Soft Version** | ✅ `PreferredTimesConstraintSoft` (penalty: 10,000) | — |
| **Verdict** | ✅ Same outcome |

---

## Summary Table

| Constraint | Severity | Same Outcome? | AI More Efficient? | Soft Version? | Action Needed |
|---|---|---|---|---|---|
| NoDoubleBookingTeams | 1 | ✅ | ✅ Slightly | ❌ | None |
| NoDoubleBookingFields | 1 | ✅ | ✅ Slightly | ❌ | None |
| EnsureEqualGamesAndBalanceMatchUps | 1 | ✅ | ➖ Same | ❌ | None |
| PHLAndSecondGradeAdjacency | 1 | ⚠️ | ➖ Same | ❌ | Review dropped case |
| PHLAndSecondGradeTimes | 1 | ✅ | ✅ Cleaner | ❌ | None |
| FiftyFiftyHomeandAway | 1 | ✅ | ✅ Cleaner | ❌ | None |
| MaxMaitlandHomeWeekends | 1 | ✅ | ➖ Same | ❌ | None |
| MaitlandHomeGrouping | 1 | ✅ | ➖ Same | ❌ | None |
| TeamConflictConstraint | 2 | ✅ | ✅ Slightly | ✅ | None |
| ClubDayConstraint | 2 | ✅ | ✅ Cleaner | ✅ | None |
| AwayAtMaitlandGrouping | 2 | ✅ | ➖ Same | ✅ | None |
| EqualMatchUpSpacing | 3 | ❌ | N/A | ✅ | **Port entire spacing logic** |
| ClubGradeAdjacency | 3 | ✅ | ✅ Cleaner | ✅ | Minor: dup-grade edge case |
| ClubVsClubAlignment | 3 | ⚠️ | N/A | ✅ | **Port field alignment** |
| EnsureBestTimeslotChoices | 4 | ❌ | N/A | ✅ | **Port slot-number bounding** |
| MaximiseClubsPerTimeslotBroadmeadow | 4 | ⚠️ | N/A | ✅ | **Port hard minimum** |
| MinimiseClubsOnAFieldBroadmeadow | 4 | ✅ | ➖ Same | ✅ | None |
| PreferredTimesConstraint | 4 | ✅ | ➖ Same | ✅ | None |

### Critical Fixes Required

1. **`EqualMatchUpSpacingConstraintAI`** — Complete no-op. Must port spacing bounds.
2. **`EnsureBestTimeslotChoicesAI`** — Missing slot-number bounding and Broadmeadow cap.
3. **`ClubVsClubAlignmentAI`** — Missing Sunday field alignment.
4. **`MaximiseClubsPerTimeslotBroadmeadowAI`** — Missing dynamic hard minimum.
5. **`PHLAndSecondGradeAdjacencyAI`** — Missing one enforcement case (needs human review).
