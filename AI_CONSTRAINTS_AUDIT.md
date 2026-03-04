# AI Constraints Audit Report

> **Date:** 2026-02-27  
> **Branch:** `feat/ai-updates`  
> **Auditor:** GitHub Copilot  
> **Referenced by:** `GOALS.md`

## Context

The project has two parallel constraint implementations:

- **`constraints.py`** — Original human-written constraints (currently used by the solver)
- **`constraints_ai.py`** — AI-enhanced rewrites (currently only exercised in tests)

Both `main_staged.py` and `main.py` import **only** from `constraints.py`. The AI versions are never used in live solves.

This report compares each pair for **outcome equivalence** and **efficiency**.

---

## Constraint-by-Constraint Comparison

### ✅ NoDoubleBookingTeamsConstraint

| | Original | AI |
|---|---|---|
| **Goal** | No team plays more than once per week | Same |
| **Outcome** | Identical | — |
| **Efficiency** | Adds constraint even when only 1 var exists | AI skips trivially-satisfied single-var groups (`len > 1` guard) |
| **Verdict** | ✅ Same outcome, AI slightly more efficient |

---

### ✅ NoDoubleBookingFieldsConstraint

| | Original | AI |
|---|---|---|
| **Goal** | No field hosts more than one game per timeslot | Same |
| **Outcome** | Identical | — |
| **Efficiency** | Same approach, AI has `len > 1` guard | — |
| **Verdict** | ✅ Same outcome, AI slightly more efficient |

---

### ✅ EnsureEqualGamesAndBalanceMatchUps

| | Original | AI |
|---|---|---|
| **Goal** | Each team plays exactly R games; each pair meets base or base+1 times | Same |
| **Outcome** | Identical formulas: `sum == R`, `sum >= base`, `sum <= base + 1` | — |
| **Efficiency** | Original counts `num_teams` by scanning `data['teams']` each time; AI uses `len(teams_in_grade)` | Marginally cleaner |
| **Verdict** | ✅ Same outcome |

---

### ⚠️ PHLAndSecondGradeAdjacency

| | Original | AI |
|---|---|---|
| **Goal** | PHL and 2nds from same club don't play in adjacent slots at different locations | Same core goal |
| **Difference** | Original also enforces: if same location but *outside* time window, block. AI only blocks adjacent-time + different-location | **Behavioural gap** |
| **Impact** | The original's "same location, far apart in time" block may be unintentional (it would prevent a PHL game and a 2nd grade game at the same venue if 3+ hours apart). Needs human review. |
| **Verdict** | ⚠️ **AI drops one case** — likely acceptable but needs confirmation |

---

### ✅ PHLAndSecondGradeTimes

| | Original | AI |
|---|---|---|
| **Goal** | No concurrent PHL at Broadmeadow; no PHL+2nd same club at Broadmeadow; ≤3 Friday games; preferred date penalties | Same |
| **Outcome** | Identical constraints and penalties | — |
| **Efficiency** | AI pre-filters and uses constants (`BROADMEADOW`, `MAX_FRIDAY_GAMES`) | Cleaner |
| **Verdict** | ✅ Same outcome |

---

### ✅ FiftyFiftyHomeandAway

| | Original | AI |
|---|---|---|
| **Goal** | 50/50 home/away split for Maitland and Gosford teams | Same |
| **Outcome** | Same formula: `home * 2 >= total - 1` and `home * 2 <= total + 1` | — |
| **Efficiency** | AI uses table-driven `AWAY_VENUES` dict, single loop | Cleaner, same constraint count |
| **Verdict** | ✅ Same outcome |

---

### ✅ TeamConflictConstraint

| | Original | AI |
|---|---|---|
| **Goal** | Prevent specified team pairs from playing at the same timeslot | Same |
| **Outcome** | Identical | — |
| **Efficiency** | AI pre-groups by slot then applies per conflict pair (fewer iterations) | Slightly faster setup |
| **Verdict** | ✅ Same outcome |

---

### ✅ MaxMaitlandHomeWeekends

| | Original | AI |
|---|---|---|
| **Goal** | Limit playable weekends at away venues | Same |
| **Outcome** | Both use `AddMaxEquality` indicator + `sum <= max_games // 2 + 1` | — |
| **Verdict** | ✅ Same outcome |

---

### ❌ EnsureBestTimeslotChoices

| | Original | AI |
|---|---|---|
| **Goal** | (1) No gaps between used timeslots; (2) Bound slot numbers so games fill from earliest slots, with Broadmeadow capped at 6 | **AI only implements part (1)** |
| **Missing in AI** | The entire `AddDivisionEquality` / slot-number-bounding block. This pushes games into the earliest available timeslots at each venue and caps Broadmeadow at 6 timeslots. | — |
| **Impact** | **High** — without slot bounding, the solver can scatter games across any timeslots, producing impractical schedules. |
| **Verdict** | ❌ **AI is incomplete — needs slot-number bounding logic ported** |

---

### ✅ ClubDayConstraint

| | Original | AI |
|---|---|---|
| **Goal** | Club days: all teams play, intra-club matchups, single field, contiguous slots | Same |
| **Outcome** | Identical logic, AI uses cleaner sub-constraint organisation | — |
| **Verdict** | ✅ Same outcome |

---

### ❌ EqualMatchUpSpacingConstraint

| | Original | AI |
|---|---|---|
| **Goal** | Spread matchups evenly across rounds | Same goal, but... |
| **Missing in AI** | The AI creates all the right variables (`K`, `round_sum`, `max_round`, `meets_twice`) but **never adds the actual spacing bound constraints**. The method ends with `constraints_added += 4` and `return` — no `model.Add()` calls for bounds. | — |
| **Impact** | **Critical** — this constraint is a **complete no-op**. Matchups will cluster arbitrarily. |
| **Verdict** | ❌ **AI is broken — no constraints are actually added** |

---

### ✅ ClubGradeAdjacencyConstraint

| | Original | AI |
|---|---|---|
| **Goal** | Prevent adjacent grades from same club playing simultaneously | Same |
| **Difference** | Original has extra `club_dup_games` logic to handle multiple teams in the same grade within a club. AI drops this. | Minor — only affects clubs with 2+ teams in one grade |
| **Verdict** | ✅ Mostly same outcome (minor edge case) |

---

### ⚠️ ClubVsClubAlignment

| | Original | AI |
|---|---|---|
| **Goal** | Align club-vs-club matchups across grades in the same round, and put them on the same field on Sundays | AI only does round alignment |
| **Missing in AI** | The `fields_dict` and Sunday field-consolidation logic. Original ensures that when two clubs' teams coincide on a round, Sunday games are on the same field. | — |
| **Impact** | **Medium** — clubs may have games scattered across fields on Sundays instead of consolidated. |
| **Verdict** | ⚠️ **AI drops Sunday field alignment** |

---

### ⚠️ MaitlandHomeGrouping

| | Original | AI |
|---|---|---|
| **Goal** | Group Maitland games as all-home or all-away per week; no back-to-back home weekends | Same |
| **Outcome** | Identical | — |
| **Verdict** | ✅ Same outcome |

---

### ✅ AwayAtMaitlandGrouping

| | Original | AI |
|---|---|---|
| **Goal** | Limit away clubs at Maitland per weekend (hard: ≤3, soft penalty) | Same |
| **Outcome** | Identical | — |
| **Verdict** | ✅ Same outcome |

---

### ⚠️ MaximiseClubsPerTimeslotBroadmeadow

| | Original | AI |
|---|---|---|
| **Goal** | Maximise club diversity per timeslot at Broadmeadow | Same core goal |
| **Missing in AI** | Original has a dynamic hard minimum: `total_games / 2 + HARD_LIMIT` clubs must be present. AI only has the soft penalty. | — |
| **Note** | Original has a **typo**: `'Satuday'` instead of `'Saturday'`, meaning Saturday games may not actually be filtered in the original either. |
| **Verdict** | ⚠️ **AI drops dynamic hard minimum** |

---

### ✅ MinimiseClubsOnAFieldBroadmeadow

| | Original | AI |
|---|---|---|
| **Goal** | Minimise clubs per field per day at Broadmeadow (hard: ≤5, penalty: \|clubs - 2\|) | Same |
| **Outcome** | Identical | — |
| **Verdict** | ✅ Same outcome |

---

### ✅ PreferredTimesConstraint

| | Original | AI |
|---|---|---|
| **Goal** | Penalise scheduling at non-preferred times | Same |
| **Difference** | Original uses two `allowed_keys` orderings; AI does direct key indexing. Functionally equivalent. AI uses `hash(key)` for variable naming (unlikely collision risk). |
| **Verdict** | ✅ Same outcome |

---

## Summary Table

| Constraint | Same Outcome? | AI More Efficient? | Action Needed |
|---|---|---|---|
| NoDoubleBookingTeams | ✅ | ✅ Slightly | None |
| NoDoubleBookingFields | ✅ | ✅ Slightly | None |
| EnsureEqualGamesAndBalanceMatchUps | ✅ | ➖ Same | None |
| PHLAndSecondGradeAdjacency | ⚠️ | ➖ Same | Review dropped case |
| PHLAndSecondGradeTimes | ✅ | ✅ Cleaner | None |
| FiftyFiftyHomeandAway | ✅ | ✅ Cleaner | None |
| TeamConflictConstraint | ✅ | ✅ Slightly | None |
| MaxMaitlandHomeWeekends | ✅ | ➖ Same | None |
| EnsureBestTimeslotChoices | ❌ | N/A | **Port slot-number bounding** |
| ClubDayConstraint | ✅ | ✅ Cleaner | None |
| EqualMatchUpSpacing | ❌ | N/A | **Port entire spacing logic** |
| ClubGradeAdjacency | ✅ | ✅ Cleaner | Minor: dup-grade edge case |
| ClubVsClubAlignment | ⚠️ | N/A | **Port field alignment** |
| MaitlandHomeGrouping | ✅ | ➖ Same | None |
| AwayAtMaitlandGrouping | ✅ | ➖ Same | None |
| MaximiseClubsPerTimeslotBroadmeadow | ⚠️ | N/A | **Port hard minimum** |
| MinimiseClubsOnAFieldBroadmeadow | ✅ | ➖ Same | None |
| PreferredTimesConstraint | ✅ | ➖ Same | None |

### Critical Fixes Required

1. **`EqualMatchUpSpacingConstraintAI`** — Complete no-op. Must port spacing bounds.
2. **`EnsureBestTimeslotChoicesAI`** — Missing slot-number bounding and Broadmeadow cap.
3. **`ClubVsClubAlignmentAI`** — Missing Sunday field alignment.
4. **`MaximiseClubsPerTimeslotBroadmeadowAI`** — Missing dynamic hard minimum.
5. **`PHLAndSecondGradeAdjacencyAI`** — Missing one enforcement case (needs human review).
