"""spec-027 regen-soft atom: soft analogue of `ClubVsClubStackedWeekends`.

The hard atom (`constraints/atoms/club_vs_club_stacked_weekends.py`) pins each
grade's Sunday meeting count to a budget (`sum == budget`) and enforces a strict
nested-superset implication chain across grades (a higher-meeting-count grade
must play whenever a lower-count grade plays for the same club-pair that Sunday).

This regen-soft atom mirrors that structure but emits PENALTIES instead of hard
clauses, so the model stays FEASIBLE for ANY X:

  * Budget `sum_w play[g, w] == budget`  →  absolute-deviation penalty
    `dev[pair, g] = |sum_w play - budget|` (IntVar, pinned by two `>=`).
    One penalty UNIT per weekend the grade's Sunday count is off-budget.

  * Implication `lo_play <= hi_play`     →  penalty BoolVar
    `v = lo_play AND NOT hi_play` (a missing stacked coincidence: the lower-count
    grade plays this Sunday but the higher-count grade does not).
    One penalty UNIT per missing coincidence.

All play indicators are built DIRECTLY from X inside this atom (raw BoolVars
channeled with `AddMaxEquality`). The hard atom is NOT applied in a regen run, so
its registry helper vars (under `STACK_PLAY_PREFIX`) won't exist — we never read
them. Structural indicator-defining constraints (channeling, deviation pinning)
are fine; there is NO hard clause forbidding any X pattern.

Budget computation is FORCED-Friday aware, replicated exactly from the hard atom
via `pair_grade_sunday_meetings` (which subtracts FORCED PHL Fridays via the
spec-004 `phl_forced_friday_meetings` helper).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from constraints.atoms.base import Atom
from constraints.atoms._club_vs_club_stacked_shared import (
    enumerate_club_pairs,
    pair_grade_sunday_meetings,
    per_pair_grade_meeting_counts,
    sorted_grades_by_desc_count,
)


# Default penalty weight when `PENALTY_WEIGHTS['regen_clubvsclub_stacked_weekends']`
# is unset. Large (CRITICAL-tier in regen) — the hard original is a CRITICAL
# stacking-structure constraint, so its soft analogue must dominate ordinary
# soft preferences.
REGEN_CLUBVSCLUB_STACKED_WEEKENDS_DEFAULT_WEIGHT = 80000


class ClubVsClubStackedWeekendsRegenSoft(Atom):
    """SOFT analogue of `ClubVsClubStackedWeekends`.

    Per club-pair, per grade: penalise the absolute deviation of the grade's
    Sunday-play count from its budget, and penalise every missing nested-superset
    coincidence (lower-count grade plays without the higher-count grade). Model
    stays feasible for any X.
    """

    canonical_name = 'ClubVsClubStackedWeekendsRegenSoft'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:
        weight = data.get('penalty_weights', {}).get(
            'regen_clubvsclub_stacked_weekends',
            REGEN_CLUBVSCLUB_STACKED_WEEKENDS_DEFAULT_WEIGHT,
        )
        if weight == 0:
            return 0

        all_weeks = sorted({
            key[6]
            for key in X
            if len(key) >= 11 and key[3] == 'Sunday'
        })
        if not all_weeks:
            return 0

        locked_weeks = set(data.get('locked_weeks', set()) or set())
        teams = data.get('teams', []) or []
        team_club = {t.name: t.club.name for t in teams}

        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_clubvsclub_stacked_weekends',
            {'weight': weight, 'penalties': []},
        )

        n = 0
        for pair in enumerate_club_pairs(data):
            club_a, club_b = pair

            # 1. Sunday meeting budgets per grade (FORCED-Friday aware for PHL),
            #    exactly as the hard atom computes them.
            grade_meetings_total = per_pair_grade_meeting_counts(data, pair)
            sunday_budget: Dict[str, int] = {}
            for grade, _total in grade_meetings_total.items():
                sb = pair_grade_sunday_meetings(data, pair, grade)
                if sb > 0:
                    sunday_budget[grade] = sb

            if not sunday_budget:
                continue

            # 2. Collect Sunday vars per (grade, week) DIRECTLY from X. Skip dummy
            #    keys, no-day keys, non-Sunday keys, locked weeks, and keys whose
            #    two teams aren't this exact club pair. (Inlined rather than using
            #    the shared collector so the atom is self-contained and the skip
            #    conditions are explicit per the contract.)
            per_grade_week_vars: Dict[str, Dict[int, List]] = {
                grade: defaultdict(list) for grade in sunday_budget
            }
            for key, var in X.items():
                if len(key) < 11 or not key[3]:
                    continue
                if key[3] != 'Sunday':
                    continue
                grade = key[2]
                if grade not in sunday_budget:
                    continue
                week = key[6]
                if week in locked_weeks:
                    continue
                c1 = team_club.get(key[0])
                c2 = team_club.get(key[1])
                if not c1 or not c2:
                    continue
                if {c1, c2} != {club_a, club_b}:
                    continue
                per_grade_week_vars[grade][week].append(var)

            # 3. Build per-(grade, week) play indicator DIRECTLY (raw BoolVar,
            #    AddMaxEquality channeling). Empty week => forced 0.
            grade_play_by_week: Dict[str, Dict[int, object]] = {}
            for grade in sunday_budget:
                grade_play_by_week[grade] = {}
                for w in all_weeks:
                    vars_for_week = per_grade_week_vars[grade].get(w, [])
                    ind = model.NewBoolVar(
                        f'regen_cvc_play_{club_a}_{club_b}_{grade}_{w}'
                    )
                    if vars_for_week:
                        model.AddMaxEquality(ind, vars_for_week)
                    else:
                        model.Add(ind == 0)
                    grade_play_by_week[grade][w] = ind

            # 4. SOFT budget: dev = |sum_w play[g, w] - budget|. One unit per
            #    weekend off-budget. dev >= sum-budget ; dev >= budget-sum.
            for grade, budget in sunday_budget.items():
                play_vars = [grade_play_by_week[grade][w] for w in all_weeks]
                if not play_vars:
                    continue
                # Upper bound = max possible |play_sum - budget|: play_sum
                # ranges [0, len(all_weeks)], so the deviation maxes at
                # max(budget, len(all_weeks)). (Budget can exceed the week count
                # in the regen-soft case — the hard atom would have raised, this
                # one penalises instead, so the bound must cover it.)
                dev = model.NewIntVar(
                    0, max(budget, len(all_weeks)),
                    f'regen_cvc_budget_dev_{club_a}_{club_b}_{grade}',
                )
                play_sum = sum(play_vars)
                model.Add(dev >= play_sum - budget)
                model.Add(dev >= budget - play_sum)
                bucket['penalties'].append(dev)
                n += 1

            # 5. SOFT nested-superset: per consecutive (hi, lo) grade pair sorted
            #    by descending count, per week, penalise lo_play AND NOT hi_play.
            #    v >= lo - hi ; v <= lo ; v <= 1 - hi.
            sorted_grades = sorted_grades_by_desc_count(sunday_budget)
            for i in range(len(sorted_grades) - 1):
                hi_grade, _hi_count = sorted_grades[i]
                lo_grade, _lo_count = sorted_grades[i + 1]
                for w in all_weeks:
                    hi_play = grade_play_by_week[hi_grade][w]
                    lo_play = grade_play_by_week[lo_grade][w]
                    v = model.NewBoolVar(
                        f'regen_cvc_miss_coincide_{club_a}_{club_b}'
                        f'_{lo_grade}_{hi_grade}_{w}'
                    )
                    model.Add(v >= lo_play - hi_play)
                    model.Add(v <= lo_play)
                    model.Add(v <= 1 - hi_play)
                    bucket['penalties'].append(v)
                    n += 1

        return n


__all__ = [
    'ClubVsClubStackedWeekendsRegenSoft',
    'REGEN_CLUBVSCLUB_STACKED_WEEKENDS_DEFAULT_WEIGHT',
]
