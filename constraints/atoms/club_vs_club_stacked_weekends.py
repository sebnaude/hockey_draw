"""spec-005 atom 1/2: per-club-pair stacked-weekends structure.

For each unordered pair of clubs (A, B) the atom pins each grade's Sunday
meeting count and forces a strict nested-superset structure across weeks:
if a lower-count grade plays for the pair on a Sunday, every higher-count
grade also plays on the same Sunday.

The PHL Sunday budget = `total_phl_meetings(A,B) - phl_forced_friday_meetings(A,B)`
— FORCED PHL Friday games consume the matchup count but cannot satisfy
Sunday stacking, so they're subtracted via the spec-004 helper module.

## Replaces (kept as parity reference, removed from DEFAULT_STAGES)

- `ClubVsClubCoincidence` — the loose "rounds where 2 grades coincide"
  count constraint.
- `ClubVsClubDeficitPenalty` — the soft "miss-the-target" penalty (now
  hard via the count == budget constraint on each play indicator sum).
- `PHLAnd2ndBackToBackSameField` — the PHL/2nd-specific back-to-back rule
  is now generalised to every stacked grade via the co-location atom.

## Stacking math (DoD #2)

Given grades with Sunday budgets `c[g_0] >= c[g_1] >= ... >= c[g_{n-1}]`
sorted descending:

- `sum_w play[g_k, w] == c[g_k]`                  (per grade)
- `play[g_{k+1}, w] <= play[g_k, w]`              (consec implication)

The peel-off layout (`c[g_k] - c[g_{k+1}]` weekends where exactly
`{g_0..g_k}` play) emerges from the implication chain — no need to
enumerate layers explicitly.

## Helper-var registry (spec-005)

Registers `play[pair, grade, week]` indicators under
`(STACK_PLAY_PREFIX, club_pair, grade, week)` so the co-location atom
can read them back. Each indicator = OR of the Sunday vars for that
(pair, grade, week) tuple.
"""
from __future__ import annotations

from constraints.atoms.base import Atom
from constraints.atoms._club_vs_club_stacked_shared import (
    STACK_PLAY_PREFIX,
    collect_pair_grade_week_vars,
    enumerate_club_pairs,
    pair_grade_sunday_meetings,
    per_pair_grade_meeting_counts,
    sorted_grades_by_desc_count,
)


class ClubVsClubStackedWeekends(Atom):
    """Stacking structure: pin per-(pair, grade) Sunday meetings + enforce
    nested-superset implication across grades sorted by descending count."""

    canonical_name = 'ClubVsClubStackedWeekends'
    atom_group = 'ClubVsClubStackedAlignment'

    def apply(self, model, X, data, registry) -> int:
        n = 0
        all_weeks = sorted({
            key[6]
            for key in X
            if len(key) >= 11 and key[3] == 'Sunday'
        })
        if not all_weeks:
            return 0

        for pair in enumerate_club_pairs(data):
            # 1. Compute Sunday meeting budgets per grade for this pair.
            #    PHL budget subtracts FORCED Fridays; others = full count.
            grade_meetings_total = per_pair_grade_meeting_counts(data, pair)
            sunday_budget: dict = {}
            for grade, _total in grade_meetings_total.items():
                sb = pair_grade_sunday_meetings(data, pair, grade)
                if sb > 0:
                    sunday_budget[grade] = sb

            if not sunday_budget:
                continue

            # 2. Build per-(grade, week) `play` indicator.
            #    play[g, w] = OR of all Sunday vars for (pair, grade, week).
            #    If no vars exist for a (grade, week), play[g, w] is forced to
            #    0 by the empty-OR channeling (no var can satisfy).
            per_grade_week_vars = {
                grade: collect_pair_grade_week_vars(X, data, grade, pair)
                for grade in sunday_budget
            }

            grade_play_by_week: dict = {}
            for grade, week_to_vars in per_grade_week_vars.items():
                grade_play_by_week[grade] = {}
                for w in all_weeks:
                    vars_for_week = week_to_vars.get(w, [])
                    # Use the helper-var pool. Channeling is `AddMaxEquality`
                    # for non-empty lists, `== 0` for empty — exactly the
                    # behaviour we want (empty week => can't play).
                    ind = registry.get_or_create_bool(
                        (STACK_PLAY_PREFIX, pair, grade, w),
                        vars_for_week,
                        f'cvc_stack_play_{pair[0]}_{pair[1]}_{grade}_{w}',
                    )
                    grade_play_by_week[grade][w] = ind

            # 3. Pin sum_w play[g, w] == sunday_budget[g] for each grade.
            for grade, budget in sunday_budget.items():
                play_vars = [grade_play_by_week[grade][w] for w in all_weeks]
                if not play_vars:
                    continue
                # If the budget exceeds the number of weeks-with-vars then
                # the model is infeasible — let CP-SAT report that rather
                # than silently relaxing. Validate up front for a clearer
                # error if obviously over-budget.
                non_empty_weeks = sum(
                    1 for w in all_weeks
                    if per_grade_week_vars[grade].get(w)
                )
                if budget > non_empty_weeks:
                    raise ValueError(
                        f'ClubVsClubStackedWeekends: pair={pair} grade={grade} '
                        f'budget={budget} exceeds available Sunday weeks '
                        f'({non_empty_weeks}). Check FORCED_GAMES and team data.'
                    )
                model.Add(sum(play_vars) == budget)
                n += 1

            # 4. Consec-pair implication chain in descending-count order.
            #    play[lower_count, w] <= play[higher_count, w] for every week.
            sorted_grades = sorted_grades_by_desc_count(sunday_budget)
            for i in range(len(sorted_grades) - 1):
                hi_grade, _hi_count = sorted_grades[i]
                lo_grade, _lo_count = sorted_grades[i + 1]
                for w in all_weeks:
                    hi_play = grade_play_by_week[hi_grade][w]
                    lo_play = grade_play_by_week[lo_grade][w]
                    model.Add(lo_play <= hi_play)
                    n += 1

        return n


__all__ = ['ClubVsClubStackedWeekends']
