"""spec-038: per-team-pair granularity stacked-weekends structure.

Replaces the spec-005 game-count model with the spec-038 four-layer
aligned-weekend model.

## The four layers (DoD #1–#7)

For each `(club_pair, grade)` where `pair_grade_sunday_aligned_weekends > 0`
(implies both clubs field >=1 team in `grade` AND per_matchup >= 1):

1. **Per-team-pair play indicators.** For each (team_pair, week) over Sunday
   weeks: `team_pair_play[tp, w] = OR over Sunday X-vars with (team1, team2)
   == tp and week == w`. Registered under
   `(STACK_TEAM_PAIR_PLAY_PREFIX, tp, w)`. Fixed to 0 if no vars exist.
2. **Per-team-pair budget (hard).** `sum_w team_pair_play[tp, w] ==
   per_matchup` for each team_pair.
3. **Per-pair-grade aligned-weekend indicator.** `play_pg[pair, grade, w] =
   OR over team_pair_play[tp, w] for tp in (pair, grade)`. Registered under
   the EXISTING `STACK_PLAY_PREFIX` key shape `(pair, grade, w)`. Semantics
   shifted (OR over team-pair indicators rather than OR over raw X-vars) but
   the indicator value is the same for any given placement — co-location atom
   consumer is transparent.
4. **Per-aligned-weekend cardinality (hard).** On each (pair, grade, w):
   `sum_{tp} team_pair_play[tp, w] == min(a,b) * play_pg[pair, grade, w]`.
   Implemented as two `OnlyEnforceIf` branches: `sum == min_ab` when
   `play_pg`; `sum == 0` when `play_pg.Not()`.
5. **Total aligned-weekend budget (hard).** `sum_w play_pg[pair, grade, w]
   == weekends_budget` where `weekends_budget =
   pair_grade_sunday_aligned_weekends(data, pair, grade)`. PHL forced-Friday
   subtraction is handled inside that helper.
6. **Cross-grade nested-superset chain (hard).** For each (pair, w), with
   grades sorted by `weekends_budget` descending: `play_pg[pair, g_{k+1}, w]
   <= play_pg[pair, g_k, w]`. Same shape as today; runs over the new
   `play_pg`.
7. **Early validation.** If `weekends_budget > available_weeks_with_vars`
   for any (pair, grade), raises `ValueError` at apply-time with the pair,
   grade, budget, and available count.

## PHL backward compatibility

PHL is always 1×1 in the current league → `max(a,b) = min(a,b) = 1`, so:
  - per_matchup budget = per_matchup (1 team-pair sums to per_matchup).
  - cardinality = 1 * play_pg (the single team-pair plays iff play_pg=1).
  - total = per_matchup - phl_forced_friday_meetings (identical to spec-005).

The PHL preservation test in
`tests/atoms/test_club_vs_club_stacked_weekends.py` confirms no PHL behaviour
change.

## Helper-var registry

- `STACK_TEAM_PAIR_PLAY_PREFIX` (`cvc_stack_team_pair_play`): new in
  spec-038. Atom is its own producer and consumer.
- `STACK_PLAY_PREFIX` (`cvc_stack_play`): existing. Key shape unchanged;
  semantics shifted from OR-of-raw-vars to OR-of-team-pair-ORs. Functionally
  identical ("≥ 1 game this weekend"), so `ClubVsClubStackedCoLocation` (the
  consumer) reads the same indicator value without modification.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from constraints.atoms.base import Atom
from constraints.atoms._club_vs_club_stacked_shared import (
    STACK_PLAY_PREFIX,
    STACK_TEAM_PAIR_PLAY_PREFIX,
    _per_matchup_for_grade,
    enumerate_club_pairs,
    enumerate_team_pairs_in_pair_grade,
    pair_grade_sunday_aligned_weekends,
    per_pair_grade_meeting_counts,
    team_pair_counts,
)


class ClubVsClubStackedWeekends(Atom):
    """spec-038 four-layer aligned-weekend stacking model.

    Layer 1: per-team-pair Sunday play indicators (`team_pair_play[tp, w]`).
    Layer 2: per-team-pair budget (each tp plays `per_matchup` weekends).
    Layer 3: per-pair-grade aligned-weekend indicator (`play_pg[pair, grade, w]`)
             channelled as OR over the team-pair indicators in that pair-grade.
    Layer 4: per-aligned-weekend cardinality (when `play_pg=1`, exactly
             `min(a,b)` team-pair indicators are 1; when `play_pg=0`, zero
             are 1).
    Plus: per-pair-grade total = `weekends_budget`, cross-grade
    nested-superset implication chain.
    """

    canonical_name = 'ClubVsClubStackedWeekends'
    atom_group = 'ClubVsClubStackedAlignment'

    def apply(self, model, X, data, registry) -> int:
        n = 0
        # Sunday weeks with at least one var. We only build team_pair_play
        # indicators on Sunday weeks because the stacking model is Sunday-only
        # (PHL Fridays are subtracted from the budget upstream via
        # `pair_grade_sunday_aligned_weekends`).
        locked_weeks = set(data.get('locked_weeks', set()) or set())
        sunday_weeks = sorted({
            key[6]
            for key in X
            if len(key) >= 11 and key[3] == 'Sunday' and key[6] not in locked_weeks
        })
        if not sunday_weeks:
            return 0

        # Pre-index Sunday X-vars by (team_pair, week) so we don't repeatedly
        # scan X for each team-pair we care about.
        #   tp_week_vars[(t1, t2, w)] = list of Sunday vars for that triple.
        # Keys are alpha-sorted (t1 < t2) which matches the canonical X key
        # convention (team1 alphabetically before team2).
        tp_week_vars: Dict[Tuple[str, str, int], List] = defaultdict(list)
        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            if key[3] != 'Sunday':
                continue
            if key[6] in locked_weeks:
                continue
            tp_week_vars[(key[0], key[1], key[6])].append(var)

        for pair in enumerate_club_pairs(data):
            # Grades to consider for this pair: those with non-zero per-pair
            # game count (i.e. clubs share cross-club matchups in that grade).
            # We use the game-count helper purely for grade enumeration — it
            # tells us "which grades have any cross-club games?". Budget for
            # each grade comes from the aligned-weekend helper.
            grade_meetings_total = per_pair_grade_meeting_counts(data, pair)

            # Build (grade, weekends_budget, per_matchup, a, b, team_pairs)
            # tuples — but only for grades where the budget is non-zero.
            pair_grade_specs: List[Tuple[str, int, int, int, int, List[Tuple[str, str]]]] = []
            for grade in grade_meetings_total.keys():
                weekends_budget = pair_grade_sunday_aligned_weekends(data, pair, grade)
                if weekends_budget == 0:
                    # No budget — skip entirely (e.g. one side has 0 teams,
                    # or PHL forced-Fridays consume the whole budget).
                    continue
                a, b = team_pair_counts(data, pair, grade)
                per_matchup = _per_matchup_for_grade(data, grade)
                if per_matchup == 0:
                    # Defensive: weekends_budget > 0 implies per_matchup > 0,
                    # but guard anyway. Skip silently — no constraints to add.
                    continue
                team_pairs = enumerate_team_pairs_in_pair_grade(data, pair, grade)
                if not team_pairs:
                    # Defensive: weekends_budget > 0 implies a*b > 0 which
                    # implies team_pairs non-empty. Guard for consistency.
                    continue
                pair_grade_specs.append(
                    (grade, weekends_budget, per_matchup, a, b, team_pairs)
                )

            if not pair_grade_specs:
                continue

            # LAYER 1: per-team-pair Sunday play indicators.
            # team_pair_play[(tp, w)] = OR over Sunday vars with that (tp, w).
            # Registered under (STACK_TEAM_PAIR_PLAY_PREFIX, tp, w).
            # If no vars exist for (tp, w), `get_or_create_bool` fixes to 0.
            #
            # Note: the same team_pair can in principle appear in MULTIPLE
            # (club_pair, grade) entries only if its two teams' clubs differ
            # by pair — which is impossible (a team-pair has exactly one
            # cross-club pair). So no de-dup concern across pair_grade_specs.
            team_pair_play: Dict[Tuple[Tuple[str, str], int], object] = {}
            for grade, weekends_budget, per_matchup, a, b, team_pairs in pair_grade_specs:
                # Early validation (DoD #7): the per-pair-grade budget cannot
                # exceed the number of Sunday weeks where at least one team-pair
                # in this pair-grade has Sunday vars. (If ALL team-pairs have
                # vars on every Sunday this is simply len(sunday_weeks); if
                # some team-pair has zero Sunday vars on every week, the budget
                # check below catches that via the per-tp == per_matchup
                # constraint becoming infeasible.)
                available_weeks_with_vars = sum(
                    1 for w in sunday_weeks
                    if any(tp_week_vars.get((tp[0], tp[1], w)) for tp in team_pairs)
                )
                if weekends_budget > available_weeks_with_vars:
                    raise ValueError(
                        f'ClubVsClubStackedWeekends: pair={pair} grade={grade} '
                        f'budget={weekends_budget} exceeds available Sunday weeks '
                        f'with vars ({available_weeks_with_vars}). Check '
                        f'FORCED_GAMES and team data.'
                    )

                for tp in team_pairs:
                    for w in sunday_weeks:
                        cache_key = (tp, w)
                        if cache_key in team_pair_play:
                            continue
                        vars_for_tp_w = tp_week_vars.get((tp[0], tp[1], w), [])
                        ind = registry.get_or_create_bool(
                            (STACK_TEAM_PAIR_PLAY_PREFIX, tp, w),
                            vars_for_tp_w,
                            f'cvc_stack_team_pair_play_{tp[0]}_{tp[1]}_w{w}',
                        )
                        team_pair_play[cache_key] = ind

            # LAYER 2: per-team-pair budget — each tp plays exactly per_matchup weekends.
            # Sum over weeks of team_pair_play[tp, w] == per_matchup.
            # Same per_matchup for every tp within a single (pair, grade).
            for grade, _wbudget, per_matchup, _a, _b, team_pairs in pair_grade_specs:
                for tp in team_pairs:
                    tp_play_vars = [team_pair_play[(tp, w)] for w in sunday_weeks]
                    model.Add(sum(tp_play_vars) == per_matchup)
                    n += 1

            # LAYER 3 + 4: per-pair-grade play_pg indicator + cardinality.
            # play_pg[(pair, grade, w)] = OR over team_pair_play[tp, w] for tp
            # in this (pair, grade). Registered under STACK_PLAY_PREFIX
            # (existing key shape, new semantics — see module docstring).
            play_pg_by_grade_week: Dict[Tuple[str, int], object] = {}
            for grade, _wbudget, _pm, a, b, team_pairs in pair_grade_specs:
                min_ab = min(a, b)
                for w in sunday_weeks:
                    tp_vars_for_week = [team_pair_play[(tp, w)] for tp in team_pairs]
                    play_pg = registry.get_or_create_bool(
                        (STACK_PLAY_PREFIX, pair, grade, w),
                        tp_vars_for_week,
                        f'cvc_stack_play_{pair[0]}_{pair[1]}_{grade}_w{w}',
                    )
                    play_pg_by_grade_week[(grade, w)] = play_pg

                    # LAYER 4: cardinality. When play_pg=1, exactly min(a,b)
                    # team-pairs play; when play_pg=0, zero play. Both branches
                    # are needed for `OnlyEnforceIf` (one-way conditional).
                    #
                    # play_pg=0 → sum==0 is already implied by play_pg = OR
                    # (if all tp_play=0, sum=0 trivially); the OnlyEnforceIf
                    # branch is redundant but harmless. The forward branch
                    # (play_pg=1 → sum==min_ab) is the new structural content.
                    model.Add(
                        sum(tp_vars_for_week) == min_ab
                    ).OnlyEnforceIf(play_pg)
                    model.Add(
                        sum(tp_vars_for_week) == 0
                    ).OnlyEnforceIf(play_pg.Not())
                    n += 2

            # LAYER 5: per-pair-grade total aligned-weekend budget.
            # sum_w play_pg[pair, grade, w] == weekends_budget.
            for grade, weekends_budget, _pm, _a, _b, _tps in pair_grade_specs:
                play_pg_vars = [play_pg_by_grade_week[(grade, w)] for w in sunday_weeks]
                model.Add(sum(play_pg_vars) == weekends_budget)
                n += 1

            # LAYER 6: cross-grade nested-superset chain. Sort grades by
            # weekends_budget descending; for each consecutive (hi, lo) pair
            # and each week: play_pg[lo, w] <= play_pg[hi, w]. Tie-break alpha
            # for determinism.
            sorted_specs = sorted(
                pair_grade_specs, key=lambda spec: (-spec[1], spec[0])
            )
            for i in range(len(sorted_specs) - 1):
                hi_grade = sorted_specs[i][0]
                lo_grade = sorted_specs[i + 1][0]
                for w in sunday_weeks:
                    hi_play = play_pg_by_grade_week[(hi_grade, w)]
                    lo_play = play_pg_by_grade_week[(lo_grade, w)]
                    model.Add(lo_play <= hi_play)
                    n += 1

        return n


__all__ = ['ClubVsClubStackedWeekends']
