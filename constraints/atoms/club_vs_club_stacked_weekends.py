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

## PHL Sunday floors are away-venue-umbrella-aware (spec-044)

The per-team-pair Sunday floor (Layer 2) and the aggregate `min_budget`
(Layer 5) are computed by `team_pair_sunday_meetings_range` /
`pair_grade_sunday_aligned_weekend_range`. For PHL those LOWER bounds subtract
not only the exact per-pair forced Fridays but also the away club's
*umbrella*-forced PHL Fridays (`club_umbrella_forced_friday_meetings`, the
more-constrained club of the pair). Without this, Gosford's `gosford_friday_games=8`
and Maitland's `maitland_friday_games=2` venue-wide forced Fridays inflated those
clubs' aggregate Sunday floor past their Sunday capacity (R − forced Fridays),
making the real-2026 `core` model INFEASIBLE in presolve. The ceiling
(`tp_max` / `max_budget`) is left intact, so any single pair may still align the
full `base+1` on Sunday.

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
from constraints.atoms._club_day_shared import parse_club_day_entries
from constraints.atoms._club_vs_club_stacked_shared import (
    STACK_PLAY_PREFIX,
    STACK_TEAM_PAIR_PLAY_PREFIX,
    enumerate_club_pairs,
    enumerate_team_pairs_in_pair_grade,
    pair_grade_sunday_aligned_weekend_range,
    per_pair_grade_meeting_counts,
    team_pair_counts,
    team_pair_sunday_meetings_range,
)
from utils import get_nearest_week_by_date


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

        # Club-day exemption (spec-038): on a club's club-day weekend the entire
        # host roster is pinned to one Sunday (ClubDayParticipation), the
        # multi-team grades turn inward (derbies) and the single-team grades must
        # reach outside — so the cross-club same-opponent nesting (Layer 6) is
        # structurally unsatisfiable there AND redundant (ClubDaySameField +
        # ClubDayContiguousSlots already co-locate the whole roster that day).
        # We therefore skip the Layer-6 implication for any pair whose member is
        # a club-day host that week. Week-based and type-agnostic: covers both
        # opponent (Type A) and no-opponent/derby (Type B) club days — Type A
        # still drags the nesting via host's free second team. All other layers
        # and all non-host pairs keep their nesting unchanged.
        club_day_hosts_by_week: Dict[int, set] = defaultdict(set)
        for club_name, date_str, _opponent in parse_club_day_entries(data):
            cd_week = get_nearest_week_by_date(date_str, data.get('timeslots', []))
            club_day_hosts_by_week[cd_week].add(club_name)

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

            # Build per-pair-grade specs with BalancedMatchups-mirrored ranges
            # (spec-038 fix): the atom budget is a RANGE `[base, base+1]` per
            # team-pair (and `[max(a,b)*base, max(a,b)*(base+1)]` aggregate),
            # mirroring `EnsureEqualGamesAndBalanceMatchUps`. The earlier exact
            # `== per_matchup` made the atom INFEASIBLE on `season_test` for
            # grades with extras > 0 (5th: T=9 R=16 → all 36 pairs MUST meet 2
            # times to satisfy per-team total, but exact `== 1` forbade that).
            #
            # Tuple shape: (grade, min_budget, max_budget, tp_min, tp_max, a, b, team_pairs)
            pair_grade_specs: List[Tuple[str, int, int, int, int, int, int, List[Tuple[str, str]]]] = []
            for grade in grade_meetings_total.keys():
                min_budget, max_budget = pair_grade_sunday_aligned_weekend_range(
                    data, pair, grade
                )
                if max_budget == 0:
                    # No budget — skip entirely (e.g. one side has 0 teams,
                    # or PHL forced-Fridays consume the whole budget).
                    continue
                tp_min, tp_max = team_pair_sunday_meetings_range(data, pair, grade)
                if tp_max == 0:
                    continue
                a, b = team_pair_counts(data, pair, grade)
                team_pairs = enumerate_team_pairs_in_pair_grade(data, pair, grade)
                if not team_pairs:
                    # Defensive: max_budget > 0 implies a*b > 0 which
                    # implies team_pairs non-empty. Guard for consistency.
                    continue
                pair_grade_specs.append(
                    (grade, min_budget, max_budget, tp_min, tp_max, a, b, team_pairs)
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
            for grade, min_budget, max_budget, _tp_min, _tp_max, a, b, team_pairs in pair_grade_specs:
                # Early validation (DoD #7): the LOWER bound of the aligned-
                # weekend budget cannot exceed the number of Sunday weeks where
                # at least one team-pair has Sunday vars. (Range upper bound is
                # allowed to exceed since the constraint is `<=`; only the lower
                # bound `>=` would be unsatisfiable.)
                available_weeks_with_vars = sum(
                    1 for w in sunday_weeks
                    if any(tp_week_vars.get((tp[0], tp[1], w)) for tp in team_pairs)
                )
                if min_budget > available_weeks_with_vars:
                    raise ValueError(
                        f'ClubVsClubStackedWeekends: pair={pair} grade={grade} '
                        f'min_budget={min_budget} exceeds available Sunday weeks '
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

            # LAYER 2: per-team-pair Sunday budget — RANGE mirroring
            # `EnsureEqualGamesAndBalanceMatchUps` per-pair bound `[base, base+1]`,
            # restricted to Sundays via PHL forced-Friday subtraction (PHL only).
            #
            # For non-PHL: each tp plays `[base, base+1]` Sundays.
            # For PHL 1×1: each tp plays `[base - forced_fri, base+1 - forced_fri]`
            # Sundays (clamped ≥ 0).
            #
            # Exact range (vs the earlier exact `== per_matchup`) is needed for
            # grades with `extras > 0` (R - base*T_eff != 0): those grades require
            # some pairs to land at `base+1` to satisfy the per-team total.
            for grade, _min_b, _max_b, tp_min, tp_max, _a, _b, team_pairs in pair_grade_specs:
                for tp in team_pairs:
                    tp_play_vars = [team_pair_play[(tp, w)] for w in sunday_weeks]
                    model.Add(sum(tp_play_vars) >= tp_min)
                    model.Add(sum(tp_play_vars) <= tp_max)
                    n += 2

            # LAYER 3 + 4: per-pair-grade play_pg indicator + cardinality.
            # play_pg[(pair, grade, w)] = OR over team_pair_play[tp, w] for tp
            # in this (pair, grade). Registered under STACK_PLAY_PREFIX
            # (existing key shape, new semantics — see module docstring).
            play_pg_by_grade_week: Dict[Tuple[str, int], object] = {}
            for grade, _min_b, _max_b, _tp_min, _tp_max, a, b, team_pairs in pair_grade_specs:
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

            # LAYER 5: per-pair-grade total aligned-weekend budget — RANGE.
            # `[min_budget, max_budget]` = `[max(a,b)*base, max(a,b)*(base+1)]`
            # before PHL forced-Friday subtraction. Mirrors BalancedMatchups
            # aggregate range so extras-receiving pairs (e.g. 5th grade where
            # all pairs MUST meet `base+1` times) are satisfiable.
            for grade, min_budget, max_budget, _tp_min, _tp_max, _a, _b, _tps in pair_grade_specs:
                play_pg_vars = [play_pg_by_grade_week[(grade, w)] for w in sunday_weeks]
                model.Add(sum(play_pg_vars) >= min_budget)
                model.Add(sum(play_pg_vars) <= max_budget)
                n += 2

            # LAYER 6: cross-grade nested-superset chain. Sort grades by
            # max_budget descending (the upper bound of aligned weekends);
            # for each consecutive (hi, lo) pair and each week:
            # play_pg[lo, w] <= play_pg[hi, w]. Tie-break alpha for determinism.
            sorted_specs = sorted(
                pair_grade_specs, key=lambda spec: (-spec[2], spec[0])
            )
            for i in range(len(sorted_specs) - 1):
                hi_grade = sorted_specs[i][0]
                lo_grade = sorted_specs[i + 1][0]
                for w in sunday_weeks:
                    # Exempt the host club's club-day weekend (see above).
                    hosts = club_day_hosts_by_week.get(w)
                    if hosts and (pair[0] in hosts or pair[1] in hosts):
                        continue
                    hi_play = play_pg_by_grade_week[(hi_grade, w)]
                    lo_play = play_pg_by_grade_week[(lo_grade, w)]
                    model.Add(lo_play <= hi_play)
                    n += 1

        return n


__all__ = ['ClubVsClubStackedWeekends']
