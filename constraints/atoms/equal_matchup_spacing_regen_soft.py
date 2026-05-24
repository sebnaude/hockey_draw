"""EqualMatchUpSpacingRegenSoft atom (spec-027 regen-soft).

SOFT analogue of the EqualMatchUpSpacing hard constraint. Instead of a hard
clause that forbids a pair from meeting more than once inside a sliding window
of `space = T - 1` consecutive rounds, this atom emits an IntVar penalty that
captures the *violation amount* (number of meetings beyond 1 inside the window).
The model remains feasible for any variable assignment.

**Penalty semantics (1 unit = 1 too-close repeat meeting in one window):**

For each (team1, team2, grade) pair and each sliding window of `space = T - 1`
consecutive rounds, we create:

    pen >= sum(window_vars) - 1,   pen >= 0

where ``sum(window_vars)`` is the total meetings of that pair inside the window.
When both teams meet exactly once in the window, ``sum == 1`` and ``pen == 0``
(ideal). When they meet twice, ``pen >= 1`` (one too-close repeat). The
minimising objective drives pen to the smallest possible value, so it always
settles to ``max(0, sum - 1)``.

Because windows overlap, the same "too-close" pairing can be counted in
multiple windows. The test hand-computes the exact total across all windows
and asserts that value.

**Config:**

    penalty_weights['regen_equal_matchup_spacing']  (default: 60000)

Set to 0 to disable the atom entirely (returns 0 immediately).

**Skip conditions (standard across all atoms):**

- Dummy keys: ``len(key) < 11``
- No-day keys: ``not key[3]``
- Locked weeks: ``key[6] in locked_weeks``
- Grades with ``T < 2`` or ``space >= R`` (no pairwise spacing possible)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict

from constraints.atoms.base import Atom

# Default penalty weight. High enough to strongly discourage back-to-back
# meetings (same order of magnitude as ClubVsClubAlignment=100000) without
# completely dominating all other soft objectives.
REGEN_EQUAL_MATCHUP_SPACING_DEFAULT_WEIGHT = 60000


class EqualMatchUpSpacingRegenSoft(Atom):
    """SOFT sliding-window matchup-spacing penalty (spec-027).

    Mirrors ``_matchup_spacing_soft`` in ``unified.py`` as a standalone atom.
    Adds no hard ``model.Add`` that can make the model infeasible — only
    IntVar penalties that the objective minimises.
    """

    canonical_name = 'EqualMatchUpSpacingRegenSoft'
    atom_group = ''

    def apply(self, model, X: Dict, data: Dict, registry) -> int:
        """Add penalty vars for too-close repeat matchups.

        Returns the number of penalty IntVars created.
        """
        weight = data.get('penalty_weights', {}).get(
            'regen_equal_matchup_spacing', REGEN_EQUAL_MATCHUP_SPACING_DEFAULT_WEIGHT
        )
        if weight == 0:
            return 0

        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_equal_matchup_spacing', {'weight': weight, 'penalties': []}
        )

        # -----------------------------------------------------------------
        # Build groupings from X directly (no engine internals).
        # Group real vars by (t1, t2, grade) -> {round_no: [vars]}.
        # -----------------------------------------------------------------
        locked_weeks = set(data.get('locked_weeks', set()))
        # (t1, t2, grade) -> {round_no -> [vars]}
        by_pair_round: Dict = defaultdict(lambda: defaultdict(list))

        for key, var in X.items():
            if len(key) < 11:
                continue           # dummy key
            if not key[3]:
                continue           # no-day timeslot
            week = key[6]
            if locked_weeks and week in locked_weeks:
                continue           # locked week
            t1, t2, grade = key[0], key[1], key[2]
            round_no = key[8]
            by_pair_round[(t1, t2, grade)][round_no].append(var)

        # -----------------------------------------------------------------
        # Per-grade team count (T) and max round (R).
        # Mirrors unified.py::_matchup_spacing_soft which uses g.num_teams.
        # -----------------------------------------------------------------
        grade_num_teams: Dict[str, int] = {}
        for g in data.get('grades', []):
            # Prefer g.num_teams if available; fall back to len(g.teams).
            if hasattr(g, 'num_teams') and g.num_teams:
                grade_num_teams[g.name] = g.num_teams
            elif hasattr(g, 'teams') and g.teams:
                grade_num_teams[g.name] = len(g.teams)
            else:
                grade_num_teams[g.name] = 0

        R = data.get('num_rounds', {}).get('max', 0)
        if not R:
            return 0

        # -----------------------------------------------------------------
        # Emit penalty vars: one per (pair, window).
        # -----------------------------------------------------------------
        n = 0
        for (t1, t2, grade), round_map in by_pair_round.items():
            T = grade_num_teams.get(grade, 0)
            if T < 2:
                continue
            space = T - 1
            if space >= R:
                continue
            # Windows: r_start in [1 .. R - space + 1] (inclusive).
            for r_start in range(1, R - space + 2):
                r_end = r_start + space - 1
                window_vars = []
                for r in range(r_start, r_end + 1):
                    if r in round_map:
                        window_vars.extend(round_map[r])
                if len(window_vars) < 2:
                    continue
                pen = model.NewIntVar(
                    0, len(window_vars),
                    f'u_regensp_wpen_{t1}_{t2}_{grade}_w{r_start}',
                )
                model.Add(pen >= sum(window_vars) - 1)
                bucket['penalties'].append(pen)
                n += 1

        return n
