"""BalancedByeSpacingRegenSoft atom — SOFT analogue of BalancedByeSpacing.

This is a *regen-soft* atom: instead of a hard clause that forbids a
too-close bye pair, it emits a PENALTY BoolVar equal to the violation
amount. The model always remains feasible; the objective subtracts the
weighted penalty sum.

Bye spacing semantics (mirrors BalancedByeSpacing exactly):

  - For each grade ``g`` with ``T`` teams and ``R`` playable rounds:
        games_per_team = max_games_per_grade(g)
        byes_per_team  = R - games_per_team

  - For each team ``t`` build a per-round bye indicator BoolVar ``B_t_r``:
        B_t_r  +  sum(team_vars_in_round_r)  ==  1
    This *structural* equality defines ``B_t_r``; it is NOT a forbidding
    clause and is kept unchanged in the soft version.

  - Pair check: for every (r1, r2) with ``r2 - r1 <= S`` we want at most
    one bye. The HARD atom adds ``B_r1 + B_r2 <= 1``. This SOFT atom
    instead creates a penalty BoolVar ``v`` that is 1 exactly when BOTH
    are byes (the violation), using three linear inequalities:

        v >= B_r1 + B_r2 - 1   (v fires when both are byes)
        v <= B_r1               (v can only fire when r1 is a bye)
        v <= B_r2               (v can only fire when r2 is a bye)

    Each ``v == 1`` represents one too-close bye pair for that team.

  ``S = ideal_bye_gap(R, byes_per_team) - slack``  (clamped at 0).

Penalty weight:
  ``data['penalty_weights']['regen_balanced_bye_spacing']``
  Default: 50 000. Weight == 0 disables the atom entirely (returns 0).

Penalty bucket:
  ``data['penalties']['regen_balanced_bye_spacing']``
  Shape: ``{'weight': int, 'penalties': [BoolVar, ...]}``.

One unit of penalty = one too-close bye pair (a pair of rounds r1 < r2
with ``r2 - r1 <= S`` where BOTH rounds are byes for the same team).

No-op conditions (identical to the hard atom):
  - Grade has ``byes_per_team < 2`` (no pairwise check possible).
  - ``S <= 0`` after applying slack.
  - ``weight == 0``.
  - All rounds locked (every pair is locked-locked → skipped).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from constraints.atoms.base import Atom
from constraints.atoms._spacing import ideal_bye_gap


# Default penalty weight for regen_balanced_bye_spacing.
# High enough to strongly discourage violations without dominating structural
# constraints; the same order of magnitude as other regen-soft atoms.
REGEN_BALANCED_BYE_SPACING_DEFAULT_WEIGHT = 50_000


class BalancedByeSpacingRegenSoft(Atom):
    """SOFT bye-spacing: penalise (not forbid) too-close bye pairs.

    Mirrors ``BalancedByeSpacing`` in structure; replaces every hard
    ``B_r1 + B_r2 <= 1`` with a penalty BoolVar that fires when both
    rounds are byes. The model stays feasible for any X.
    """

    canonical_name = 'BalancedByeSpacingRegenSoft'
    atom_group = ''

    def apply(self, model, X, data, registry) -> int:  # noqa: D102
        # ----------------------------------------------------------------
        # Weight resolution — weight==0 means fully disabled.
        # ----------------------------------------------------------------
        weight = data.get('penalty_weights', {}).get(
            'regen_balanced_bye_spacing', REGEN_BALANCED_BYE_SPACING_DEFAULT_WEIGHT
        )
        if weight == 0:
            return 0

        teams = data.get('teams', []) or []
        grades = data.get('grades', []) or []
        if not teams or not grades:
            return 0

        # Slack uses the same key and math as the hard atom.
        slack_dict = data.get('constraint_slack', {}) or {}
        config_slack = int(slack_dict.get('BalancedByeSpacing', 0) or 0)
        defaults = data.get('constraint_defaults', {}) or {}
        base_slack = int(defaults.get('bye_spacing_base_slack', 0) or 0)

        locked_weeks = set(data.get('locked_weeks', set()))

        # ----------------------------------------------------------------
        # Collect per-team per-round game variables (non-dummy, real days).
        # ----------------------------------------------------------------
        per_team_round_vars: Dict = defaultdict(lambda: defaultdict(list))
        for key, var in X.items():
            if len(key) < 11:
                continue  # skip dummy keys
            if not key[3]:
                continue  # skip no-day slots
            t1, t2, grade, _day, _slot, _time, _week, _date, round_no, _fn, _fl = key
            per_team_round_vars[(t1, grade)][round_no].append(var)
            if t2 != t1:
                per_team_round_vars[(t2, grade)][round_no].append(var)

        # ----------------------------------------------------------------
        # All playable rounds.
        # ----------------------------------------------------------------
        all_rounds = self._all_playable_rounds(data)
        if not all_rounds:
            return 0
        R = len(all_rounds)
        rounds_sorted = sorted(all_rounds)

        # Per-grade games_per_team (same priority chain as hard atom).
        games_per_grade: Dict[str, int] = self._compute_games_per_team(grades, data)

        # ----------------------------------------------------------------
        # Penalty bucket (get-or-create).
        # ----------------------------------------------------------------
        bucket = data.setdefault('penalties', {}).setdefault(
            'regen_balanced_bye_spacing',
            {'weight': weight, 'penalties': []},
        )

        # ----------------------------------------------------------------
        # Main loop — per grade, per team, per close bye-pair.
        # ----------------------------------------------------------------
        n_penalty_terms = 0
        for g in grades:
            grade_name = g.name
            games = games_per_grade.get(grade_name, 0)
            if games <= 0:
                continue
            byes_per_team = R - games
            if byes_per_team < 2:
                # 0 or 1 bye per team — no pairwise check to penalise.
                continue
            S_base = ideal_bye_gap(R, byes_per_team)
            S = max(0, S_base - base_slack - config_slack)
            if S <= 0:
                # Slack has fully disabled bye-spacing for this grade.
                continue

            team_names = [t.name for t in teams if t.grade == grade_name]
            for team_name in team_names:
                round_vars = per_team_round_vars.get((team_name, grade_name), {})

                # Build bye-indicator BoolVars (structural equality, not a
                # forbidding clause — identical to the hard atom).
                bye_var: Dict[int, object] = {}
                for r in rounds_sorted:
                    vars_in_round = round_vars.get(r, [])
                    if not vars_in_round:
                        # No candidate vars → forced bye (constant 1).
                        bye_var[r] = model.NewConstant(1)
                    else:
                        bv = model.NewBoolVar(
                            f'u_bye_soft_{team_name}_{grade_name}_r{r}'
                        )
                        # bv = 1  iff  sum(vars_in_round) == 0.
                        # With NoDoubleBookingTeams, sum ∈ {0,1}, so this is
                        # equivalent to the complement.
                        model.Add(bv + sum(vars_in_round) == 1)
                        bye_var[r] = bv

                # Pairwise SOFT check: emit a penalty BoolVar for every
                # (r1, r2) with gap <= S.  Skip locked-locked pairs
                # (identical policy to the hard atom).
                for i, r1 in enumerate(rounds_sorted):
                    locked1 = r1 in locked_weeks
                    for r2 in rounds_sorted[i + 1:]:
                        gap = r2 - r1
                        if gap > S:
                            break
                        locked2 = r2 in locked_weeks
                        if locked1 and locked2:
                            # Both sides are constants in the locked draw;
                            # the hard atom skips these, so we do too.
                            continue

                        # v == 1  iff  B_r1 == 1  AND  B_r2 == 1.
                        # Three linear constraints pin v exactly (valid for
                        # 0/1 BoolVars; equivalent to v == B_r1 * B_r2):
                        #   v >= B_r1 + B_r2 - 1
                        #   v <= B_r1
                        #   v <= B_r2
                        v = model.NewBoolVar(
                            f'bye_soft_pen_{team_name}_{grade_name}_r{r1}_r{r2}'
                        )
                        model.Add(v >= bye_var[r1] + bye_var[r2] - 1)
                        model.Add(v <= bye_var[r1])
                        model.Add(v <= bye_var[r2])
                        bucket['penalties'].append(v)
                        n_penalty_terms += 1

        return n_penalty_terms

    # ------------------------------------------------------------------
    # Static helpers — identical to BalancedByeSpacing's helpers.
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_games_per_team(grades, data: Dict) -> Dict[str, int]:
        """Per-grade games-per-team (same three-tier priority as hard atom)."""
        override = data.get('GRADE_ROUNDS_OVERRIDE') or {}
        num_rounds = data.get('num_rounds') or {}
        out: Dict[str, int] = {}
        for g in grades:
            name = g.name
            if name in override:
                out[name] = int(override[name])
                continue
            if name in num_rounds:
                out[name] = int(num_rounds[name])
                continue
            max_r = num_rounds.get('max', 0)
            out[name] = int(max_r) if max_r else 0
        return out

    @staticmethod
    def _all_playable_rounds(data: Dict) -> List[int]:
        """Sorted list of playable round numbers (same logic as hard atom)."""
        num_rounds = data.get('num_rounds') or {}
        max_r = num_rounds.get('max')
        if isinstance(max_r, int) and max_r > 0:
            return list(range(1, max_r + 1))
        timeslots = data.get('timeslots') or []
        seen = set()
        for ts in timeslots:
            r = getattr(ts, 'round_no', None)
            day = getattr(ts, 'day', None)
            if r and day:
                seen.add(int(r))
        return sorted(seen)
