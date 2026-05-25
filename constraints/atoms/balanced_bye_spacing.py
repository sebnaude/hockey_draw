"""BalancedByeSpacing atom (spec-008 Part B).

A team's *bye* rounds — rounds in which it does NOT play — carry the same
"spread evenly" intent as repeat matchups. With unequal grade sizes some
teams end up with consecutive byes early in the season then play every
round; that's the convenor's complaint this atom fixes.

HARD constraint, severity 2 (HIGH) — important but not a feasibility
blocker. Its own slack key `BalancedByeSpacing` lets the convenor loosen
bye spacing without touching matchup spacing.

Math (mirrors `EqualMatchUpSpacing` via the shared `_spacing` helpers):

  - For each grade ``g`` with ``T`` teams and ``R`` playable rounds:
        games_per_team = max_games_per_grade(g)
        byes_per_team  = R - games_per_team
    A team plays in some subset of rounds; the *bye-rounds* are the
    complement.

  - For each team ``t`` in grade ``g`` we build a per-round "bye"
    indicator BoolVar ``B_t_r``:
        B_t_r := 1 - sum(team_vars_in_round_r)
    Because the solver enforces "team plays at most one game per round"
    (NoDoubleBookingTeams), ``sum(team_vars_in_round_r) ∈ {0, 1}`` and
    ``B_t_r`` is itself effectively boolean.

  - Pairwise no-double-bye check: for every pair of rounds ``(r1, r2)``
    with ``r2 - r1 <= S``, enforce ``B_t_r1 + B_t_r2 <= 1`` (mirrors the
    matchup spacing structure with B in place of pair-var sums).

    ``S = ideal_bye_gap(R, byes_per_team) - slack``  (clamped at 0).
    See `_spacing.ideal_bye_gap` for the formula and rationale.

Locked-week handling: a bye in a locked week is fixed (the team isn't
playing that round in the locked draw and we can't change it). We still
*encode* the constraint for the locked-vs-unlocked boundary so a forced
bye in a locked round prevents an adjacent bye in an unlocked round —
but we skip the (locked, locked) pairs since both terms are constants.
The atom uses the X dict directly (not the engine's iterators) so it
sees ALL variables including locked ones, then filters.

Severity & slack:
- canonical_name = ``BalancedByeSpacing``
- severity_level = 2
- slack_key      = ``BalancedByeSpacing``

Returns 0 (no constraints added) when:
- A grade has byes_per_team <= 1 (no pairwise check possible).
- All byes are zero (every team plays every round → ideal_bye_gap=0).
- Slack zeroes S for every grade.

Normal-mode soft analogue (spec-033 Unit B):
- The hard floor above forbids bye pairs with gap <= S. Because S now
  sits ``bye_spacing_base_slack`` (=2 for 2026) rounds *below* the raw
  ideal ``S_base = ideal_bye_gap(R, byes)``, the hard rule tolerates bye
  pairs in the band ``(S, S_base]`` — closer than ideal but not forbidden.
- This atom adds a SOFT penalty pushing those tolerated-but-suboptimal
  pairs toward the ideal: for every (r1, r2) with ``S < gap <= S_base``
  it emits a penalty BoolVar ``p`` that fires iff BOTH rounds are byes,
  using the same three-inequality channelling the regen-soft sibling uses
  (``p >= B_r1 + B_r2 - 1``, ``p <= B_r1``, ``p <= B_r2``). Each firing =
  one closer-than-ideal bye pair for that team.
- It REUSES the very same per-round ``bye_var`` indicators the hard clause
  builds (no second variable family) and appends to the bucket keyed
  exactly ``'BalancedByeSpacing'`` (NOT the regen atom's
  ``'regen_balanced_bye_spacing'``). The weight is read via the direct-dict
  pattern ``data.get('penalty_weights', {}).get('BalancedByeSpacing', …)``.
- The soft term adds ONLY sub-threshold pressure on gaps the hard floor
  already permits — it NEVER relaxes the hard floor ``S``. When
  ``S >= S_base`` (impossible while base_slack/config_slack are >= 0, since
  S = max(0, S_base - base - cfg) <= S_base) the soft band is empty and no
  penalties are emitted.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

from constraints.atoms.base import Atom
from constraints.atoms._spacing import ideal_bye_gap


# Default normal-mode soft penalty weight for BalancedByeSpacing.
# Parity with the EqualMatchUpSpacing / ClubGameSpread soft weights
# (100_000) — set in config PENALTY_WEIGHTS; this is the fallback when the
# config key is absent.
BALANCED_BYE_SPACING_DEFAULT_WEIGHT = 100_000


class BalancedByeSpacing(Atom):
    canonical_name = 'BalancedByeSpacing'
    atom_group = ''  # standalone atom (no legacy combined class)

    def apply(self, model, X, data, registry) -> int:
        teams = data.get('teams', []) or []
        grades = data.get('grades', []) or []
        if not teams or not grades:
            return 0

        # Slack: this atom owns its own slack key separate from matchup spacing.
        slack_dict = data.get('constraint_slack', {}) or {}
        config_slack = int(slack_dict.get('BalancedByeSpacing', 0) or 0)
        defaults = data.get('constraint_defaults', {}) or {}
        base_slack = int(defaults.get('bye_spacing_base_slack', 0) or 0)

        locked_weeks = set(data.get('locked_weeks', set()))

        # Normal-mode soft analogue (spec-033 Unit B). Weight read via the
        # direct-dict pattern atoms use (NOT the engine-only _get_penalty_weight).
        # weight==0 disables the soft term entirely (the hard clause is unaffected).
        soft_weight = data.get('penalty_weights', {}).get(
            'BalancedByeSpacing', BALANCED_BYE_SPACING_DEFAULT_WEIGHT
        )
        soft_bucket = None
        if soft_weight:
            soft_bucket = data.setdefault('penalties', {}).setdefault(
                'BalancedByeSpacing',
                {'weight': soft_weight, 'penalties': []},
            )

        # Per-team list of (round_no, var) for non-dummy real games.
        per_team_round_vars: Dict[Tuple[str, str], Dict[int, List]] = defaultdict(
            lambda: defaultdict(list)
        )
        for key, var in X.items():
            if len(key) < 11:
                continue  # skip dummy
            if not key[3]:
                continue  # skip no-day
            t1, t2, grade, _day, _slot, _time, _week, _date, round_no, _fn, _fl = key
            per_team_round_vars[(t1, grade)][round_no].append(var)
            if t2 != t1:
                per_team_round_vars[(t2, grade)][round_no].append(var)

        # All playable rounds in the season (from data['num_rounds'] or by
        # walking timeslots if not present).
        all_rounds = self._all_playable_rounds(data)
        if not all_rounds:
            return 0
        R = len(all_rounds)
        rounds_sorted = sorted(all_rounds)

        # Cache per-grade games_per_team. Reads `num_rounds[grade_name]`
        # (or per-grade override / max as fallback) for parity with the
        # rest of the unified engine (`unified.py::_matchup_spacing_hard`
        # also pulls from `num_rounds`).
        games_per_grade: Dict[str, int] = self._compute_games_per_team(
            grades, data
        )

        # Per-team bye-indicator BoolVars, keyed by (team_name, grade, round).
        # Build on-demand inside the constraint loop so we only create vars
        # for grades that actually need bye spacing.
        n_constraints = 0
        for g in grades:
            grade_name = g.name
            games = games_per_grade.get(grade_name, 0)
            if games <= 0:
                continue
            byes_per_team = R - games
            if byes_per_team < 2:
                # 0 or 1 bye per team — no pairwise spacing to enforce.
                continue
            S_base = ideal_bye_gap(R, byes_per_team)
            S = max(0, S_base - base_slack - config_slack)
            # The soft band is the gap range the hard floor TOLERATES but
            # the ideal does not: (S, S_base]. Penalising it pushes byes
            # toward the ideal spread without relaxing the hard floor.
            # When S >= S_base the band is empty (no soft penalties).
            soft_active = soft_bucket is not None and S_base > S
            # Pairwise iteration must reach the largest enforced gap, which
            # is max(S, S_base when the soft band is active).
            max_gap = S_base if soft_active else S
            if S <= 0 and not soft_active:
                continue

            # For each team in this grade build per-round bye indicators
            # and emit pairwise constraints. Skip pairs where both rounds
            # are locked.
            team_names = [t.name for t in teams if t.grade == grade_name]
            for team_name in team_names:
                round_vars = per_team_round_vars.get((team_name, grade_name), {})
                # Build bye var per round. For rounds with no candidate game
                # vars (team can't play that round at all), the team has a
                # forced bye — we model this as the constant 1.
                bye_var: Dict[int, object] = {}
                for r in rounds_sorted:
                    vars_in_round = round_vars.get(r, [])
                    if not vars_in_round:
                        # No vars => forced bye (constant 1). Use the model's
                        # built-in constant via NewConstant for clean Add().
                        bye_var[r] = model.NewConstant(1)
                    else:
                        bv = model.NewBoolVar(
                            f'u_bye_{team_name}_{grade_name}_r{r}'
                        )
                        # bye = 1 - sum(plays). With NoDoubleBookingTeams in
                        # force, sum(plays) is 0 or 1, so bye is exactly the
                        # complement. We encode this as a linear equality.
                        model.Add(bv + sum(vars_in_round) == 1)
                        bye_var[r] = bv

                # Pairwise check on the bye-rounds. The hard clause covers
                # gap <= S; the soft band covers S < gap <= S_base.
                for i, r1 in enumerate(rounds_sorted):
                    locked1 = r1 in locked_weeks
                    for r2 in rounds_sorted[i + 1:]:
                        gap = r2 - r1
                        if gap > max_gap:
                            break
                        locked2 = r2 in locked_weeks
                        if locked1 and locked2:
                            # Both fixed by the locked draw — nothing to enforce.
                            continue
                        if gap <= S:
                            # HARD: forbid both being byes within the floor.
                            model.Add(bye_var[r1] + bye_var[r2] <= 1)
                            n_constraints += 1
                        elif soft_active:
                            # SOFT band (S < gap <= S_base): penalise (not
                            # forbid) byes that are closer than ideal. Same
                            # channelling the regen-soft sibling uses, but the
                            # normal-mode bucket key and the SAME bye_var
                            # indicators built above (no second var family).
                            #   p == 1  iff  bye_var[r1] == 1 AND bye_var[r2] == 1
                            p = model.NewBoolVar(
                                f'u_bye_soft_pen_{team_name}_{grade_name}_r{r1}_r{r2}'
                            )
                            model.Add(p >= bye_var[r1] + bye_var[r2] - 1)
                            model.Add(p <= bye_var[r1])
                            model.Add(p <= bye_var[r2])
                            soft_bucket['penalties'].append(p)
        return n_constraints

    @staticmethod
    def _compute_games_per_team(grades, data: Dict) -> Dict[str, int]:
        """Per-grade games-per-team lookup with safe fallbacks.

        Priority:
          1. ``data['GRADE_ROUNDS_OVERRIDE'][grade.name]`` (exact match — same
             as ``utils.max_games_per_grade``).
          2. ``data['num_rounds'][grade.name]``.
          3. ``data['num_rounds']['max']``.
          4. 0 (atom will skip the grade as a no-op).
        """
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
        """Return the sorted list of playable round numbers for the season.

        Prefers ``data['num_rounds']['max']`` (i.e. R = 1..max). Falls back
        to walking ``data['timeslots']`` and collecting every distinct
        ``round_no`` whose timeslot has a day (skips no-play weeks).
        """
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
