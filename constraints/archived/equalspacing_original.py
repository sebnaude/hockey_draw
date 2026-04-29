# archived_equalspacing_original.py
"""
ARCHIVED: Original EqualMatchUpSpacingConstraint (human-written).

Replaced 2026-03-21 by the pairwise forbidden gaps + sliding window density
formulation. This file preserves the original algebraic approach for reference.

The original used per-matchup-pair:
  - R BoolVars (round indicators) + R IntVars (week values)
  - 3 AddMultiplicationEquality, 1 AddDivisionEquality
  - 1 AddMaxEquality, 1 AddAbsEquality
  - ~13 intermediate IntVars

On 2026 season data (185 matchup pairs):
  - 122,511 variables, 13,886 proto constraints
  - 555 multiplications, 185 divisions, 3,669 max equalities

The replacement uses:
  - 0 nonlinear operations
  - 115,059 variables, 14,843 proto constraints
  - Pairwise forbidden gaps (HARD) + sliding window density (SOFT)
"""
from ortools.sat.python import cp_model
from abc import ABC, abstractmethod
from collections import defaultdict


class Constraint(ABC):
    """Abstract base class for all scheduling constraints."""

    @abstractmethod
    def apply(self, model: cp_model.CpModel, X: dict, data: dict):
        """Apply constraint to the OR-Tools model."""
        pass


class EqualMatchUpSpacingConstraintOriginal(Constraint):
    """
    Spread out matchups evenly across rounds.

    For each pair of teams that meet multiple times, ensures spacing is reasonable.

    Ideal spacing = T - 1 (each team should see all other opponents before a rematch).
    Minimum gap = 2*(T-1)//3 (two-thirds of the ideal spacing).
    base_slack = (T-1) - 2*(T-1)//3, so that space - base_slack = 2*(T-1)//3.

    Examples:
    - 10 teams: space=9, base_slack=3, min gap=6
    - 4 teams: space=3, base_slack=1, min gap=2
    - 8 teams: space=7, base_slack=3, min gap=4

    --slack N adds to base_slack, further reducing the minimum required gap.

    Only minimum gap is enforced as a HARD constraint (no maximum gap).
    The SOFT penalty prefers ideal spacing but allows larger gaps.
    """

    def apply(self, model, X, data):
        games = data['games']
        timeslots = data['timeslots']
        R = data['num_rounds']['max']
        grades = {g.name: g.num_teams for g in data['grades']}

        # Get additional slack from config (--slack flag)
        config_slack = data.get('constraint_slack', {}).get('EqualMatchUpSpacingConstraint', 0)

        # Penalty weight for soft constraint
        PENALTY_WEIGHT = 5000

        # Initialize penalty tracking
        if 'penalties' not in data:
            data['penalties'] = {'EqualMatchUpSpacing': {'weight': PENALTY_WEIGHT, 'penalties': []}}
        elif 'EqualMatchUpSpacing' not in data['penalties']:
            data['penalties']['EqualMatchUpSpacing'] = {'weight': PENALTY_WEIGHT, 'penalties': []}

        # Derive base slack so that min gap = 2*(T-1)//3
        # space = T - 1 (ideal: see all other opponents before rematch)
        # base_slack = (T-1) - 2*(T-1)//3 (approx one-third of T-1)
        def get_base_slack(num_teams):
            ideal = num_teams - 1
            return max(1, ideal - 2 * ideal // 3)

        slack_per_grade = {
            name: min(get_base_slack(T) + config_slack, T // 2 + 1)
            for name, T in grades.items()
        }

        # Ideal spacing = T - 1 (see all other opponents before rematch)
        space_per_grade = {
            name: T - 1
            for name, T in grades.items()
        }

        # Gather game-vars by (t1, t2, grade, round_no)
        meetings = defaultdict(lambda: defaultdict(list))
        for t in timeslots:
            if not t.day:
                continue
            for (t1, t2, grade) in games:
                key = (t1, t2, grade, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
                if key in X:
                    meetings[(t1, t2, grade)][t.round_no].append(X[key])

        # For each matchup pair, build spacing constraints
        for (t1, t2, grade), round_map in meetings.items():
            base_slack = slack_per_grade[grade]
            space = space_per_grade[grade]

            # Build round indicators and week vars
            indic = []
            weekvs = []
            for r in range(1, R + 1):
                vars_at_r = round_map.get(r, [])
                # indicator: do they meet in round r?
                b = model.NewBoolVar(f"eqsp_ind_{t1}_{t2}_{grade}_r{r}")
                if vars_at_r:
                    model.AddMaxEquality(b, vars_at_r)
                else:
                    model.Add(b == 0)
                indic.append(b)

                # week var: =r if meeting, else 0
                wv = model.NewIntVar(0, R, f"eqsp_wv_{t1}_{t2}_{grade}_r{r}")
                model.Add(wv == r).OnlyEnforceIf(b)
                model.Add(wv == 0).OnlyEnforceIf(b.Not())
                weekvs.append(wv)

            # K = total number of meetings
            K = model.NewIntVar(0, R, f"eqsp_K_{t1}_{t2}_{grade}")
            model.Add(K == sum(indic))

            # Only enforce when K >= 2
            meets2 = model.NewBoolVar(f"eqsp_m2_{t1}_{t2}_{grade}")
            model.Add(K >= 2).OnlyEnforceIf(meets2)
            model.Add(K < 2).OnlyEnforceIf(meets2.Not())

            # round_sum = sum of weeks where they meet
            round_sum = model.NewIntVar(0, R * R, f"eqsp_rs_{t1}_{t2}_{grade}")
            model.Add(round_sum == sum(weekvs))

            # max_r = latest week they meet
            max_r = model.NewIntVar(0, R, f"eqsp_mr_{t1}_{t2}_{grade}")
            model.AddMaxEquality(max_r, weekvs)

            # Compute K*(K-1)/2 for spacing formula
            km1 = model.NewIntVar(-R, R, f"eqsp_km1_{t1}_{t2}_{grade}")
            prod = model.NewIntVar(-R * R, R * R, f"eqsp_prd_{t1}_{t2}_{grade}")
            half = model.NewIntVar(0, R * (R - 1) // 2, f"eqsp_half_{t1}_{t2}_{grade}")
            model.Add(km1 == K - 1)
            model.AddMultiplicationEquality(prod, [K, km1]).OnlyEnforceIf(meets2)
            model.AddDivisionEquality(half, prod, 2).OnlyEnforceIf(meets2)

            # ideal = K * max_r - space * half
            # This is the expected sum if meetings were evenly spaced backward from max_r
            kmax = model.NewIntVar(0, R * R, f"eqsp_kmax_{t1}_{t2}_{grade}")
            space_half = model.NewIntVar(-R * R, R * R, f"eqsp_sphalf_{t1}_{t2}_{grade}")
            ideal = model.NewIntVar(-R * R, R * R, f"eqsp_ideal_{t1}_{t2}_{grade}")
            model.AddMultiplicationEquality(kmax, [K, max_r]).OnlyEnforceIf(meets2)
            model.AddMultiplicationEquality(space_half, [space, half]).OnlyEnforceIf(meets2)
            model.Add(ideal == kmax - space_half).OnlyEnforceIf(meets2)

            # Compute allowed slack = base_slack * half (scales with number of meetings)
            max_slack = model.NewIntVar(0, R * (R - 1) // 2 * (base_slack + 5), f"eqsp_maxslk_{t1}_{t2}_{grade}")
            model.Add(max_slack == half * base_slack).OnlyEnforceIf(meets2)

            # diff = round_sum - ideal
            # Positive diff means games are bunched (gap < ideal) → enforce min gap
            # Negative diff means games are spread (gap > ideal) → allow freely
            diff = model.NewIntVar(-R * R, R * R, f"eqsp_diff_{t1}_{t2}_{grade}")
            model.Add(diff == round_sum - ideal).OnlyEnforceIf(meets2)

            # HARD constraint: diff <= max_slack (enforces minimum gap only, no maximum gap)
            model.Add(diff <= max_slack).OnlyEnforceIf(meets2)

            # SOFT penalty: minimize |diff| to prefer spacing close to ideal
            abs_diff = model.NewIntVar(0, R * R, f"eqsp_absdiff_{t1}_{t2}_{grade}")
            model.AddAbsEquality(abs_diff, diff)
            data['penalties']['EqualMatchUpSpacing']['penalties'].append(abs_diff)
