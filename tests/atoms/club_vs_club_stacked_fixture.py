"""Fixture for spec-005 `ClubVsClubStackedAlignment` atom tests.

Builds a small two-club universe with configurable per-grade matchup counts,
Sunday + Friday slots, and enough weeks to exercise the stacking implication
chain. No mocks — real CP-SAT model + real data shapes.

The fixture follows the convenor's example from spec-005 DoD #7:
  (Maitland, Norths) with PHL=4, 2nd=3, 3rd=2, 4th=2, 5th=1, 6th=0.

`num_rounds[grade]` and `Grade.num_teams` are chosen so that `R // (T-1)` (for
even T) or `R // T` (for odd T) — the per-team game count — equals each
desired meeting count. We use the simplest model: T=2 per grade (so each
matchup happens R times) and `R == desired_meeting_count`. So we set
`num_rounds[grade] = desired_count` per grade and `num_rounds[max] = max of
those` (drives spacing math but irrelevant for stacking).

This guarantees `per_pair_grade_meeting_counts({Maitland,Norths})[grade]` ==
`desired_count` for each grade, matching the DoD example exactly.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from constraints.atoms.base import BROADMEADOW, MAITLAND
from models import Club, Grade, PlayingField, Team, Timeslot


# DoD #7 default: PHL=4, 2nd=3, 3rd=2, 4th=2, 5th=1, 6th=0.
DEFAULT_GRADE_COUNTS = {
    'PHL': 4, '2nd': 3, '3rd': 2, '4th': 2, '5th': 1,
}


def build_stacked_fixture(
    grade_counts: Optional[Dict[str, int]] = None,
    *,
    num_weeks: int = 6,
    slots_per_field: int = 4,
    extra_teams_in_grade: Optional[Dict[str, Dict[str, int]]] = None,
) -> Dict:
    """Build a two-club (Maitland, Norths) fixture for stacking tests.

    Args:
      grade_counts: per-grade desired meeting count for the (Maitland, Norths)
          pair. Default: DoD #7.
      num_weeks: number of Sunday weeks generated. Must be >= max(grade_counts).
      slots_per_field: how many day_slots per field per week (≥4 to exercise
          contiguity gating with a middle slot).
      extra_teams_in_grade: `{grade: {'Maitland': N, 'Norths': N}}` adds N
          additional teams per club in that grade. Used by DoD #7 scenario 3
          (multiple teams per club per grade) — each extra team creates extra
          matchups (so `meetings = matchups * R // (T-1)` grows).
    """
    grade_counts = dict(grade_counts or DEFAULT_GRADE_COUNTS)
    extra = extra_teams_in_grade or {}

    ef = PlayingField(location=BROADMEADOW, name='EF')
    wf = PlayingField(location=BROADMEADOW, name='WF')
    mp = PlayingField(location=MAITLAND, name='Maitland Main Field')
    fields = [ef, wf, mp]

    clubs = [
        Club(name='Maitland', home_field=MAITLAND),
        Club(name='Norths', home_field=BROADMEADOW),
    ]

    teams: List[Team] = []
    for grade in grade_counts:
        # Default: 1 team per club per grade. Optional extras per scenario.
        for club in clubs:
            n_extra = extra.get(grade, {}).get(club.name, 0)
            base = 1
            for idx in range(base + n_extra):
                suffix = f'-{idx + 1}' if (base + n_extra) > 1 else ''
                teams.append(Team(
                    name=f'{club.name}{suffix} {grade}',
                    club=club,
                    grade=grade,
                ))

    # Build Grade objects (num_teams from team count).
    grade_objs = []
    for g_name in grade_counts:
        members = [t.name for t in teams if t.grade == g_name]
        grade_objs.append(Grade(name=g_name, teams=members))

    # Games: every cross-team pair within a grade.
    games: List[Tuple[str, str, str]] = []
    for grade in grade_objs:
        for t1, t2 in combinations(grade.teams, 2):
            t1s, t2s = sorted((t1, t2))
            games.append((t1s, t2s, grade.name))

    # Compute per_team_games per grade so that R // (T-1) (even T) or R // T
    # (odd T) == grade_counts[grade]. With T=2 it's R. With T=4 (extras) we
    # need R/(T-1) == per-pair meetings, i.e. R = per_meetings * (T-1). The
    # test path that uses extras (DoD scenario 3) computes its own expected
    # value via `per_pair_grade_meeting_counts` — so we just set R so the
    # default 2-team case matches and let the multi-team case follow.
    num_rounds: Dict[str, int] = {}
    for g_name, count in grade_counts.items():
        # T is 2 by default; if extras, T is larger.
        n_teams = sum(1 for t in teams if t.grade == g_name)
        if n_teams == 2:
            num_rounds[g_name] = count
        else:
            # Even-T case: R // (T-1) == per_matchup_meetings.
            # We want a per-matchup meeting count of 1 typically; choose
            # R = 1 * (T-1) for even-T. For odd-T R = 1 * T.
            per_matchup = 1
            if n_teams % 2 == 0:
                num_rounds[g_name] = per_matchup * (n_teams - 1)
            else:
                num_rounds[g_name] = per_matchup * n_teams
    num_rounds['max'] = max(num_rounds.values())

    # Sunday timeslots: each week has Broadmeadow EF/WF + Maitland MP, with
    # `slots_per_field` slots each. Slot times don't matter for the atom.
    timeslots: List[Timeslot] = []
    sunday_dates = [(w, f'2026-{(2 + (w - 1) // 4):02d}-{((w - 1) % 4) * 7 + 22:02d}')
                    for w in range(1, num_weeks + 1)]
    for week, date_str in sunday_dates:
        for field in fields:
            for slot in range(1, slots_per_field + 1):
                timeslots.append(Timeslot(
                    date=date_str, day='Sunday', time=f'{10 + slot}:00',
                    week=week, day_slot=slot, field=field, round_no=week,
                ))

    # One Friday slot per week (PHL Maitland Park) — used by FORCED-Friday tests.
    for week, _ in sunday_dates:
        timeslots.append(Timeslot(
            date=f'2026-04-{week + 20:02d}', day='Friday', time='19:00',
            week=week, day_slot=1, field=mp, round_no=week,
        ))

    return {
        'games': games,
        'timeslots': timeslots,
        'teams': teams,
        'grades': grade_objs,
        'clubs': clubs,
        'fields': fields,
        'current_week': 0,
        'locked_weeks': set(),
        'num_rounds': num_rounds,
        'constraint_slack': {},
        'penalty_weights': {},
        'forced_games': [],
        'blocked_games': [],
        'team_conflicts': [],
        'phl_preferences': {},
        'club_days': {},
        'preference_no_play': {},
        'home_field_map': {'Maitland': MAITLAND},
        'constraint_defaults': {},
        'penalties': {},
    }


def build_model_X(data: Dict) -> Tuple[cp_model.CpModel, Dict]:
    """Build a CP-SAT model + X dict.

    PHL gets every (Sunday, Friday) candidate slot. Other grades get Sunday
    candidates only — matches the production filter (only PHL plays Friday).
    """
    model = cp_model.CpModel()
    X: Dict = {}
    for (t1, t2, grade) in data['games']:
        for t in data['timeslots']:
            if not t.day:
                continue
            if grade != 'PHL' and t.day == 'Friday':
                continue
            key = (
                t1, t2, grade, t.day, t.day_slot, t.time,
                t.week, t.date, t.round_no, t.field.name, t.field.location,
            )
            X[key] = model.NewBoolVar(
                f'X_{t1}_{t2}_{grade}_w{t.week}_s{t.day_slot}'
                f'_{t.field.name}_{t.day[:3]}'
            )
    return model, X


def solve_with_timeout(model: cp_model.CpModel, seconds: float = 10.0):
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 4
    status = solver.Solve(model)
    return status, solver
