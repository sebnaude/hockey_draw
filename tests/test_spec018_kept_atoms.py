"""spec-018 DoD 11 / Unit B — KEPT-atoms verification after deleting the
venue-sequencing rules.

The three deleted rules were:
  - NonDefaultHomeGrouping / MaxMaitlandHomeWeekends (consecutive home-weekend ban)
  - AwayAtNonDefaultGrouping / AwayAtMaitlandGrouping (away-clubs-per-weekend cap)
  - MaitlandAlternateHomeAway (HH/AA alternation soft penalty)

This test proves, with a real hand-crafted draw and the real DrawTester (no
mocks), that:

  * A draw with TWO consecutive Maitland home weekends AND multiple distinct
    away clubs visiting Maitland in the SAME weekend is now perfectly clean —
    the tester reports NO `MaxMaitlandHomeWeekends` and NO `AwayAtMaitlandGrouping`
    violations, and never produces a `maitland_alternate_home_away` penalty
    bucket (those checks/buckets were removed).

  * The KEPT spec-004 behaviour still applies: per-pair 50/50 home/away balance
    (`_check_fifty_fifty_home_away` -> `FiftyFiftyHomeAway`) is still enforced —
    a balanced draw PASSES and an imbalanced draw is FLAGGED.

Hand-computed oracle. Maitland fields a PHL team and a 3rd team; each plays
Norths and Wests (same grade) twice, arranged so every pair is exactly 1H/1A:
    Week 1 (Maitland Park): Maitland PHL v Norths PHL (HOME, slot 1)
                            Maitland 3rd v Wests 3rd (HOME, slot 2)
        -> week 1 hosts TWO distinct visiting clubs (Norths + Wests) at Maitland
           Park, which the deleted AwayAtMaitlandGrouping cap would have policed,
           with NO single team double-booked.
    Week 2 (Maitland Park): Maitland PHL v Wests PHL (HOME)
                            Maitland 3rd v Norths 3rd (HOME)
        -> weeks 1 and 2 are CONSECUTIVE Maitland home weekends, which the
           deleted MaxMaitlandHomeWeekends window would have banned.
    Week 3 (Broadmeadow):   Maitland PHL v Norths PHL (AWAY)
                            Maitland 3rd v Wests 3rd (AWAY)
    Week 4 (Broadmeadow):   Maitland PHL v Wests PHL (AWAY)
                            Maitland 3rd v Norths 3rd (AWAY)
  Per-pair balance: every Maitland team is 1H/1A vs each opponent —
  so `_check_fifty_fifty_home_away` PASSES on the balanced draw.
"""
from __future__ import annotations

import os
import sys
from typing import List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.storage import DrawStorage, StoredGame
from analytics.tester import DrawTester
from models import Club, Grade, Team


NIHC = 'Newcastle International Hockey Centre'
MAITLAND_PARK = 'Maitland Park'

# Constraint names that spec-018 removed from the tester. They must NEVER appear
# in a violation report nor as a penalty bucket again.
REMOVED_NAMES = {
    'MaxMaitlandHomeWeekends',
    'MaitlandHomeGrouping',
    'AwayAtMaitlandGrouping',
    'maitland_alternate_home_away',
    'MaitlandAlternateHomeAway',
}


def _g(gid, t1, t2, grade, week, date, location, day_slot=1, time='10:00'):
    return StoredGame(
        game_id=gid, team1=t1, team2=t2, grade=grade, week=week, round_no=week,
        date=date, day='Sunday', time=time, day_slot=day_slot,
        field_name='Main' if location == MAITLAND_PARK else 'EF',
        field_location=location,
    )


def _make_data():
    clubs = [
        Club(name='Maitland', home_field=MAITLAND_PARK),
        Club(name='Norths', home_field=NIHC),
        Club(name='Wests', home_field=NIHC),
    ]
    teams = []
    for c in clubs:
        for grade in ('PHL', '3rd'):
            teams.append(Team(name=f'{c.name} {grade}', club=c, grade=grade))
    grades = [
        Grade(name='PHL', teams=[t.name for t in teams if t.grade == 'PHL']),
        Grade(name='3rd', teams=[t.name for t in teams if t.grade == '3rd']),
    ]
    return {
        'clubs': clubs,
        'teams': teams,
        'grades': grades,
        'num_rounds': {'PHL': 4, '3rd': 4, 'max': 4},
        'timeslots': [],
        'constraint_defaults': {},
        'home_field_map': {'Maitland': MAITLAND_PARK},
        'away_venue_rules': {},
    }


def _balanced_games() -> List[StoredGame]:
    """Balanced 50/50 draw, with consecutive Maitland home weekends (weeks 1+2)
    AND two distinct visiting clubs at Maitland Park in the same weekend."""
    return [
        # Week 1 at Maitland Park: Norths (PHL) + Wests (3rd) both visit — two
        # distinct away clubs in one weekend, no single team double-booked.
        _g('G01', 'Maitland PHL', 'Norths PHL', 'PHL', 1, '2026-03-22', MAITLAND_PARK, day_slot=1),
        _g('G02', 'Maitland 3rd', 'Wests 3rd', '3rd', 1, '2026-03-22', MAITLAND_PARK, day_slot=2),
        # Week 2 at Maitland Park: Maitland home AGAIN (consecutive home weekend).
        _g('G03', 'Maitland PHL', 'Wests PHL', 'PHL', 2, '2026-03-29', MAITLAND_PARK, day_slot=1),
        _g('G04', 'Maitland 3rd', 'Norths 3rd', '3rd', 2, '2026-03-29', MAITLAND_PARK, day_slot=2),
        # Away legs at Broadmeadow — one per opponent, giving each pair 1H/1A.
        _g('G05', 'Maitland PHL', 'Norths PHL', 'PHL', 3, '2026-04-05', NIHC, day_slot=1),
        _g('G06', 'Maitland 3rd', 'Wests 3rd', '3rd', 3, '2026-04-05', NIHC, day_slot=2),
        _g('G07', 'Maitland PHL', 'Wests PHL', 'PHL', 4, '2026-04-12', NIHC, day_slot=1),
        _g('G08', 'Maitland 3rd', 'Norths 3rd', '3rd', 4, '2026-04-12', NIHC, day_slot=2),
    ]


def _imbalanced_games() -> List[StoredGame]:
    """Same matchups but Maitland PHL is HOME for both meetings vs Norths PHL
    (2H/0A), which must trip FiftyFiftyHomeAway."""
    games = _balanced_games()
    # Flip G05 (the Norths PHL away leg) to a Maitland home game.
    games[4] = _g('G05', 'Maitland PHL', 'Norths PHL', 'PHL', 3, '2026-04-05',
                  MAITLAND_PARK, day_slot=1)
    return games


def _report(games):
    data = _make_data()
    draw = DrawStorage(
        description='spec-018 kept-atoms', num_weeks=4,
        num_games=len(games), games=games,
    )
    return DrawTester(draw, data).run_violation_check()


class TestRemovedRulesNoLongerFire:
    def test_consecutive_home_and_many_away_clubs_is_clean(self):
        report = _report(_balanced_games())
        violated_names = {v.constraint for v in report.violations}
        result_names = {r.constraint for r in report.constraint_results}

        # None of the deleted rules may appear as a violation OR as a tester check.
        assert not (REMOVED_NAMES & violated_names), (
            f'Removed venue-sequencing rule(s) flagged a violation: '
            f'{REMOVED_NAMES & violated_names}'
        )
        assert not (REMOVED_NAMES & result_names), (
            f'Removed venue-sequencing rule(s) still run as tester checks: '
            f'{REMOVED_NAMES & result_names}'
        )

    def test_no_alternation_penalty_bucket(self):
        report = _report(_balanced_games())
        # The penalty bucket lived on metadata.penalties; it must never appear.
        penalties = getattr(report, 'penalties', {}) or {}
        assert 'maitland_alternate_home_away' not in penalties


class TestKeptFiftyFiftyStillApplies:
    def test_balanced_draw_passes_fifty_fifty(self):
        report = _report(_balanced_games())
        result_map = {r.constraint: r for r in report.constraint_results}
        # Canonical tester key is 'FiftyFiftyHomeandAway'; individual violations
        # are tagged 'FiftyFiftyHomeAway'.
        assert 'FiftyFiftyHomeandAway' in result_map
        assert result_map['FiftyFiftyHomeandAway'].status == 'PASSED'

    def test_imbalanced_draw_flags_fifty_fifty(self):
        report = _report(_imbalanced_games())
        violated_names = {v.constraint for v in report.violations}
        assert 'FiftyFiftyHomeAway' in violated_names, (
            f'Expected FiftyFiftyHomeAway violation on 2H/0A pair; '
            f'flagged: {violated_names}'
        )
