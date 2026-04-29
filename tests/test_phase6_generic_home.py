"""Phase 6: generic non-default-home constraint behaviour.

Verifies that the renamed-from-Maitland constraint logic iterates over every
club in `home_field_map` and reads per-club tuning from `AWAY_VENUE_RULES`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import pytest

from analytics.storage import DrawStorage, StoredGame
from analytics.tester import DrawTester
from constraints.registry import CONSTRAINT_REGISTRY
from models import Club, Grade, Team


NIHC = 'Newcastle International Hockey Centre'
MAITLAND_PARK = 'Maitland Park'
CCHP = 'Central Coast Hockey Park'
WYONG = 'Wyong Hockey Park'  # imaginary 3rd non-default home for tests


def _make_game(gid, t1, t2, week, date, location, day='Sunday', slot=1, time='10:00'):
    return StoredGame(
        game_id=gid, team1=t1, team2=t2, grade='3rd', week=week, round_no=week,
        date=date, day=day, time=time, day_slot=slot, field_name='Main',
        field_location=location,
    )


def _make_data(home_field_map, away_venue_rules=None, slack=0,
               consecutive=1, max_away_clubs=3):
    clubs = [
        Club(name='Tigers', home_field=NIHC),
        Club(name='Wests', home_field=NIHC),
        Club(name='Norths', home_field=NIHC),
        Club(name='Maitland', home_field=MAITLAND_PARK),
        Club(name='Gosford', home_field=CCHP),
        Club(name='Wyong', home_field=WYONG),
    ]
    teams: List[Team] = [
        Team(name=f'{c.name} 3rd', club=c, grade='3rd') for c in clubs
    ]
    grades = [Grade(name='3rd', teams=[t.name for t in teams])]
    return {
        'clubs': clubs, 'teams': teams, 'grades': grades,
        'num_rounds': {'3rd': 6, 'max': 6},
        'timeslots': [],
        'home_field_map': home_field_map,
        'away_venue_rules': away_venue_rules or {},
        'constraint_defaults': {
            'maitland_max_consecutive_home': consecutive,
            'away_maitland_max_clubs': max_away_clubs,
        },
        'constraint_slack': {
            'MaitlandHomeGrouping': slack,
            'AwayAtMaitlandGrouping': slack,
        },
    }


class TestNonDefaultHomeGroupingIteratesClubs:
    def test_third_club_added_to_home_field_map_is_checked(self):
        """Adding 'Wyong' to home_field_map causes its games to be checked."""
        home_field_map = {'Wyong': WYONG}
        data = _make_data(home_field_map, consecutive=1)
        # Wyong plays 3 home games in a row at WYONG → violation.
        games = [
            _make_game('G1', 'Wyong 3rd', 'Tigers 3rd', 1, '2026-04-05', WYONG),
            _make_game('G2', 'Wyong 3rd', 'Wests 3rd', 2, '2026-04-12', WYONG),
            _make_game('G3', 'Wyong 3rd', 'Norths 3rd', 3, '2026-04-19', WYONG),
        ]
        draw = DrawStorage(description='t', num_weeks=3, num_games=3, games=games)
        tester = DrawTester(draw, data)
        violations = tester._check_maitland_back_to_back()
        assert len(violations) >= 1
        assert any('Wyong' in v.message for v in violations)

    def test_removing_club_from_home_field_map_silences_constraint(self):
        """A club without an entry in home_field_map gets no grouping check."""
        # Maitland has back-to-back home games but is NOT in home_field_map.
        data = _make_data({}, consecutive=1)
        games = [
            _make_game('G1', 'Maitland 3rd', 'Tigers 3rd', 1, '2026-04-05', MAITLAND_PARK),
            _make_game('G2', 'Maitland 3rd', 'Wests 3rd', 2, '2026-04-12', MAITLAND_PARK),
        ]
        draw = DrawStorage(description='t', num_weeks=2, num_games=2, games=games)
        tester = DrawTester(draw, data)
        violations = tester._check_maitland_back_to_back()
        assert violations == []


class TestPerClubAwayVenueRules:
    def test_per_club_consecutive_home_override(self):
        """AWAY_VENUE_RULES[club]['max_consecutive_home']=3 overrides default of 1."""
        home_field_map = {'Maitland': MAITLAND_PARK}
        rules = {'Maitland': {'max_consecutive_home': 3}}
        data = _make_data(home_field_map, away_venue_rules=rules, consecutive=1)
        # 3 consecutive home weeks would normally violate (default=1) but rule allows 3.
        games = [
            _make_game('G1', 'Maitland 3rd', 'Tigers 3rd', 1, '2026-04-05', MAITLAND_PARK),
            _make_game('G2', 'Maitland 3rd', 'Wests 3rd', 2, '2026-04-12', MAITLAND_PARK),
            _make_game('G3', 'Maitland 3rd', 'Norths 3rd', 3, '2026-04-19', MAITLAND_PARK),
        ]
        draw = DrawStorage(description='t', num_weeks=3, num_games=3, games=games)
        tester = DrawTester(draw, data)
        violations = tester._check_maitland_back_to_back()
        assert violations == []

    def test_none_max_away_clubs_skips_constraint(self):
        """A club with max_away_clubs=None (e.g. Gosford) skips the away constraint."""
        home_field_map = {'Gosford': CCHP, 'Maitland': MAITLAND_PARK}
        rules = {'Gosford': {'max_away_clubs': None}}
        data = _make_data(home_field_map, away_venue_rules=rules, max_away_clubs=2)
        # 3 distinct away clubs at CCHP in week 1 (over Maitland's limit of 2)
        # — but Gosford's None disables the check at CCHP.
        games = [
            _make_game('G1', 'Gosford 3rd', 'Tigers 3rd', 1, '2026-04-05', CCHP),
            _make_game('G2', 'Gosford 3rd', 'Wests 3rd', 1, '2026-04-05',
                       CCHP, slot=2, time='11:30'),
            _make_game('G3', 'Gosford 3rd', 'Norths 3rd', 1, '2026-04-05',
                       CCHP, slot=3, time='13:00'),
        ]
        draw = DrawStorage(description='t', num_weeks=1, num_games=3, games=games)
        tester = DrawTester(draw, data)
        ccvio = [v for v in tester._check_maitland_away_clubs_limit()
                 if 'Central Coast' in v.message]
        assert ccvio == [], f'expected 0 CCHP violations but got: {ccvio}'


class TestAliasRegistryEntries:
    def test_non_default_aliases_present(self):
        for name in ('NonDefaultHomeGrouping', 'AwayAtNonDefaultGrouping'):
            assert name in CONSTRAINT_REGISTRY, f'{name} missing'

    def test_aliases_share_severity_and_slack(self):
        assert CONSTRAINT_REGISTRY['NonDefaultHomeGrouping'].slack_key == 'MaitlandHomeGrouping'
        assert CONSTRAINT_REGISTRY['AwayAtNonDefaultGrouping'].slack_key == 'AwayAtMaitlandGrouping'
