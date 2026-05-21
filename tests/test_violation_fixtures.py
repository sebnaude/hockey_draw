"""Phase 7a: walk tests/fixtures/violations/ and assert each fixture's listed
violations are flagged by DrawTester.

Each fixture has top-level metadata:
  - `_violations`: list of canonical names that should appear in the report
  - `_description`: human note about why this fixture exists

Fixtures may flag MORE than they list (latent violations) — that's fine. The
test only asserts the listed violations are present.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pytest

from analytics.storage import DrawStorage, StoredGame
from analytics.tester import DrawTester
from models import Club, Grade, Team


FIXTURES_DIR = Path(__file__).parent / 'fixtures' / 'violations'


def _load_fixture(path: Path) -> tuple[List[str], DrawStorage, str, Dict[str, Any]]:
    with open(path) as f:
        raw = json.load(f)
    expected = raw.pop('_violations', [])
    desc = raw.pop('_description', '')
    overrides = raw.pop('_teams_override', None)
    games = [StoredGame(**g) for g in raw['games']]
    draw = DrawStorage(
        description=raw.get('description', path.stem),
        num_weeks=raw.get('num_weeks', 1),
        num_games=len(games),
        games=games,
    )
    return expected, draw, desc, overrides


def _build_data_for_fixture(
    games: List[StoredGame],
    teams_override: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a minimal data dict from the team names that appear in games.

    `teams_override` lets a fixture declare team -> (club, grade) explicitly,
    bypassing the simple "club is everything before the last space" parser
    used by fixtures whose team names follow the canonical convention. Useful
    for fixtures that need multiple teams in the same (club, grade) bucket
    (e.g. spec-007's same-grade-same-club concurrency check).
    """
    team_names = set()
    for g in games:
        team_names.add(g.team1)
        team_names.add(g.team2)
    clubs_by_name: Dict[str, Club] = {}
    teams: List[Team] = []
    for team_name in sorted(team_names):
        if teams_override and team_name in teams_override:
            entry = teams_override[team_name]
            club_name = entry['club']
            grade = entry['grade']
        else:
            parts = team_name.rsplit(' ', 1)
            club_name, grade = parts[0], parts[1] if len(parts) == 2 else 'PHL'
        if club_name not in clubs_by_name:
            home = (
                'Maitland Park' if club_name == 'Maitland'
                else 'Central Coast Hockey Park' if club_name == 'Gosford'
                else 'Newcastle International Hockey Centre'
            )
            clubs_by_name[club_name] = Club(name=club_name, home_field=home)
        teams.append(Team(
            name=team_name, club=clubs_by_name[club_name], grade=grade,
        ))
    grades_by_name: Dict[str, List[str]] = {}
    for t in teams:
        grades_by_name.setdefault(t.grade, []).append(t.name)
    grades = [Grade(name=g, teams=names) for g, names in grades_by_name.items()]
    home_field_map = {
        c.name: c.home_field for c in clubs_by_name.values()
        if c.home_field != 'Newcastle International Hockey Centre'
    }
    return {
        'clubs': list(clubs_by_name.values()),
        'teams': teams,
        'grades': grades,
        'num_rounds': {g.name: 1 for g in grades} | {'max': 10},
        'timeslots': [],
        # spec-018: maitland_max_consecutive_home / away_maitland_max_clubs
        # removed (venue-sequencing rules deleted).
        'constraint_defaults': {},
        'home_field_map': home_field_map,
        'away_venue_rules': {},
    }


def _all_fixtures() -> List[Path]:
    if not FIXTURES_DIR.exists():
        return []
    return sorted(FIXTURES_DIR.glob('*.json'))


@pytest.mark.parametrize('fixture_path', _all_fixtures(), ids=lambda p: p.stem)
def test_fixture_flags_expected_violations(fixture_path: Path):
    """Each fixture's `_violations` list must be a subset of the actual violations flagged."""
    expected, draw, _desc, overrides = _load_fixture(fixture_path)
    assert expected, f'Fixture {fixture_path.name} has no _violations declared'

    data = _build_data_for_fixture(draw.games, teams_override=overrides)
    tester = DrawTester(draw, data)
    report = tester.run_violation_check()
    flagged = {v.constraint for v in report.violations}
    missing = set(expected) - flagged
    assert not missing, (
        f'Fixture {fixture_path.name} expected violations {expected} but '
        f'these were not flagged: {missing}. Flagged: {flagged}'
    )


def test_violation_fixtures_present():
    """Phase 7a delivers a starter set of violation fixtures.

    The set is intentionally narrow — fixtures only earn their place when they
    reliably exercise an atom violation path with a minimal hand-crafted draw.
    Adding more is good; the bar here is that we have the agreed coverage.
    """
    fixtures = _all_fixtures()
    # spec-018 removed the two Maitland-grouping fixtures (maitland_back_to_back +
    # away_at_maitland_overflow) when the venue-sequencing rules were deleted.
    assert len(fixtures) >= 6, (
        f'Phase 7a expects at least 6 violation fixtures; found {len(fixtures)}'
    )


def test_each_fixture_has_metadata():
    """Every fixture must declare `_violations` and `_description`."""
    for path in _all_fixtures():
        with open(path) as f:
            raw = json.load(f)
        assert '_violations' in raw, f'{path.name}: missing _violations'
        assert '_description' in raw, f'{path.name}: missing _description'
