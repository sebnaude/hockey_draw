"""Phase 7a expansion: tests that the tester populates `Violation.affected_clubs`
and `Violation.metric_value`, and that these flow through `ViolationReport.breakdown`.

Loads fixtures from `tests/fixtures/violations/` and runs the same DrawTester
plumbing that production uses, then verifies the breakdown rollups.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analytics.storage import DrawStorage, StoredGame
from analytics.tester import DrawTester
from tests.test_violation_fixtures import _build_data_for_fixture


FIXTURES_DIR = Path(__file__).parent / 'fixtures' / 'violations'


def _load(name: str):
    with open(FIXTURES_DIR / name) as f:
        raw = json.load(f)
    raw.pop('_violations', None)
    raw.pop('_description', None)
    overrides = raw.pop('_teams_override', None)
    games = [StoredGame(**g) for g in raw['games']]
    draw = DrawStorage(
        description=raw.get('description', name),
        num_weeks=raw.get('num_weeks', 1),
        num_games=len(games),
        games=games,
    )
    return draw, overrides


def _run(name: str):
    draw, overrides = _load(name)
    data = _build_data_for_fixture(draw.games, teams_override=overrides)
    return DrawTester(draw, data).run_violation_check()


class TestAffectedClubsPopulated:
    def test_maitland_back_to_back_flags_maitland(self):
        report = _run('maitland_back_to_back.json')
        v = next(v for v in report.violations if v.constraint == 'MaxMaitlandHomeWeekends')
        assert 'Maitland' in v.affected_clubs
        assert v.metric_value is not None and v.metric_value >= 2

    def test_away_at_maitland_overflow_flags_clubs(self):
        report = _run('away_at_maitland_overflow.json')
        v = next(v for v in report.violations if v.constraint == 'AwayAtMaitlandGrouping')
        # 4 distinct away clubs visited Maitland Park
        assert v.metric_value == 4
        assert len(v.affected_clubs) >= 3

    def test_home_away_imbalance_flags_team_club(self):
        report = _run('home_away_imbalance.json')
        v = next(v for v in report.violations if v.constraint == 'FiftyFiftyHomeAway')
        assert v.affected_clubs  # at least one club populated
        assert v.metric_value is not None

    def test_club_grade_adjacency_flags_club(self):
        report = _run('club_grade_adjacency.json')
        v = next(v for v in report.violations if v.constraint == 'ClubGradeAdjacency')
        assert v.affected_clubs  # club name populated

    def test_club_vs_club_alignment_flags_pair(self):
        report = _run('club_vs_club_non_coincident.json')
        v = next(v for v in report.violations if v.constraint == 'ClubVsClubAlignment')
        assert set(v.affected_clubs) == {'Tigers', 'Norths'}
        assert v.metric_value is not None and v.metric_value >= 1

    def test_club_game_spread_flags_club_with_metric(self):
        report = _run('club_game_spread_overlap.json')
        v = next(v for v in report.violations if v.constraint == 'ClubGameSpread')
        assert v.affected_clubs  # club name populated
        assert v.metric_value is not None and v.metric_value >= 1


class TestBreakdownRollups:
    """The breakdown rolls violations up by club / type / soft-pressure using
    the fields populated above."""

    def test_breakdown_by_club_aggregates(self):
        report = _run('away_at_maitland_overflow.json')
        breakdown = report.breakdown
        # Multiple away clubs visited Maitland Park — they all show up under by_club.
        assert breakdown.by_club, 'breakdown.by_club empty'
        # Maitland is the host club. The visiting clubs are flagged.
        flagged_clubs = set(breakdown.by_club.keys())
        assert any(c in flagged_clubs for c in ['Tigers', 'Wests', 'Norths', 'Easts'])

    def test_soft_pressure_records_metric(self):
        report = _run('maitland_back_to_back.json')
        breakdown = report.breakdown
        # MaxMaitlandHomeWeekends entry should appear in soft_pressure with non-zero metric.
        assert 'MaxMaitlandHomeWeekends' in breakdown.soft_pressure
        bucket = breakdown.soft_pressure['MaxMaitlandHomeWeekends']
        assert bucket['over_limit'] >= 1
        assert bucket['worst_value'] is not None
        assert bucket['worst_club'] == 'Maitland'

    def test_breakdown_by_type_groups_per_constraint(self):
        report = _run('club_vs_club_non_coincident.json')
        breakdown = report.breakdown
        assert 'ClubVsClubAlignment' in breakdown.by_type
        assert len(breakdown.by_type['ClubVsClubAlignment']) >= 1
