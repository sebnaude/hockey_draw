"""Tests for metadata-aware DrawTester behavior.

Uses REAL DrawStorage objects -- no mocks, no patches.
"""

import pytest
import sys
import os
import json
import pickle
import tempfile
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.tester import DrawTester, Violation, ViolationReport, ConstraintResult
from analytics.storage import DrawStorage, StoredGame
from models import Team, Club, Grade


# ============== Helpers ==============

NIHC = 'Newcastle International Hockey Centre'
MAITLAND_PARK = 'Maitland Park'


def make_game(game_id, team1, team2, grade, week, round_no, date, day='Sunday',
              time='10:00', day_slot=1, field_name='EF', field_location=NIHC):
    return StoredGame(
        game_id=game_id, team1=team1, team2=team2, grade=grade,
        week=week, round_no=round_no, date=date, day=day, time=time,
        day_slot=day_slot, field_name=field_name, field_location=field_location,
    )


def make_draw(games, description='Test', metadata=None):
    weeks = set(g.week for g in games)
    d = DrawStorage(
        description=description,
        num_weeks=len(weeks),
        num_games=len(games),
        games=games,
    )
    if metadata:
        d.metadata = metadata
    return d


def make_clubs():
    return [
        Club(name='Tigers', home_field=NIHC),
        Club(name='Wests', home_field=NIHC),
        Club(name='Maitland', home_field=MAITLAND_PARK),
    ]


def make_teams(clubs, grades_list=None):
    if grades_list is None:
        grades_list = ['PHL', '3rd']
    teams = []
    for club in clubs:
        for grade in grades_list:
            teams.append(Team(name=f'{club.name} {grade}', club=club, grade=grade))
    return teams


def make_grades(teams):
    by_grade = defaultdict(list)
    for t in teams:
        by_grade[t.grade].append(t.name)
    return [Grade(name=g, teams=tnames) for g, tnames in by_grade.items()]


def make_data(clubs=None, teams=None, grades=None, num_rounds=None, **extras):
    if clubs is None:
        clubs = make_clubs()
    if teams is None:
        teams = make_teams(clubs)
    if grades is None:
        grades = make_grades(teams)
    if num_rounds is None:
        num_rounds = {g.name: 4 for g in grades}
    data = {
        'clubs': clubs,
        'teams': teams,
        'grades': grades,
        'num_rounds': num_rounds,
        'timeslots': [],
        'constraint_defaults': {},
    }
    data.update(extras)
    return data


def _minimal_games():
    """A few games that won't cause double-booking violations."""
    return [
        make_game('G001', 'Tigers PHL', 'Wests PHL', 'PHL', 1, 1, '2026-03-22'),
        make_game('G002', 'Tigers PHL', 'Maitland PHL', 'PHL', 2, 2, '2026-03-29'),
        make_game('G003', 'Wests PHL', 'Maitland PHL', 'PHL', 3, 3, '2026-04-05'),
        make_game('G004', 'Tigers 3rd', 'Wests 3rd', '3rd', 1, 1, '2026-03-22', day_slot=2),
        make_game('G005', 'Tigers 3rd', 'Maitland 3rd', '3rd', 2, 2, '2026-03-29', day_slot=2),
        make_game('G006', 'Wests 3rd', 'Maitland 3rd', '3rd', 3, 3, '2026-04-05', day_slot=2),
    ]


# ============== Tests ==============


class TestNoMetadataRunsAllChecks:
    """DrawTester with no metadata runs all checks."""

    def test_no_metadata_runs_all_checks(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data)
        report = tester.run_violation_check()

        # Should have constraint_results for all 19 canonical constraints
        assert len(report.constraint_results) == 19
        assert report.metadata_source == 'none'
        # Every result should be PASSED or VIOLATED (none SKIPPED)
        statuses = {r.status for r in report.constraint_results}
        assert 'SKIPPED' not in statuses


class TestConstraintsAppliedFilters:
    """Only listed constraints are checked when constraints_applied is set."""

    def test_constraints_applied_filters_checks(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data,
                            constraints_applied=['NoDoubleBookingTeamsConstraint'])
        report = tester.run_violation_check()

        passed_or_violated = [r for r in report.constraint_results
                              if r.status in ('PASSED', 'VIOLATED')]
        skipped = [r for r in report.constraint_results if r.status == 'SKIPPED']

        # NoDoubleBookingTeams should run, plus ClubFieldConcentration (tester-only)
        run_names = {r.constraint for r in passed_or_violated}
        assert 'NoDoubleBookingTeams' in run_names
        assert 'ClubFieldConcentration' in run_names  # tester-only always runs
        assert len(skipped) >= 15  # most are skipped
        assert report.metadata_source == 'draw_json'


class TestExcludedConstraints:
    """Excluded constraints are skipped."""

    def test_excluded_constraints_skipped(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data,
                            excluded_constraints=['NoDoubleBookingTeamsConstraint'])
        report = tester.run_violation_check()

        result_map = {r.constraint: r for r in report.constraint_results}
        assert result_map['NoDoubleBookingTeams'].status == 'SKIPPED'
        assert result_map['NoDoubleBookingTeams'].skip_reason == 'excluded'
        # Others should still run
        assert result_map['NoDoubleBookingFields'].status in ('PASSED', 'VIOLATED')


class TestSlackFromMetadata:
    """Slack values from draw metadata are used."""

    def test_slack_from_metadata(self):
        data = make_data()
        meta = {
            'solver_config': {
                'constraint_slack': {
                    'MaitlandHomeGrouping': 5,
                }
            }
        }
        draw = make_draw(_minimal_games(), metadata=meta)
        tester = DrawTester(draw, data)
        assert tester.constraint_slack.get('MaitlandHomeGrouping') == 5

    def test_caller_overrides_metadata_slack(self):
        """Data dict slack overrides metadata slack."""
        data = make_data(constraint_slack={'MaitlandHomeGrouping': 10})
        meta = {
            'solver_config': {
                'constraint_slack': {
                    'MaitlandHomeGrouping': 5,
                }
            }
        }
        draw = make_draw(_minimal_games(), metadata=meta)
        tester = DrawTester(draw, data)
        assert tester.constraint_slack['MaitlandHomeGrouping'] == 10


class TestConstraintResultStatuses:
    """PASSED/VIOLATED/SKIPPED correctly assigned."""

    def test_constraint_result_statuses(self):
        data = make_data()
        # Create a double-booking to trigger NoDoubleBookingTeams violation
        games = _minimal_games()
        # Add a duplicate game for Tigers PHL in week 1
        games.append(make_game('G999', 'Tigers PHL', 'Maitland PHL', 'PHL', 1, 1,
                               '2026-03-22', day_slot=3))
        draw = make_draw(games)
        tester = DrawTester(draw, data)
        report = tester.run_violation_check()

        result_map = {r.constraint: r for r in report.constraint_results}
        assert result_map['NoDoubleBookingTeams'].status == 'VIOLATED'
        assert len(result_map['NoDoubleBookingTeams'].violations) > 0

        # NoDoubleBookingFields should pass (different day_slots)
        assert result_map['NoDoubleBookingFields'].status == 'PASSED'


class TestViolationReportHasConstraintResults:
    """New field populated."""

    def test_violation_report_has_constraint_results(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data)
        report = tester.run_violation_check()

        assert hasattr(report, 'constraint_results')
        assert isinstance(report.constraint_results, list)
        assert all(isinstance(r, ConstraintResult) for r in report.constraint_results)
        assert hasattr(report, 'metadata_source')


class TestConstraintsSummaryString:
    """Summary text correct."""

    def test_constraints_summary_string(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data)
        report = tester.run_violation_check()

        summary = report.constraints_summary()
        assert 'passed' in summary
        assert 'violated' in summary
        assert 'skipped' in summary

    def test_constraints_summary_with_skipped(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data,
                            constraints_applied=['NoDoubleBookingTeamsConstraint'])
        report = tester.run_violation_check()
        summary = report.constraints_summary()
        # Should show skipped > 0
        assert 'skipped' in summary


class TestFromCheckpointLoadsMetadata:
    """Create temp checkpoint dir with pkl + json."""

    def test_from_checkpoint_loads_metadata(self):
        data = make_data()

        # Build a minimal X solution
        games = _minimal_games()
        # X_solution needs real variable keys -- use 11-tuple format
        X_solution = {}
        for i, g in enumerate(games):
            # Use the 11-tuple format that from_X_solution expects
            key = (g.team1, g.team2, g.grade, g.day, g.day_slot, g.time,
                   g.week, g.date, g.round_no, g.field_name, g.field_location)
            X_solution[key] = 1

        metadata = {
            'constraints_applied': [
                {'name': 'NoDoubleBookingTeamsConstraint'},
                {'name': 'EnsureEqualGamesAndBalanceMatchUps'},
            ],
            'excluded_constraints': ['ClubGameSpread'],
            'constraint_slack': {'MaitlandHomeGrouping': 2},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = os.path.join(tmpdir, 'solution.pkl')
            meta_path = os.path.join(tmpdir, 'metadata.json')

            with open(pkl_path, 'wb') as f:
                pickle.dump(X_solution, f)
            with open(meta_path, 'w') as f:
                json.dump(metadata, f)

            tester = DrawTester.from_checkpoint(tmpdir, data, description='Test CP')

        # Check constraint filtering was applied
        assert tester._constraints_applied is not None
        assert 'NoDoubleBookingTeams' in tester._constraints_applied
        assert 'EqualGamesAndBalanceMatchUps' in tester._constraints_applied
        assert 'ClubGameSpread' in tester._excluded_constraints
        assert tester.constraint_slack.get('MaitlandHomeGrouping') == 2

    def test_from_checkpoint_missing_pkl_raises(self):
        data = make_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                DrawTester.from_checkpoint(tmpdir, data)

    def test_from_checkpoint_no_metadata_file(self):
        """Checkpoint with solution but no metadata.json still works."""
        data = make_data()
        games = _minimal_games()
        X_solution = {}
        for g in games:
            key = (g.team1, g.team2, g.grade, g.day, g.day_slot, g.time,
                   g.week, g.date, g.round_no, g.field_name, g.field_location)
            X_solution[key] = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'solution.pkl'), 'wb') as f:
                pickle.dump(X_solution, f)

            tester = DrawTester.from_checkpoint(tmpdir, data)
            assert tester._constraints_applied is None  # legacy mode
            report = tester.run_violation_check()
            assert len(report.constraint_results) == 19


class TestFromFileAutodetectsJson:
    """Load draw JSON with metadata."""

    def test_from_file_use_metadata(self):
        data = make_data()
        games = _minimal_games()
        meta = {
            'constraints_applied': [
                {'name': 'NoDoubleBookingTeamsConstraint'},
            ],
            'solver_config': {
                'excluded_constraints': ['ClubGameSpread'],
                'constraint_slack': {},
            },
        }
        draw = make_draw(games, metadata=meta)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_draw.json')
            draw.save(path)

            tester = DrawTester.from_file(path, data, use_metadata=True)
            assert tester._constraints_applied is not None
            assert 'NoDoubleBookingTeams' in tester._constraints_applied

    def test_from_file_no_metadata_flag(self):
        """Default from_file does NOT auto-detect metadata."""
        data = make_data()
        games = _minimal_games()
        meta = {
            'constraints_applied': [
                {'name': 'NoDoubleBookingTeamsConstraint'},
            ],
            'solver_config': {
                'excluded_constraints': [],
                'constraint_slack': {},
            },
        }
        draw = make_draw(games, metadata=meta)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_draw.json')
            draw.save(path)

            tester = DrawTester.from_file(path, data)
            assert tester._constraints_applied is None  # legacy mode


class TestLegacyDrawNoMetadata:
    """Old draws without metadata still work."""

    def test_legacy_draw_no_metadata(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        # Empty metadata (DrawStorage uses pydantic, so metadata={} not None)
        draw.metadata = {}
        tester = DrawTester(draw, data)
        report = tester.run_violation_check()
        assert len(report.constraint_results) == 19
        assert report.metadata_source == 'none'

    def test_legacy_draw_empty_metadata(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        draw.metadata = {}
        tester = DrawTester(draw, data)
        report = tester.run_violation_check()
        assert len(report.constraint_results) == 19


class TestSolverNameNormalization:
    """'NoDoubleBookingTeamsConstraintAI' resolves correctly."""

    def test_solver_name_normalization(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data,
                            constraints_applied=['NoDoubleBookingTeamsConstraintAI'])
        report = tester.run_violation_check()

        result_map = {r.constraint: r for r in report.constraint_results}
        assert result_map['NoDoubleBookingTeams'].status in ('PASSED', 'VIOLATED')

    def test_ai_and_original_map_to_same(self):
        """Both AI and original solver names map to same canonical."""
        data = make_data()
        draw1 = make_draw(_minimal_games())
        draw2 = make_draw(_minimal_games())

        t1 = DrawTester(draw1, data,
                         constraints_applied=['NoDoubleBookingTeamsConstraint'])
        t2 = DrawTester(draw2, data,
                         constraints_applied=['NoDoubleBookingTeamsConstraintAI'])

        assert t1._constraints_applied == t2._constraints_applied


class TestDualCheckConstraint:
    """EqualGames + BalancedMatchups both run/skip together."""

    def test_dual_check_both_run(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data,
                            constraints_applied=['EnsureEqualGamesAndBalanceMatchUps'])
        report = tester.run_violation_check()

        result_map = {r.constraint: r for r in report.constraint_results}
        # Both EqualGames and BalancedMatchups map to the same canonical
        eq_result = result_map.get('EqualGamesAndBalanceMatchUps')
        assert eq_result is not None
        assert eq_result.status in ('PASSED', 'VIOLATED')

    def test_dual_check_both_skip(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data,
                            excluded_constraints=['EnsureEqualGamesAndBalanceMatchUps'])
        report = tester.run_violation_check()

        result_map = {r.constraint: r for r in report.constraint_results}
        assert result_map['EqualGamesAndBalanceMatchUps'].status == 'SKIPPED'


class TestTesterOnlyConstraint:
    """ClubFieldConcentration always runs (no solver equivalent)."""

    def test_tester_only_constraint_always_runs(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        # Only apply one solver constraint, but ClubFieldConcentration should still run
        tester = DrawTester(draw, data,
                            constraints_applied=['NoDoubleBookingTeamsConstraint'])
        report = tester.run_violation_check()

        result_map = {r.constraint: r for r in report.constraint_results}
        assert result_map['ClubFieldConcentration'].status in ('PASSED', 'VIOLATED')

    def test_tester_only_respects_explicit_exclude(self):
        """ClubFieldConcentration can be explicitly excluded."""
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data,
                            excluded_constraints=['ClubFieldConcentration'])
        report = tester.run_violation_check()

        result_map = {r.constraint: r for r in report.constraint_results}
        assert result_map['ClubFieldConcentration'].status == 'SKIPPED'


class TestMetadataSourceField:
    """Report shows 'draw_json', 'checkpoint', or 'none'."""

    def test_metadata_source_none(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data)
        report = tester.run_violation_check()
        assert report.metadata_source == 'none'

    def test_metadata_source_draw_json(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data,
                            constraints_applied=['NoDoubleBookingTeamsConstraint'])
        report = tester.run_violation_check()
        assert report.metadata_source == 'draw_json'


class TestFullReportIncludesConstraintResults:
    """full_report() still works and includes new constraint results section."""

    def test_full_report_with_constraint_results(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data)
        report = tester.run_violation_check()
        text = report.full_report()
        assert 'CONSTRAINT VIOLATION REPORT' in text
        assert 'Constraints:' in text
        assert 'passed' in text

    def test_full_report_skipped_section(self):
        data = make_data()
        draw = make_draw(_minimal_games())
        tester = DrawTester(draw, data,
                            constraints_applied=['NoDoubleBookingTeamsConstraint'])
        report = tester.run_violation_check()
        text = report.full_report()
        assert 'Skipped:' in text
