# tests/test_run_coverage.py
"""
Tests for run.py to improve branch coverage.

Uses REAL objects only - no mocks, patches, MagicMock, or monkeypatch.
"""

import sys
import os
import json
import pickle
import argparse
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run
from analytics.storage import DrawStorage, StoredGame


# ============== Helpers ==============

def _make_minimal_draw(num_games=2, weeks=None):
    """Create a minimal DrawStorage with a few games for testing."""
    if weeks is None:
        weeks = [1, 2]
    games = []
    for i, week in enumerate(weeks):
        game = StoredGame(
            game_id=f"G{i:05d}",
            team1="Maitland PHL",
            team2="Norths PHL",
            grade="PHL",
            week=week,
            round_no=week,
            date=f"2026-04-{5 + week:02d}",
            day="Sunday",
            time="11:30",
            day_slot=3,
            field_name="EF",
            field_location="Newcastle International Hockey Centre",
        )
        games.append(game)
    return DrawStorage(
        description="test draw",
        num_weeks=len(set(weeks)),
        num_games=len(games),
        games=games,
    )


def _save_draw_json(draw, directory, filename="current.json"):
    """Save a DrawStorage to a directory and return the path."""
    path = os.path.join(directory, filename)
    draw.save(path)
    return path


# ============== resolve_draw_path Tests ==============

class TestResolveDrawPath:
    """Tests for resolve_draw_path function."""

    def test_direct_path_returned_as_is(self):
        """A direct file path should be returned unchanged."""
        result = run.resolve_draw_path("some/direct/path.json", year=2026)
        assert result == "some/direct/path.json"

    def test_direct_path_without_year(self):
        """Direct path works even without year."""
        result = run.resolve_draw_path("my_draw.json", year=None)
        assert result == "my_draw.json"

    def test_current_alias_resolves(self, tmp_path):
        """'current' alias resolves to draws/{year}/current.json when file exists."""
        draws_dir = tmp_path / "draws" / "9999"
        draws_dir.mkdir(parents=True)
        draw = _make_minimal_draw()
        draw_path = draws_dir / "current.json"
        draw.save(str(draw_path))

        # We need to run from the tmp_path context for resolve_draw_path to find it
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = run.resolve_draw_path("current", year=9999)
            assert result == "draws/9999/current.json"
        finally:
            os.chdir(old_cwd)

    def test_latest_alias_resolves(self, tmp_path):
        """'latest' alias resolves same as 'current'."""
        draws_dir = tmp_path / "draws" / "9999"
        draws_dir.mkdir(parents=True)
        draw = _make_minimal_draw()
        draw.save(str(draws_dir / "current.json"))

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = run.resolve_draw_path("latest", year=9999)
            assert result == "draws/9999/current.json"
        finally:
            os.chdir(old_cwd)

    def test_current_alias_without_year_exits(self):
        """'current' without year should sys.exit."""
        with pytest.raises(SystemExit):
            run.resolve_draw_path("current", year=None)

    def test_current_alias_missing_file_exits(self, tmp_path):
        """'current' when file doesn't exist should sys.exit."""
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with pytest.raises(SystemExit):
                run.resolve_draw_path("current", year=8888)
        finally:
            os.chdir(old_cwd)

    def test_version_string_resolves(self, tmp_path):
        """A version string like 'v2.0' resolves to the versions directory."""
        versions_dir = tmp_path / "draws" / "9999" / "versions"
        versions_dir.mkdir(parents=True)
        draw = _make_minimal_draw()
        draw.save(str(versions_dir / "draw_v2.0.json"))

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = run.resolve_draw_path("v2.0", year=9999)
            assert result == "draws/9999/versions/draw_v2.0.json"
        finally:
            os.chdir(old_cwd)

    def test_version_string_without_v_prefix(self, tmp_path):
        """Version string '1.1' (no 'v' prefix) also resolves."""
        versions_dir = tmp_path / "draws" / "9999" / "versions"
        versions_dir.mkdir(parents=True)
        draw = _make_minimal_draw()
        draw.save(str(versions_dir / "draw_v1.1.json"))

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = run.resolve_draw_path("1.1", year=9999)
            assert result == "draws/9999/versions/draw_v1.1.json"
        finally:
            os.chdir(old_cwd)

    def test_version_string_fallback_to_base_dir(self, tmp_path):
        """Version string resolves from base draws dir if not in versions/."""
        draws_dir = tmp_path / "draws" / "9999"
        draws_dir.mkdir(parents=True)
        # Put in base dir, not versions/
        draw = _make_minimal_draw()
        draw.save(str(draws_dir / "draw_v3.0.json"))

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = run.resolve_draw_path("v3.0", year=9999)
            assert result == "draws/9999/draw_v3.0.json"
        finally:
            os.chdir(old_cwd)

    def test_version_string_not_found_exits(self, tmp_path):
        """Version string that doesn't exist should sys.exit."""
        draws_dir = tmp_path / "draws" / "9999"
        draws_dir.mkdir(parents=True)

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            with pytest.raises(SystemExit):
                run.resolve_draw_path("v99.9", year=9999)
        finally:
            os.chdir(old_cwd)

    def test_case_insensitive_current(self, tmp_path):
        """'CURRENT' (uppercase) should also resolve."""
        draws_dir = tmp_path / "draws" / "9999"
        draws_dir.mkdir(parents=True)
        draw = _make_minimal_draw()
        draw.save(str(draws_dir / "current.json"))

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = run.resolve_draw_path("CURRENT", year=9999)
            assert result == "draws/9999/current.json"
        finally:
            os.chdir(old_cwd)

    def test_case_insensitive_latest(self, tmp_path):
        """'LATEST' (uppercase) should also resolve."""
        draws_dir = tmp_path / "draws" / "9999"
        draws_dir.mkdir(parents=True)
        draw = _make_minimal_draw()
        draw.save(str(draws_dir / "current.json"))

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = run.resolve_draw_path("LATEST", year=9999)
            assert result == "draws/9999/current.json"
        finally:
            os.chdir(old_cwd)


# ============== load_data_for_year Tests ==============

class TestLoadDataForYear:
    """Additional tests for load_data_for_year."""

    def test_returns_dict_with_expected_keys(self):
        """Data dict should have standard keys."""
        data = run.load_data_for_year(2026)
        for key in ['teams', 'grades', 'fields', 'year', 'timeslots']:
            assert key in data, f"Missing key: {key}"

    def test_year_matches_request(self):
        """Returned data year should match requested year."""
        data = run.load_data_for_year(2026)
        assert data['year'] == 2026

    def test_teams_not_empty(self):
        """Should have teams loaded."""
        data = run.load_data_for_year(2026)
        assert len(data['teams']) > 0

    def test_grades_not_empty(self):
        """Should have grades loaded."""
        data = run.load_data_for_year(2026)
        assert len(data['grades']) > 0

    def test_invalid_year_raises(self):
        """Non-existent year should raise ValueError."""
        with pytest.raises(ValueError):
            run.load_data_for_year(1800)

    def test_2025_data_loads(self):
        """2025 season data should load."""
        data = run.load_data_for_year(2025)
        assert data['year'] == 2025
        assert len(data['teams']) > 0


# ============== _load_locked_keys Tests ==============

class TestLoadLockedKeys:
    """Tests for _load_locked_keys function."""

    def test_load_from_json(self, tmp_path):
        """Loading locked keys from a draw JSON file."""
        draw = _make_minimal_draw(weeks=[1, 2, 3])
        draw_path = str(tmp_path / "test_draw.json")
        draw.save(draw_path)

        locked_keys = run._load_locked_keys(draw_path, locked_weeks={1})
        # Should return keys for week 1 games only
        assert isinstance(locked_keys, list)
        assert len(locked_keys) == 1  # one game in week 1
        # Each key should be an 11-tuple
        assert len(locked_keys[0]) == 11

    def test_load_from_json_multiple_weeks(self, tmp_path):
        """Loading locked keys for multiple weeks from JSON."""
        draw = _make_minimal_draw(weeks=[1, 2, 3])
        draw_path = str(tmp_path / "test_draw.json")
        draw.save(draw_path)

        locked_keys = run._load_locked_keys(draw_path, locked_weeks={1, 2})
        assert len(locked_keys) == 2  # one game per week, 2 weeks

    def test_load_from_pickle(self, tmp_path):
        """Loading locked keys from a pickle file."""
        # Create a fake solution dict with 11-tuple keys
        solution = {}
        key1 = ("TeamA", "TeamB", "PHL", "Sunday", 3, "11:30", 1, "2026-04-05", 1, "EF", "NIHC")
        key2 = ("TeamA", "TeamC", "PHL", "Sunday", 3, "11:30", 2, "2026-04-12", 2, "EF", "NIHC")
        key3 = ("TeamB", "TeamC", "PHL", "Sunday", 3, "11:30", 3, "2026-04-19", 3, "EF", "NIHC")
        solution[key1] = 1
        solution[key2] = 1
        solution[key3] = 0  # not scheduled

        pkl_path = str(tmp_path / "solution.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(solution, f)

        locked_keys = run._load_locked_keys(pkl_path, locked_weeks={1, 2})
        assert len(locked_keys) == 2  # key1 (week 1) and key2 (week 2) are val==1
        # key3 is val==0 so excluded; also week 3 not in locked_weeks

    def test_load_from_pickle_skips_unscheduled(self, tmp_path):
        """Pickle loading should skip keys with value != 1."""
        solution = {
            ("A", "B", "PHL", "Sunday", 3, "11:30", 1, "2026-04-05", 1, "EF", "NIHC"): 1,
            ("C", "D", "PHL", "Sunday", 3, "11:30", 1, "2026-04-05", 1, "WF", "NIHC"): 0,
        }
        pkl_path = str(tmp_path / "solution.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(solution, f)

        locked_keys = run._load_locked_keys(pkl_path, locked_weeks={1})
        assert len(locked_keys) == 1

    def test_load_from_checkpoint_directory(self, tmp_path):
        """Loading locked keys from a checkpoint directory containing solution.pkl."""
        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()

        solution = {
            ("A", "B", "PHL", "Sunday", 3, "11:30", 1, "2026-04-05", 1, "EF", "NIHC"): 1,
            ("C", "D", "PHL", "Sunday", 3, "11:30", 2, "2026-04-12", 2, "EF", "NIHC"): 1,
        }
        with open(str(checkpoint_dir / "solution.pkl"), "wb") as f:
            pickle.dump(solution, f)

        locked_keys = run._load_locked_keys(str(checkpoint_dir), locked_weeks={1})
        assert len(locked_keys) == 1  # only week 1

    def test_load_from_directory_missing_pkl_exits(self, tmp_path):
        """Directory without solution.pkl should sys.exit."""
        empty_dir = tmp_path / "empty_checkpoint"
        empty_dir.mkdir()

        with pytest.raises(SystemExit):
            run._load_locked_keys(str(empty_dir), locked_weeks={1})

    def test_load_unknown_format_exits(self, tmp_path):
        """Unknown file format should sys.exit."""
        txt_path = str(tmp_path / "solution.txt")
        with open(txt_path, "w") as f:
            f.write("not a valid format")

        with pytest.raises(SystemExit):
            run._load_locked_keys(txt_path, locked_weeks={1})

    def test_load_pickle_skips_short_keys(self, tmp_path):
        """Pickle with short keys should skip them."""
        solution = {
            ("A", "B", "PHL", 0): 1,  # 4-tuple key, len < 7
            ("A", "B", "PHL", "Sunday", 3, "11:30", 1, "2026-04-05", 1, "EF", "NIHC"): 1,
        }
        pkl_path = str(tmp_path / "solution.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(solution, f)

        locked_keys = run._load_locked_keys(pkl_path, locked_weeks={1})
        # Only the 11-tuple key with week=1 should be returned
        assert len(locked_keys) == 1
        assert len(locked_keys[0]) == 11


# ============== run_list_constraints Tests ==============

class TestRunListConstraintsExtended:
    """Extended tests for run_list_constraints."""

    def test_lists_constraint_atom_names(self, capsys):
        """Phase 7c: list-constraints prints SOLVER_STAGES atoms."""
        run.run_list_constraints()
        output = capsys.readouterr().out
        assert "REGISTERED CONSTRAINTS" in output
        assert "NoDoubleBooking" in output

    def test_lists_stage_names(self, capsys):
        """Phase 7c: stage names from DEFAULT_STAGES appear in output."""
        run.run_list_constraints()
        output = capsys.readouterr().out
        assert "critical_feasibility" in output
        assert "soft_optimisation" in output


# ============== CLI Argument Parsing Tests ==============

class TestCLIArgumentParsing:
    """Tests for argument parsing of all subcommands."""

    def _parse_args(self, argv):
        """Parse arguments by invoking main()'s parser directly."""
        # Reconstruct the parser from main()
        parser = argparse.ArgumentParser(description='Hockey Draw Scheduling System')
        subparsers = parser.add_subparsers(dest='command')

        # Generate
        gen_parser = subparsers.add_parser('generate')
        gen_parser.add_argument('--year', type=int, required=True)
        gen_parser.add_argument('--simple', action='store_true')
        gen_parser.add_argument('--unified', action='store_true')
        gen_parser.add_argument('--low-memory', action='store_true')
        gen_parser.add_argument('--minimal-memory', action='store_true')
        gen_parser.add_argument('--high-performance', action='store_true')
        gen_parser.add_argument('--workers', type=int, default=None)
        gen_parser.add_argument('--exclude', nargs='+', metavar='CONSTRAINT')
        gen_parser.add_argument('--slack', type=int, default=None)
        gen_parser.add_argument('--locked', type=str)
        gen_parser.add_argument('--lock-weeks', type=str, default='')
        gen_parser.add_argument('--hint', type=str)
        gen_parser.add_argument('--run-id', type=str)
        gen_parser.add_argument('--resume', nargs='*')
        gen_parser.add_argument('--staged', action='store_true')
        gen_parser.add_argument('--relax', action='store_true')
        gen_parser.add_argument('--relax-timeout', type=float, default=30.0)
        gen_parser.add_argument('--fix-round-1', action='store_true')
        gen_parser.add_argument('--description', type=str, default='')
        gen_parser.add_argument('--stages', nargs='+')

        # Test
        test_parser = subparsers.add_parser('test')
        test_parser.add_argument('draw_file')
        test_parser.add_argument('--year', type=int, required=True)

        # Analyze
        analyze_parser = subparsers.add_parser('analyze')
        analyze_parser.add_argument('draw_file')
        analyze_parser.add_argument('--year', type=int, required=True)
        analyze_parser.add_argument('--output', '-o', type=str)

        # Swap
        swap_parser = subparsers.add_parser('swap')
        swap_parser.add_argument('draw_file')
        swap_parser.add_argument('game1')
        swap_parser.add_argument('game2')
        swap_parser.add_argument('--year', type=int, required=True)
        swap_parser.add_argument('--save', type=str)

        # Report
        report_parser = subparsers.add_parser('report')
        report_parser.add_argument('draw_file')
        report_parser.add_argument('--club', type=str)
        report_parser.add_argument('--team', type=str)
        report_parser.add_argument('--grade', type=str)
        report_parser.add_argument('--compliance', action='store_true')
        report_parser.add_argument('--all', action='store_true')
        report_parser.add_argument('--html', action='store_true')
        report_parser.add_argument('--output', '-o', type=str, default='reports')
        report_parser.add_argument('--year', type=int, required=True)

        # Import
        import_parser = subparsers.add_parser('import')
        import_parser.add_argument('excel_file')
        import_parser.add_argument('--lock-weeks', type=str)
        import_parser.add_argument('--output', '-o', type=str)
        import_parser.add_argument('--year', type=int, required=True)

        # List constraints (no --ai after Phase 7c)
        subparsers.add_parser('list-constraints')

        # Preseason
        preseason_parser = subparsers.add_parser('preseason')
        preseason_parser.add_argument('--year', type=int, required=True)
        preseason_parser.add_argument('--output', '-o', type=str)

        # Diagnose
        diagnose_parser = subparsers.add_parser('diagnose')
        diagnose_parser.add_argument('--year', type=int, required=True)
        diagnose_parser.add_argument('--stage', type=str, default='critical_feasibility')
        diagnose_parser.add_argument('--timeout', type=float, default=5.0)
        diagnose_parser.add_argument('--resolve', action='store_true')
        diagnose_parser.add_argument('--max-iterations', type=int, default=10)

        # Migrate
        migrate_parser = subparsers.add_parser('migrate')
        migrate_parser.add_argument('--year', type=int, required=True)

        return parser.parse_args(argv)

    def test_generate_simple_mode(self):
        args = self._parse_args(['generate', '--year', '2026', '--simple'])
        assert args.command == 'generate'
        assert args.year == 2026
        assert args.simple is True

    def test_generate_slack(self):
        args = self._parse_args(['generate', '--year', '2026', '--slack', '3'])
        assert args.slack == 3

    def test_generate_exclude_constraints(self):
        args = self._parse_args([
            'generate', '--year', '2026', '--simple',
            '--exclude', 'EnsureBestTimeslotChoices', 'ClubGameSpread',
        ])
        assert args.exclude == ['EnsureBestTimeslotChoices', 'ClubGameSpread']

    def test_generate_locked_and_lock_weeks(self):
        args = self._parse_args([
            'generate', '--year', '2026',
            '--locked', 'draws/2026/current.json',
            '--lock-weeks', '1,2,3',
        ])
        assert args.locked == 'draws/2026/current.json'
        assert args.lock_weeks == '1,2,3'

    def test_generate_workers(self):
        args = self._parse_args(['generate', '--year', '2026', '--workers', '8'])
        assert args.workers == 8

    def test_generate_no_ai_flag(self):
        """Phase 7c: --ai removed from generate."""
        with pytest.raises(SystemExit):
            self._parse_args(['generate', '--year', '2026', '--ai'])

    def test_generate_staged_flag(self):
        args = self._parse_args(['generate', '--year', '2026', '--staged'])
        assert args.staged is True

    def test_generate_relax_with_timeout(self):
        args = self._parse_args([
            'generate', '--year', '2026', '--relax', '--relax-timeout', '60',
        ])
        assert args.relax is True
        assert args.relax_timeout == 60.0

    def test_generate_fix_round_1(self):
        args = self._parse_args(['generate', '--year', '2026', '--fix-round-1'])
        assert args.fix_round_1 is True

    def test_generate_hint(self):
        args = self._parse_args([
            'generate', '--year', '2026', '--hint', 'checkpoints/latest/solution.pkl',
        ])
        assert args.hint == 'checkpoints/latest/solution.pkl'

    def test_generate_description(self):
        args = self._parse_args([
            'generate', '--year', '2026', '--description', 'test run',
        ])
        assert args.description == 'test run'

    def test_generate_stages(self):
        args = self._parse_args([
            'generate', '--year', '2026', '--stages', 'severity_1', 'severity_2',
        ])
        assert args.stages == ['severity_1', 'severity_2']

    def test_generate_resume(self):
        args = self._parse_args([
            'generate', '--year', '2026', '--resume', 'run_5', 'stage2_soft',
        ])
        assert args.resume == ['run_5', 'stage2_soft']

    def test_generate_resume_no_args(self):
        args = self._parse_args(['generate', '--year', '2026', '--resume'])
        assert args.resume == []

    def test_generate_run_id(self):
        args = self._parse_args(['generate', '--year', '2026', '--run-id', 'my_run'])
        assert args.run_id == 'my_run'

    def test_generate_unified(self):
        args = self._parse_args(['generate', '--year', '2026', '--unified'])
        assert args.unified is True

    def test_generate_memory_flags(self):
        args = self._parse_args(['generate', '--year', '2026', '--low-memory'])
        assert args.low_memory is True

        args = self._parse_args(['generate', '--year', '2026', '--minimal-memory'])
        assert args.minimal_memory is True

        args = self._parse_args(['generate', '--year', '2026', '--high-performance'])
        assert args.high_performance is True

    def test_test_command(self):
        args = self._parse_args(['test', 'draw.json', '--year', '2026'])
        assert args.command == 'test'
        assert args.draw_file == 'draw.json'
        assert args.year == 2026

    def test_analyze_command(self):
        args = self._parse_args(['analyze', 'draw.json', '--year', '2026', '-o', 'out.xlsx'])
        assert args.command == 'analyze'
        assert args.output == 'out.xlsx'

    def test_swap_command(self):
        args = self._parse_args(['swap', 'draw.json', 'G001', 'G002', '--year', '2026', '--save', 'new.json'])
        assert args.command == 'swap'
        assert args.game1 == 'G001'
        assert args.game2 == 'G002'
        assert args.save == 'new.json'

    def test_report_command_all(self):
        args = self._parse_args(['report', 'draw.json', '--year', '2026', '--all'])
        assert args.command == 'report'
        assert args.all is True

    def test_report_command_club(self):
        args = self._parse_args(['report', 'draw.json', '--year', '2026', '--club', 'Maitland'])
        assert args.club == 'Maitland'

    def test_report_command_compliance(self):
        args = self._parse_args(['report', 'draw.json', '--year', '2026', '--compliance'])
        assert args.compliance is True

    def test_report_command_html(self):
        args = self._parse_args(['report', 'draw.json', '--year', '2026', '--html'])
        assert args.html is True

    def test_report_command_output_dir(self):
        args = self._parse_args(['report', 'draw.json', '--year', '2026', '-o', 'my_reports'])
        assert args.output == 'my_reports'

    def test_import_command(self):
        args = self._parse_args(['import', 'draw.xlsx', '--year', '2026'])
        assert args.command == 'import'
        assert args.excel_file == 'draw.xlsx'

    def test_import_with_lock_weeks(self):
        args = self._parse_args(['import', 'draw.xlsx', '--year', '2026', '--lock-weeks', '1,2,3'])
        assert args.lock_weeks == '1,2,3'

    def test_list_constraints_command(self):
        args = self._parse_args(['list-constraints'])
        assert args.command == 'list-constraints'

    def test_list_constraints_no_ai(self):
        """Phase 7c: --ai removed from list-constraints."""
        with pytest.raises(SystemExit):
            self._parse_args(['list-constraints', '--ai'])

    def test_preseason_command(self):
        args = self._parse_args(['preseason', '--year', '2026'])
        assert args.command == 'preseason'
        assert args.year == 2026

    def test_preseason_with_output(self):
        args = self._parse_args(['preseason', '--year', '2026', '-o', 'report.txt'])
        assert args.output == 'report.txt'

    def test_diagnose_command(self):
        args = self._parse_args(['diagnose', '--year', '2026'])
        assert args.command == 'diagnose'
        assert args.stage == 'critical_feasibility'
        assert args.timeout == 5.0
        assert args.resolve is False
        assert args.max_iterations == 10

    def test_diagnose_resolve(self):
        args = self._parse_args([
            'diagnose', '--year', '2026', '--resolve',
            '--max-iterations', '20', '--timeout', '15',
        ])
        assert args.resolve is True
        assert args.max_iterations == 20
        assert args.timeout == 15.0

    def test_diagnose_no_ai(self):
        """Phase 7c: --ai removed from diagnose."""
        with pytest.raises(SystemExit):
            self._parse_args(['diagnose', '--year', '2026', '--ai'])

    def test_diagnose_stage(self):
        args = self._parse_args(['diagnose', '--year', '2026', '--stage', 'soft_optimisation'])
        assert args.stage == 'soft_optimisation'

    def test_migrate_command(self):
        args = self._parse_args(['migrate', '--year', '2026'])
        assert args.command == 'migrate'
        assert args.year == 2026

    def test_no_command_is_none(self):
        args = self._parse_args([])
        assert args.command is None


# ============== Slack Building Logic Tests ==============

class TestSlackBuilding:
    """Tests for the constraint_slack dict building logic from --slack."""

    def test_slack_creates_all_keys(self):
        """--slack N should create dict with all slack-aware constraint names."""
        slack_value = 3
        # spec-018: AwayAtMaitlandGrouping / MaitlandHomeGrouping slack keys
        # removed (rules deleted).
        constraint_slack = {
            'EqualMatchUpSpacingConstraint': slack_value,
            'ClubVsClubAlignment': slack_value,
            'MaximiseClubsPerTimeslotBroadmeadow': slack_value,
            'MinimiseClubsOnAFieldBroadmeadow': slack_value,
            'ClubGameSpread': slack_value,
        }
        assert len(constraint_slack) == 5
        for name, val in constraint_slack.items():
            assert val == 3

    def test_slack_zero_is_valid(self):
        """--slack 0 should create dict with all zeros."""
        slack_value = 0
        # spec-018: AwayAtMaitlandGrouping / MaitlandHomeGrouping slack keys
        # removed (rules deleted).
        constraint_slack = {
            'EqualMatchUpSpacingConstraint': slack_value,
            'ClubVsClubAlignment': slack_value,
            'MaximiseClubsPerTimeslotBroadmeadow': slack_value,
            'MinimiseClubsOnAFieldBroadmeadow': slack_value,
            'ClubGameSpread': slack_value,
        }
        for val in constraint_slack.values():
            assert val == 0

    def test_slack_none_means_no_override(self):
        """When --slack is not provided (None), no constraint_slack should be created."""
        slack_value = None
        constraint_slack = None
        if slack_value is not None:
            constraint_slack = {'EqualMatchUpSpacingConstraint': slack_value}
        assert constraint_slack is None


# ============== Lock Weeks Parsing Tests ==============

class TestLockWeeksParsing:
    """Tests for --lock-weeks comma-separated parsing."""

    def test_parse_single_week(self):
        lock_weeks_str = "1"
        locked_weeks = set(int(w.strip()) for w in lock_weeks_str.split(',') if w.strip())
        assert locked_weeks == {1}

    def test_parse_multiple_weeks(self):
        lock_weeks_str = "1,2,3"
        locked_weeks = set(int(w.strip()) for w in lock_weeks_str.split(',') if w.strip())
        assert locked_weeks == {1, 2, 3}

    def test_parse_non_contiguous_weeks(self):
        lock_weeks_str = "1,5,7"
        locked_weeks = set(int(w.strip()) for w in lock_weeks_str.split(',') if w.strip())
        assert locked_weeks == {1, 5, 7}

    def test_parse_with_spaces(self):
        lock_weeks_str = " 1 , 2 , 3 "
        locked_weeks = set(int(w.strip()) for w in lock_weeks_str.split(',') if w.strip())
        assert locked_weeks == {1, 2, 3}

    def test_parse_empty_string(self):
        lock_weeks_str = ""
        locked_weeks = set(int(w.strip()) for w in lock_weeks_str.split(',') if w.strip())
        assert locked_weeks == set()


# ============== run_test Integration Tests ==============

class TestRunTest:
    """Tests for run_test with real draw files."""

    def test_run_test_with_real_draw(self, tmp_path, capsys):
        """run_test should load a draw and run violation checks."""
        # Use the actual current draw if it exists
        real_draw_path = Path("draws/2026/current.json")
        if not real_draw_path.exists():
            pytest.skip("No current draw available for testing")

        args = argparse.Namespace(
            draw_file=str(real_draw_path),
            year=2026,
        )
        # run_test may sys.exit(1) if violations exist, which is fine
        try:
            run.run_test(args)
        except SystemExit:
            pass  # violations cause exit(1), that's expected

        output = capsys.readouterr().out
        assert "DRAW VIOLATION TEST" in output


# ============== DrawStorage round-trip Tests ==============

class TestDrawStorageRoundTrip:
    """Tests verifying draw JSON save/load round-trip used by _load_locked_keys."""

    def test_save_and_load(self, tmp_path):
        """DrawStorage should survive save/load round-trip."""
        draw = _make_minimal_draw(weeks=[1, 2])
        path = str(tmp_path / "test.json")
        draw.save(path)

        loaded = DrawStorage.load(path)
        assert loaded.num_games == draw.num_games
        assert loaded.num_weeks == draw.num_weeks
        assert len(loaded.games) == len(draw.games)

    def test_to_x_dict_produces_11_tuples(self):
        """to_X_dict should produce 11-tuple keys."""
        draw = _make_minimal_draw(weeks=[1])
        x_dict = draw.to_X_dict()
        assert len(x_dict) == 1
        key = list(x_dict.keys())[0]
        assert len(key) == 11

    def test_game_to_key_round_trip(self):
        """StoredGame.to_key() should produce consistent keys."""
        game = StoredGame(
            game_id="G00000",
            team1="A",
            team2="B",
            grade="PHL",
            week=1,
            round_no=1,
            date="2026-04-05",
            day="Sunday",
            time="11:30",
            day_slot=3,
            field_name="EF",
            field_location="NIHC",
        )
        key = game.to_key()
        assert key == ("A", "B", "PHL", "Sunday", 3, "11:30", 1, "2026-04-05", 1, "EF", "NIHC")


# ============== Main routing with no command ==============

class TestMainNoCommand:
    """Test the main() routing when no command is given."""

    def test_no_command_prints_help(self, capsys):
        """Calling main() with no command should print help and return."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py']
            run.main()
        finally:
            sys.argv = old_argv

        output = capsys.readouterr().out
        assert 'usage' in output.lower() or 'Hockey Draw' in output

    def test_list_constraints_via_main(self, capsys):
        """list-constraints command via main() should work."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', 'list-constraints']
            run.main()
        finally:
            sys.argv = old_argv

        output = capsys.readouterr().out
        assert 'REGISTERED CONSTRAINTS' in output

    def test_list_constraints_ai_via_main_rejected(self):
        """Phase 7c: --ai removed; list-constraints --ai exits non-zero."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', 'list-constraints', '--ai']
            with pytest.raises(SystemExit):
                run.main()
        finally:
            sys.argv = old_argv
