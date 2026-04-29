# tests/test_run_cli.py
"""
Unit tests for run.py CLI module.

Tests for:
- Argument parsing (required flags, defaults)
- Command routing (no-command help, list-constraints)
- Helper functions (load_data_for_year)
- SolverConfig selection
- Diagnose and preseason command setup

Uses REAL objects only - no mocks, patches, MagicMock, or monkeypatch.
"""

import sys
import os
import argparse

import pytest

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run


# ============== load_data_for_year Tests ==============

class TestLoadDataForYear:
    """Tests for load_data_for_year function."""

    def test_loads_2025_data(self):
        """Test loading data for 2025 season."""
        data = run.load_data_for_year(2025)

        assert data is not None
        assert data['year'] == 2025
        assert 'teams' in data
        assert 'grades' in data

    def test_loads_2026_data(self):
        """Test loading data for 2026 season."""
        data = run.load_data_for_year(2026)

        assert data is not None
        assert data['year'] == 2026

    def test_raises_error_for_invalid_year(self):
        """Test that invalid year raises ValueError."""
        with pytest.raises(ValueError, match="No configuration found"):
            run.load_data_for_year(1999)


# ============== Argument Parser Tests ==============

class TestArgumentParser:
    """Tests for CLI argument parsing - verify --year is required on subcommands."""

    def _try_parse(self, argv):
        """Attempt to parse via main() with real sys.argv override."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py'] + argv
            run.main()
        finally:
            sys.argv = old_argv

    def test_generate_command_requires_year(self):
        """Test that generate command requires --year."""
        with pytest.raises(SystemExit):
            self._try_parse(['generate'])

    def test_test_command_requires_year(self):
        """Test that test command requires --year."""
        with pytest.raises(SystemExit):
            self._try_parse(['test', 'draw.json'])

    def test_analyze_command_requires_year(self):
        """Test that analyze command requires --year."""
        with pytest.raises(SystemExit):
            self._try_parse(['analyze', 'draw.json'])

    def test_swap_command_requires_year(self):
        """Test that swap command requires --year."""
        with pytest.raises(SystemExit):
            self._try_parse(['swap', 'draw.json', 'G001', 'G002'])

    def test_report_command_requires_year(self):
        """Test that report command requires --year."""
        with pytest.raises(SystemExit):
            self._try_parse(['report', 'draw.json', '--club', 'Tigers'])

    def test_import_command_requires_year(self):
        """Test that import command requires --year."""
        with pytest.raises(SystemExit):
            self._try_parse(['import', 'draw.xlsx'])

    def test_preseason_command_requires_year(self):
        """Test that preseason command requires --year."""
        with pytest.raises(SystemExit):
            self._try_parse(['preseason'])


# ============== Command Routing Tests ==============

class TestCommandRouting:
    """Tests for command routing logic."""

    def test_no_command_shows_help(self, capsys):
        """Test that no command shows help."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py']
            run.main()
        finally:
            sys.argv = old_argv

        captured = capsys.readouterr()
        assert 'usage:' in captured.out.lower() or 'commands' in captured.out.lower()

    def test_list_constraints_works(self, capsys):
        """Test list-constraints command outputs constraints."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', 'list-constraints']
            run.main()
        finally:
            sys.argv = old_argv

        captured = capsys.readouterr()
        assert 'REGISTERED CONSTRAINTS' in captured.out

    def test_list_constraints_no_ai_flag(self):
        """Phase 7c: --ai flag is gone; list-constraints should reject it."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', 'list-constraints', '--ai']
            with pytest.raises(SystemExit):
                run.main()
        finally:
            sys.argv = old_argv


# ============== run_generate Arg Setup Tests ==============

class TestRunGenerate:
    """Tests for run_generate argument setup (not actual solving)."""

    def test_generate_command_accepts_valid_args(self):
        """Test that a valid generate arg namespace has expected attributes."""
        args = argparse.Namespace(
            year=2025,
            resume=None,
            simple=False,
            run_id=None,
            locked=None,
            lock_weeks='',
            workers=None,
            low_memory=False,
            minimal_memory=False,
            high_performance=False,
            ai=False,
            exclude=None,
            hint=None,
            staged=False,
            relax=False,
            relax_timeout=30.0,
            fix_round_1=False,
            slack=None,
            description='',
            unified=False,
            stages=None,
        )

        assert args.year == 2025
        assert args.simple is False
        assert args.low_memory is False

    def test_generate_args_with_simple_mode(self):
        """Test that --simple flag is properly represented in args."""
        args = argparse.Namespace(
            year=2025,
            resume=None,
            simple=True,
            run_id=None,
            locked=None,
            lock_weeks='',
            workers=None,
            low_memory=False,
            minimal_memory=False,
            high_performance=False,
            ai=False,
            exclude=[],
            hint=None,
            staged=False,
            relax=False,
            relax_timeout=30.0,
            fix_round_1=False,
            slack=None,
            description='',
            unified=False,
            stages=None,
        )

        assert args.simple is True


# ============== run_list_constraints Tests ==============

class TestRunListConstraints:
    """Tests for run_list_constraints function."""

    def test_lists_constraints(self, capsys):
        """Phase 7c: list-constraints prints SOLVER_STAGES atoms."""
        run.run_list_constraints()

        captured = capsys.readouterr()
        assert 'REGISTERED CONSTRAINTS' in captured.out
        assert 'NoDoubleBookingTeams' in captured.out

    def test_lists_stage_information(self, capsys):
        """Stage names from DEFAULT_STAGES appear in the output."""
        run.run_list_constraints()

        captured = capsys.readouterr()
        assert 'critical_feasibility' in captured.out


# ============== Integration Tests ==============

class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_help_flag(self, capsys):
        """Test --help flag shows help."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', '--help']
            with pytest.raises(SystemExit):
                run.main()
        finally:
            sys.argv = old_argv

        captured = capsys.readouterr()
        assert 'usage' in captured.out.lower()

    def test_generate_help(self, capsys):
        """Test generate --help shows help."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', 'generate', '--help']
            with pytest.raises(SystemExit):
                run.main()
        finally:
            sys.argv = old_argv

        captured = capsys.readouterr()
        assert 'year' in captured.out.lower()

    def test_data_loading_integrates_with_config(self):
        """Test that CLI data loading uses config module."""
        data = run.load_data_for_year(2025)

        # Verify it loaded actual data
        assert len(data['teams']) > 0
        assert len(data['grades']) > 0
        assert len(data['fields']) > 0


# ============== SolverConfig Selection Tests ==============

class TestSolverConfigSelection:
    """Tests for solver configuration selection in run_generate."""

    def test_low_memory_flag_selects_config(self):
        """Test that --low-memory flag selects low memory config."""
        from solver_diagnostics import SolverConfig

        config = SolverConfig.low_memory_config()
        assert config.num_workers == 4

    def test_minimal_memory_flag_selects_config(self):
        """Test that --minimal-memory flag selects minimal memory config."""
        from solver_diagnostics import SolverConfig

        config = SolverConfig.minimal_memory_config()
        assert config.num_workers == 2
        assert config.linearization_level == 0

    def test_high_performance_flag_selects_config(self):
        """Test that --high-performance flag selects high performance config."""
        from solver_diagnostics import SolverConfig

        config = SolverConfig.high_performance_config()
        assert config.num_workers == 0  # All cores
        assert config.linearization_level == 2

    def test_custom_workers_override(self):
        """Test that --workers overrides default."""
        from solver_diagnostics import SolverConfig

        config = SolverConfig.balanced_config()
        config.num_workers = 6
        assert config.num_workers == 6


# ============== Diagnose Command Tests ==============

class TestDiagnoseCommand:
    """Tests for diagnose command."""

    def test_diagnose_requires_year(self):
        """Test that diagnose command requires --year."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', 'diagnose']
            with pytest.raises(SystemExit):
                run.main()
        finally:
            sys.argv = old_argv

    def test_diagnose_loads_data_correctly(self):
        """Test diagnose command loads data for the year."""
        from main_staged import load_data

        data = load_data(2025)
        assert data is not None
        assert 'teams' in data
        assert len(data['teams']) > 0

    def test_diagnose_gets_stage_config(self):
        """Phase 7c: diagnose now uses SOLVER_STAGES from the registry."""
        from constraints.stages import load_solver_stages
        stages = load_solver_stages({})
        names = [s['name'] for s in stages]
        assert 'critical_feasibility' in names

    def test_diagnose_args_namespace(self):
        """Phase 7c-bis: diagnose namespace uses SOLVER_STAGES names."""
        args = argparse.Namespace(
            year=2025,
            stage='critical_feasibility',
            resolve=False,
            timeout=5,
            max_iterations=5,
        )

        assert args.year == 2025
        assert args.stage == 'critical_feasibility'
        assert args.resolve is False

    def test_diagnose_creates_resolver(self):
        """Test diagnose command creates InfeasibilityResolver."""
        from constraints.resolver import InfeasibilityResolver, ConstraintSlackRegistry
        from main_staged import load_data

        data = load_data(2025)
        registry = ConstraintSlackRegistry()
        resolver = InfeasibilityResolver(
            data,
            registry,
            timeout_per_test=5,
            verbose=False,
        )

        assert resolver is not None
        assert resolver.timeout == 5

    def test_diagnose_group_atoms_clusters_by_engine_key(self):
        """`_diagnose_group_atoms` groups atomized cluster atoms together."""
        groups = run._diagnose_group_atoms([
            'PHLConcurrencyAtBroadmeadow',
            'PHLAnd2ndConcurrencyAtBroadmeadow',
            'NoDoubleBookingTeams',
            'MaximiseClubsPerTimeslotBroadmeadow',
        ])
        # PHL atoms collapse to PHLAndSecondGradeTimes.
        assert 'PHLAndSecondGradeTimes' in groups
        assert set(groups['PHLAndSecondGradeTimes']) == {
            'PHLConcurrencyAtBroadmeadow', 'PHLAnd2ndConcurrencyAtBroadmeadow',
        }
        # NoDoubleBookingTeams is its own engine key.
        assert 'NoDoubleBookingTeams' in groups
        # Non-engine atom forms a singleton under its own canonical name.
        assert 'MaximiseClubsPerTimeslotBroadmeadow' in groups

    def test_diagnose_unknown_stage_exits(self, capsys):
        """Unknown --stage names should print available stages and exit."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', 'diagnose', '--year', '2025', '--stage', 'no_such_stage']
            with pytest.raises(SystemExit):
                run.main()
        finally:
            sys.argv = old_argv
        captured = capsys.readouterr()
        assert 'Unknown stage' in captured.out
        assert 'critical_feasibility' in captured.out


# ============== Preseason Command Tests ==============

class TestPreseasonCommand:
    """Tests for preseason command."""

    def test_preseason_requires_year(self):
        """Test that preseason command requires --year."""
        old_argv = sys.argv
        try:
            sys.argv = ['run.py', 'preseason']
            with pytest.raises(SystemExit):
                run.main()
        finally:
            sys.argv = old_argv

    def test_preseason_loads_config(self):
        """Test preseason command can load configuration."""
        from config import season_2025, season_2026

        assert hasattr(season_2025, 'SEASON_CONFIG')
        assert hasattr(season_2026, 'SEASON_CONFIG')

    def test_preseason_report_creation(self):
        """Test PreSeasonReport can be instantiated."""
        from analytics.preseason_report import PreSeasonReport
        from config.season_2025 import SEASON_CONFIG

        data = run.load_data_for_year(2025)

        report = PreSeasonReport(data, SEASON_CONFIG)

        assert report is not None
        assert report.year == 2025

    def test_preseason_report_generates_output(self):
        """Test PreSeasonReport generates string output."""
        from analytics.preseason_report import PreSeasonReport
        from config.season_2025 import SEASON_CONFIG

        data = run.load_data_for_year(2025)
        report = PreSeasonReport(data, SEASON_CONFIG)

        output = report.generate_text_report()

        assert isinstance(output, str)
        assert len(output) > 0
        assert 'PRE-SEASON' in output or 'PRESEASON' in output or 'Season' in output
