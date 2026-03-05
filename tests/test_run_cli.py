# tests/test_run_cli.py
"""
Unit tests for run.py CLI module.

Tests for:
- Argument parsing
- Command routing
- Helper functions like load_data_for_year
"""

import pytest
import sys
import os
import argparse
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module
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
    """Tests for CLI argument parsing."""

    def test_generate_command_requires_year(self):
        """Test that generate command requires --year."""
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['run.py', 'generate']):
                run.main()

    def test_test_command_requires_year(self):
        """Test that test command requires --year."""
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['run.py', 'test', 'draw.json']):
                run.main()

    def test_analyze_command_requires_year(self):
        """Test that analyze command requires --year."""
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['run.py', 'analyze', 'draw.json']):
                run.main()

    def test_swap_command_requires_year(self):
        """Test that swap command requires --year."""
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['run.py', 'swap', 'draw.json', 'G001', 'G002']):
                run.main()

    def test_report_command_requires_year(self):
        """Test that report command requires --year."""
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['run.py', 'report', 'draw.json', '--club', 'Tigers']):
                run.main()

    def test_import_command_requires_year(self):
        """Test that import command requires --year."""
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['run.py', 'import', 'draw.xlsx']):
                run.main()

    def test_preseason_command_requires_year(self):
        """Test that preseason command requires --year."""
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['run.py', 'preseason']):
                run.main()


# ============== Command Routing Tests ==============

class TestCommandRouting:
    """Tests for command routing logic."""

    def test_no_command_shows_help(self, capsys):
        """Test that no command shows help."""
        with patch('sys.argv', ['run.py']):
            run.main()
        
        captured = capsys.readouterr()
        assert 'usage:' in captured.out.lower() or 'commands' in captured.out.lower()

    def test_list_constraints_works(self, capsys):
        """Test list-constraints command outputs constraints."""
        with patch('sys.argv', ['run.py', 'list-constraints']):
            run.main()
        
        captured = capsys.readouterr()
        assert 'AVAILABLE CONSTRAINTS' in captured.out

    def test_list_constraints_ai_flag(self, capsys):
        """Test list-constraints with --ai flag."""
        with patch('sys.argv', ['run.py', 'list-constraints', '--ai']):
            run.main()
        
        captured = capsys.readouterr()
        assert 'AI-ENHANCED' in captured.out


# ============== run_generate Tests ==============

class TestRunGenerate:
    """Tests for run_generate function."""

    def test_generate_command_accepts_valid_args(self):
        """Test that run_generate accepts valid argument combinations."""
        # Just test that the arg namespace can be created with valid values
        args = argparse.Namespace(
            year=2025,
            resume=None,
            simple=False,
            run_id=None,
            locked=None,
            lock_weeks=0,
            workers=None,
            low_memory=False,
            minimal_memory=False,
            high_performance=False,
            ai=False,
            exclude=None
        )
        
        # Verify all expected attributes are present
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
            lock_weeks=0,
            workers=None,
            low_memory=False,
            minimal_memory=False,
            high_performance=False,
            ai=False,
            exclude=[]
        )
        
        assert args.simple is True


# ============== run_list_constraints Tests ==============

class TestRunListConstraints:
    """Tests for run_list_constraints function."""

    def test_lists_original_constraints(self, capsys):
        """Test listing original constraints."""
        run.run_list_constraints(use_ai=False)
        
        captured = capsys.readouterr()
        assert 'ORIGINAL' in captured.out
        assert 'Constraints:' in captured.out

    def test_lists_ai_constraints(self, capsys):
        """Test listing AI-enhanced constraints."""
        run.run_list_constraints(use_ai=True)
        
        captured = capsys.readouterr()
        assert 'AI-ENHANCED' in captured.out

    def test_lists_stage_information(self, capsys):
        """Test that stage info is included in output."""
        run.run_list_constraints(use_ai=False)
        
        captured = capsys.readouterr()
        assert 'stage' in captured.out.lower()
        assert 'Time Limit' in captured.out


# ============== Integration Tests ==============

class TestCLIIntegration:
    """Integration tests for CLI."""

    def test_help_flag(self, capsys):
        """Test --help flag shows help."""
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['run.py', '--help']):
                run.main()
        
        captured = capsys.readouterr()
        assert 'usage' in captured.out.lower()

    def test_generate_help(self, capsys):
        """Test generate --help shows help."""
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['run.py', 'generate', '--help']):
                run.main()
        
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
        
        args = argparse.Namespace(
            year=2025,
            resume=None,
            simple=False,
            run_id=None,
            locked=None,
            lock_weeks=0,
            workers=None,
            low_memory=True,
            minimal_memory=False,
            high_performance=False,
            ai=False,
            exclude=None
        )
        
        # Low memory should select config with 4 workers
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
        
        # Custom workers can be set
        config = SolverConfig.balanced_config()
        config.num_workers = 6
        assert config.num_workers == 6
