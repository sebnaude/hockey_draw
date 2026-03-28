# tests/test_solver_diagnostics.py
"""
Unit tests for solver_diagnostics.py - no mocks, real objects only.

Tests for:
- SolverConfig class and factory methods
- ResourceMonitor class
- ResourceSnapshot dataclass
- setup_logging function
- get_recommended_config function
- DiagnosticSolutionCallback
- Utility logging functions
"""

import pytest
import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver_diagnostics import (
    setup_logging,
    ResourceSnapshot,
    ResourceMonitor,
    SolverConfig,
    get_recommended_config,
    DiagnosticSolutionCallback,
    log_system_info,
    log_model_info,
    log_solve_result,
    PSUTIL_AVAILABLE,
)

from ortools.sat.python import cp_model


# ============== SolverConfig Tests ==============

class TestSolverConfig:
    """Tests for SolverConfig class."""

    def test_default_config(self):
        """Test default SolverConfig values."""
        config = SolverConfig()

        assert config.max_time_seconds == 259200
        assert config.num_workers == 8
        assert config.random_seed == 42
        assert config.log_search_progress is True
        assert config.linearization_level == 1
        assert config.cp_model_presolve is True
        assert config.cp_model_probing_level == 2

    def test_custom_config(self):
        """Test creating SolverConfig with custom values."""
        config = SolverConfig(
            max_time_seconds=3600,
            num_workers=4,
            random_seed=123,
            linearization_level=0
        )

        assert config.max_time_seconds == 3600
        assert config.num_workers == 4
        assert config.random_seed == 123
        assert config.linearization_level == 0

    def test_low_memory_config(self):
        """Test low_memory_config factory method."""
        config = SolverConfig.low_memory_config()

        assert config.num_workers == 4
        assert config.linearization_level == 0
        assert config.cp_model_probing_level == 1

    def test_low_memory_config_custom_time(self):
        """Test low_memory_config with custom time limit."""
        config = SolverConfig.low_memory_config(max_time=3600)

        assert config.max_time_seconds == 3600

    def test_minimal_memory_config(self):
        """Test minimal_memory_config factory method."""
        config = SolverConfig.minimal_memory_config()

        assert config.num_workers == 2
        assert config.linearization_level == 0
        assert config.cp_model_probing_level == 0
        assert config.cp_model_presolve is True

    def test_balanced_config(self):
        """Test balanced_config factory method."""
        config = SolverConfig.balanced_config()

        assert config.num_workers == 8
        assert config.linearization_level == 1
        assert config.cp_model_probing_level == 2

    def test_high_performance_config(self):
        """Test high_performance_config factory method."""
        config = SolverConfig.high_performance_config()

        assert config.num_workers == 0  # Use all cores
        assert config.linearization_level == 2
        assert config.cp_model_probing_level == 3

    def test_apply_to_solver(self):
        """Test applying config to a real CP-SAT solver."""
        config = SolverConfig(
            max_time_seconds=3600,
            num_workers=4,
            random_seed=123
        )

        solver = cp_model.CpSolver()
        config.apply_to_solver(solver)

        assert solver.parameters.max_time_in_seconds == 3600
        assert solver.parameters.num_workers == 4
        assert solver.parameters.random_seed == 123
        assert solver.parameters.log_search_progress is True


# ============== ResourceSnapshot Tests ==============

class TestResourceSnapshot:
    """Tests for ResourceSnapshot dataclass."""

    def test_create_snapshot(self):
        """Test creating a ResourceSnapshot."""
        now = datetime.now()
        snapshot = ResourceSnapshot(
            timestamp=now,
            memory_used_mb=4096.0,
            memory_percent=50.0,
            memory_available_mb=4096.0,
            cpu_percent=25.0,
            process_memory_mb=512.0,
            process_cpu_percent=10.0
        )

        assert snapshot.timestamp == now
        assert snapshot.memory_used_mb == 4096.0
        assert snapshot.memory_percent == 50.0
        assert snapshot.memory_available_mb == 4096.0
        assert snapshot.cpu_percent == 25.0
        assert snapshot.process_memory_mb == 512.0
        assert snapshot.process_cpu_percent == 10.0


# ============== ResourceMonitor Tests ==============

class TestResourceMonitor:
    """Tests for ResourceMonitor class."""

    def test_create_monitor(self):
        """Test creating a ResourceMonitor."""
        monitor = ResourceMonitor()

        assert monitor.memory_warning_threshold == 80.0
        assert monitor.memory_critical_threshold == 95.0
        assert monitor._monitoring is False

    def test_create_monitor_custom_thresholds(self):
        """Test creating ResourceMonitor with custom thresholds."""
        monitor = ResourceMonitor(
            memory_warning_threshold=70.0,
            memory_critical_threshold=90.0
        )

        assert monitor.memory_warning_threshold == 70.0
        assert monitor.memory_critical_threshold == 90.0

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_get_snapshot(self):
        """Test getting a resource snapshot with psutil available."""
        monitor = ResourceMonitor()
        snapshot = monitor.get_snapshot()

        assert snapshot is not None
        assert isinstance(snapshot, ResourceSnapshot)
        assert snapshot.memory_percent >= 0
        assert snapshot.memory_percent <= 100
        assert snapshot.process_memory_mb > 0

    def test_get_snapshot_without_psutil(self):
        """Test that get_snapshot returns None without psutil."""
        if not PSUTIL_AVAILABLE:
            monitor = ResourceMonitor()
            snapshot = monitor.get_snapshot()
            assert snapshot is None

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_log_snapshot(self):
        """Test logging a resource snapshot."""
        logger = logging.getLogger("test_monitor")
        monitor = ResourceMonitor(logger=logger)

        snapshot = monitor.log_snapshot(prefix="TEST")

        assert snapshot is not None

    def test_get_peak_memory_no_snapshots(self):
        """Test get_peak_memory with no snapshots."""
        monitor = ResourceMonitor()

        result = monitor.get_peak_memory()
        assert result is None

    def test_get_memory_summary_no_snapshots(self):
        """Test get_memory_summary with no snapshots."""
        monitor = ResourceMonitor()

        result = monitor.get_memory_summary()
        assert result == {}


# ============== setup_logging Tests ==============

class TestSetupLogging:
    """Tests for setup_logging function."""

    def _cleanup_logger(self, logger):
        """Close and remove all handlers from logger to release file locks."""
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    def test_setup_logging_creates_logger(self, tmp_path):
        """Test that setup_logging creates a logger."""
        logger = setup_logging(log_dir=str(tmp_path))

        try:
            assert logger is not None
            assert logger.name == "solver"
            assert logger.level <= logging.DEBUG
        finally:
            self._cleanup_logger(logger)

    def test_setup_logging_creates_log_directory(self, tmp_path):
        """Test that setup_logging creates log directory if needed."""
        log_dir = str(tmp_path / "sublogs")

        logger = setup_logging(log_dir=log_dir)

        try:
            assert os.path.exists(log_dir)
        finally:
            self._cleanup_logger(logger)

    def test_setup_logging_with_run_id(self, tmp_path):
        """Test setup_logging with run_id in filename."""
        logger = setup_logging(log_dir=str(tmp_path), run_id="test_run")

        try:
            log_files = list(tmp_path.glob("*.log"))
            assert len(log_files) >= 1
            has_run_id = any("test_run" in f.name for f in log_files)
            assert has_run_id
        finally:
            self._cleanup_logger(logger)

    def test_setup_logging_has_handlers(self, tmp_path):
        """Test that setup_logging configures handlers."""
        logger = setup_logging(log_dir=str(tmp_path))

        try:
            assert len(logger.handlers) >= 1
        finally:
            self._cleanup_logger(logger)


# ============== get_recommended_config Tests ==============

class TestGetRecommendedConfig:
    """Tests for get_recommended_config function."""

    def test_low_memory_recommendation(self):
        """Test config recommendation for low memory."""
        config = get_recommended_config(available_memory_mb=2000)

        assert config.num_workers == 4
        assert config.linearization_level == 0

    def test_medium_memory_recommendation(self):
        """Test config recommendation for medium memory."""
        config = get_recommended_config(available_memory_mb=6000)

        assert config.num_workers == 8

    def test_high_memory_recommendation(self):
        """Test config recommendation for high memory."""
        config = get_recommended_config(available_memory_mb=16000)

        assert config.num_workers == 0  # All cores
        assert config.linearization_level == 2

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_auto_detect_memory(self):
        """Test that auto-detection works when psutil available."""
        config = get_recommended_config()

        assert config is not None
        assert isinstance(config, SolverConfig)


# ============== DiagnosticSolutionCallback Tests ==============

class TestDiagnosticSolutionCallback:
    """Tests for DiagnosticSolutionCallback class."""

    def test_create_callback(self):
        """Test creating a DiagnosticSolutionCallback."""
        callback = DiagnosticSolutionCallback()

        assert callback.diag_solution_times == []

    def test_log_solution_found(self):
        """Test logging a solution."""
        logger = logging.getLogger("test_callback")
        callback = DiagnosticSolutionCallback(logger=logger)

        callback.log_solution_found(
            solution_num=1,
            objective=100.0,
            elapsed=10.5
        )

        assert len(callback.diag_solution_times) == 1
        assert callback.diag_solution_times[0] == (10.5, 100.0)

    def test_log_multiple_solutions(self):
        """Test logging multiple solutions."""
        callback = DiagnosticSolutionCallback()

        callback.log_solution_found(1, 100.0, 10.0)
        callback.log_solution_found(2, 90.0, 20.0)
        callback.log_solution_found(3, 85.0, 30.0)

        assert len(callback.diag_solution_times) == 3
        assert callback.diag_solution_times[2] == (30.0, 85.0)

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_log_solution_with_resource_monitor(self):
        """Test logging a solution with resource monitor attached."""
        monitor = ResourceMonitor()
        callback = DiagnosticSolutionCallback(resource_monitor=monitor)

        callback.log_solution_found(
            solution_num=1,
            objective=50.0,
            elapsed=5.0
        )

        assert len(callback.diag_solution_times) == 1
        assert callback.diag_solution_times[0] == (5.0, 50.0)


# ============== Utility Function Tests ==============

class TestUtilityFunctions:
    """Tests for utility logging functions."""

    def test_log_system_info(self):
        """Test log_system_info with a real logger."""
        logger = logging.getLogger("test_system_info")
        logger.setLevel(logging.DEBUG)

        # Should not raise any exceptions
        log_system_info(logger)

    def test_log_model_info(self):
        """Test log_model_info with a real CP-SAT model."""
        logger = logging.getLogger("test_model_info")
        logger.setLevel(logging.DEBUG)

        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        y = model.new_bool_var("y")
        model.Add(x + y <= 1)

        X = {0: x, 1: y}

        # Should not raise any exceptions
        log_model_info(model, X, logger)

    def test_log_solve_result(self):
        """Test log_solve_result with a real logger."""
        logger = logging.getLogger("test_solve_result")
        logger.setLevel(logging.DEBUG)

        log_solve_result(
            status_name="OPTIMAL",
            objective=150.0,
            solve_time=300.5,
            games_scheduled=250,
            logger=logger
        )


# ============== Integration Tests ==============

class TestSolverDiagnosticsIntegration:
    """Integration tests for solver diagnostics module."""

    def _cleanup_logger(self, logger):
        """Close and remove all handlers from logger to release file locks."""
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_full_monitoring_workflow(self, tmp_path):
        """Test complete monitoring workflow."""
        logger = setup_logging(log_dir=str(tmp_path), run_id="integration_test")

        try:
            monitor = ResourceMonitor(logger=logger)

            snapshot = monitor.get_snapshot()
            if snapshot is not None:
                assert isinstance(snapshot, ResourceSnapshot)

            config = get_recommended_config()
            assert isinstance(config, SolverConfig)
        finally:
            self._cleanup_logger(logger)

    def test_config_pipeline(self):
        """Test config selection pipeline."""
        configs = [
            SolverConfig.low_memory_config(),
            SolverConfig.minimal_memory_config(),
            SolverConfig.balanced_config(),
            SolverConfig.high_performance_config(),
        ]

        for config in configs:
            assert isinstance(config, SolverConfig)
            assert config.max_time_seconds > 0
            assert config.num_workers >= 0

    def test_solve_trivial_model_with_config(self):
        """Test applying config to a real solver and solving a trivial model."""
        model = cp_model.CpModel()
        x = model.new_bool_var("x")
        model.Maximize(x)

        solver = cp_model.CpSolver()
        config = SolverConfig(max_time_seconds=10, num_workers=1)
        config.apply_to_solver(solver)

        status = solver.Solve(model)

        assert status == cp_model.OPTIMAL
        assert solver.value(x) == 1
