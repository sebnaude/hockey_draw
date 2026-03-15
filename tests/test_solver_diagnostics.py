# tests/test_solver_diagnostics.py
"""
Unit tests for solver_diagnostics.py

Tests for:
- SolverConfig class and factory methods
- ResourceMonitor class
- ResourceSnapshot dataclass
- setup_logging function
- get_recommended_config function
- Utility logging functions
"""

import pytest
import sys
import os
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

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
        """Test applying config to a mock solver."""
        config = SolverConfig(
            max_time_seconds=3600,
            num_workers=4,
            random_seed=123
        )
        
        # Create a mock solver with parameters attribute
        mock_solver = Mock()
        mock_solver.parameters = Mock()
        
        config.apply_to_solver(mock_solver)
        
        # Verify all parameters were set
        assert mock_solver.parameters.max_time_in_seconds == 3600
        assert mock_solver.parameters.num_workers == 4
        assert mock_solver.parameters.random_seed == 123
        assert mock_solver.parameters.log_search_progress is True


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
        """Test getting a resource snapshot."""
        monitor = ResourceMonitor()
        snapshot = monitor.get_snapshot()
        
        assert snapshot is not None
        assert isinstance(snapshot, ResourceSnapshot)
        assert snapshot.memory_percent >= 0
        assert snapshot.memory_percent <= 100
        assert snapshot.process_memory_mb > 0

    def test_get_snapshot_without_psutil(self):
        """Test that get_snapshot returns None without psutil."""
        monitor = ResourceMonitor()
        
        # If psutil is not available, should return None
        if not PSUTIL_AVAILABLE:
            snapshot = monitor.get_snapshot()
            assert snapshot is None

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_log_snapshot(self):
        """Test logging a resource snapshot."""
        logger = logging.getLogger("test_monitor")
        monitor = ResourceMonitor(logger=logger)
        
        snapshot = monitor.log_snapshot(prefix="TEST")
        
        # Should return the snapshot
        if PSUTIL_AVAILABLE:
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

    def test_setup_logging_creates_logger(self):
        """Test that setup_logging creates a logger."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = setup_logging(log_dir=tmp_dir)
            
            try:
                assert logger is not None
                assert logger.name == "solver"
                # Logger level should be set (DEBUG level = 10)
                assert logger.level <= logging.DEBUG
            finally:
                self._cleanup_logger(logger)

    def test_setup_logging_creates_log_directory(self):
        """Test that setup_logging creates log directory if needed."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = os.path.join(tmp_dir, "sublogs")
            
            logger = setup_logging(log_dir=log_dir)
            
            try:
                assert os.path.exists(log_dir)
            finally:
                self._cleanup_logger(logger)

    def test_setup_logging_with_run_id(self):
        """Test setup_logging with run_id in filename."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = setup_logging(log_dir=tmp_dir, run_id="test_run")
            
            try:
                # Check that a log file was created
                log_files = list(Path(tmp_dir).glob("*.log"))
                assert len(log_files) >= 1
                # At least one should contain run_id
                has_run_id = any("test_run" in f.name for f in log_files)
                assert has_run_id
            finally:
                self._cleanup_logger(logger)

    def test_setup_logging_has_handlers(self):
        """Test that setup_logging configures handlers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            logger = setup_logging(log_dir=tmp_dir)
            
            try:
                # Should have at least one handler (file handler)
                assert len(logger.handlers) >= 1
            finally:
                self._cleanup_logger(logger)


# ============== get_recommended_config Tests ==============

class TestGetRecommendedConfig:
    """Tests for get_recommended_config function."""

    def test_low_memory_recommendation(self):
        """Test config recommendation for low memory."""
        config = get_recommended_config(available_memory_mb=2000)
        
        # Should return low-memory config
        assert config.num_workers == 4
        assert config.linearization_level == 0

    def test_medium_memory_recommendation(self):
        """Test config recommendation for medium memory."""
        config = get_recommended_config(available_memory_mb=6000)
        
        # Should return balanced config
        assert config.num_workers == 8

    def test_high_memory_recommendation(self):
        """Test config recommendation for high memory."""
        config = get_recommended_config(available_memory_mb=16000)
        
        # Should return high-performance config
        assert config.num_workers == 0  # All cores
        assert config.linearization_level == 2

    @pytest.mark.skipif(not PSUTIL_AVAILABLE, reason="psutil not installed")
    def test_auto_detect_memory(self):
        """Test that auto-detection works when psutil available."""
        config = get_recommended_config()
        
        # Should return some valid config
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


# ============== Utility Function Tests ==============

class TestUtilityFunctions:
    """Tests for utility logging functions."""

    def test_log_system_info(self):
        """Test log_system_info doesn't crash."""
        logger = logging.getLogger("test_system_info")
        logger.setLevel(logging.DEBUG)
        
        # Should not raise any exceptions
        log_system_info(logger)

    def test_log_model_info(self):
        """Test log_model_info with mock model."""
        logger = logging.getLogger("test_model_info")
        
        # Create a mock model with Proto() method
        mock_model = Mock()
        mock_proto = Mock()
        mock_proto.constraints = [1, 2, 3]  # 3 constraints
        mock_model.Proto.return_value = mock_proto
        
        X = {i: Mock() for i in range(100)}  # 100 variables
        
        # Should not raise any exceptions
        log_model_info(mock_model, X, logger)

    def test_log_solve_result(self):
        """Test log_solve_result doesn't crash."""
        logger = logging.getLogger("test_solve_result")
        
        # Should not raise any exceptions
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
    def test_full_monitoring_workflow(self):
        """Test complete monitoring workflow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Setup logging
            logger = setup_logging(log_dir=tmp_dir, run_id="integration_test")
            
            try:
                # Create and use resource monitor
                monitor = ResourceMonitor(logger=logger)
                
                # Get a snapshot
                snapshot = monitor.get_snapshot()
                if snapshot is not None:  # psutil is available
                    assert isinstance(snapshot, ResourceSnapshot)
                
                # Get recommended config
                config = get_recommended_config()
                assert isinstance(config, SolverConfig)
            finally:
                self._cleanup_logger(logger)

    def test_config_pipeline(self):
        """Test config selection pipeline."""
        # Test all factory methods produce valid configs
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
