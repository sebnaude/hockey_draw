# solver_diagnostics.py
"""
Diagnostic utilities for monitoring solver performance and resource usage.

Provides:
- Memory monitoring and logging
- CPU usage tracking  
- Detailed solver progress logging
- Resource-aware solver configuration
"""

import os
import sys
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not installed. Memory monitoring disabled. Install with: pip install psutil")


# ============== Logging Configuration ==============

def setup_logging(log_dir: str = "logs", run_id: str = None, level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up comprehensive logging for solver runs.
    
    Args:
        log_dir: Directory for log files
        run_id: Identifier for this run (used in log filename)
        level: Logging level (default DEBUG for maximum detail)
    
    Returns:
        Configured logger instance
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_suffix = f"_{run_id}" if run_id else ""
    log_file = log_path / f"solver_{timestamp}{run_suffix}.log"
    
    # Create logger
    logger = logging.getLogger("solver")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    # Console handler - debug and above (shows memory stats)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


# ============== Resource Monitoring ==============

@dataclass
class ResourceSnapshot:
    """Snapshot of system resource usage."""
    timestamp: datetime
    memory_used_mb: float
    memory_percent: float
    memory_available_mb: float
    cpu_percent: float
    process_memory_mb: float
    process_cpu_percent: float


class ResourceMonitor:
    """
    Monitor system and process resource usage.
    
    Can run in background thread for continuous monitoring during solver execution.
    """
    
    def __init__(self, logger: logging.Logger = None, 
                 memory_warning_threshold: float = 80.0,
                 memory_critical_threshold: float = 95.0):
        """
        Args:
            logger: Logger instance for output
            memory_warning_threshold: % memory usage to trigger warning
            memory_critical_threshold: % memory usage to trigger critical alert
        """
        self.logger = logger or logging.getLogger("solver.resources")
        self.memory_warning_threshold = memory_warning_threshold
        self.memory_critical_threshold = memory_critical_threshold
        
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._snapshots: list[ResourceSnapshot] = []
        self._callback: Optional[Callable[[ResourceSnapshot], None]] = None
        
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None
    
    def get_snapshot(self) -> Optional[ResourceSnapshot]:
        """Get current resource usage snapshot."""
        if not PSUTIL_AVAILABLE:
            return None
        
        try:
            mem = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            
            proc_mem = self.process.memory_info()
            proc_cpu = self.process.cpu_percent(interval=0.1)
            
            snapshot = ResourceSnapshot(
                timestamp=datetime.now(),
                memory_used_mb=mem.used / (1024 * 1024),
                memory_percent=mem.percent,
                memory_available_mb=mem.available / (1024 * 1024),
                cpu_percent=cpu,
                process_memory_mb=proc_mem.rss / (1024 * 1024),
                process_cpu_percent=proc_cpu
            )
            
            return snapshot
        except Exception as e:
            self.logger.warning(f"Failed to get resource snapshot: {e}")
            return None
    
    def log_snapshot(self, snapshot: ResourceSnapshot = None, prefix: str = ""):
        """Log a resource snapshot."""
        if snapshot is None:
            snapshot = self.get_snapshot()
        
        if snapshot is None:
            return
        
        prefix_str = f"{prefix} | " if prefix else ""
        
        # Check thresholds
        level = logging.DEBUG
        if snapshot.memory_percent >= self.memory_critical_threshold:
            level = logging.CRITICAL
        elif snapshot.memory_percent >= self.memory_warning_threshold:
            level = logging.WARNING
        
        self.logger.log(
            level,
            f"{prefix_str}Memory: {snapshot.memory_percent:.1f}% "
            f"({snapshot.memory_used_mb:.0f}MB used, {snapshot.memory_available_mb:.0f}MB available) | "
            f"Process: {snapshot.process_memory_mb:.0f}MB | "
            f"CPU: {snapshot.cpu_percent:.1f}% (process: {snapshot.process_cpu_percent:.1f}%)"
        )
        
        return snapshot
    
    def start_monitoring(self, interval: float = 30.0, 
                        callback: Callable[[ResourceSnapshot], None] = None):
        """
        Start background monitoring thread.
        
        Args:
            interval: Seconds between snapshots
            callback: Optional callback for each snapshot (e.g., to abort on high memory)
        """
        if not PSUTIL_AVAILABLE:
            self.logger.warning("Cannot start monitoring: psutil not available")
            return
        
        if self._monitoring:
            self.logger.warning("Monitoring already running")
            return
        
        self._monitoring = True
        self._callback = callback
        self._snapshots = []
        
        def monitor_loop():
            while self._monitoring:
                snapshot = self.get_snapshot()
                if snapshot:
                    self._snapshots.append(snapshot)
                    self.log_snapshot(snapshot, "MONITOR")
                    
                    if self._callback:
                        self._callback(snapshot)
                
                time.sleep(interval)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info(f"Started resource monitoring (interval: {interval}s)")
    
    def stop_monitoring(self) -> list[ResourceSnapshot]:
        """Stop background monitoring and return collected snapshots."""
        self._monitoring = False
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None
        
        self.logger.info(f"Stopped monitoring. Collected {len(self._snapshots)} snapshots")
        return self._snapshots
    
    def get_peak_memory(self) -> Optional[float]:
        """Get peak process memory usage from collected snapshots."""
        if not self._snapshots:
            return None
        return max(s.process_memory_mb for s in self._snapshots)
    
    def get_memory_summary(self) -> dict:
        """Get summary statistics of memory usage."""
        if not self._snapshots:
            return {}
        
        proc_mems = [s.process_memory_mb for s in self._snapshots]
        sys_mems = [s.memory_percent for s in self._snapshots]
        
        return {
            "process_memory_mb": {
                "min": min(proc_mems),
                "max": max(proc_mems),
                "avg": sum(proc_mems) / len(proc_mems),
            },
            "system_memory_percent": {
                "min": min(sys_mems),
                "max": max(sys_mems),
                "avg": sum(sys_mems) / len(sys_mems),
            },
            "samples": len(self._snapshots)
        }


# ============== Solver Configuration ==============

@dataclass
class SolverConfig:
    """Configuration for OR-Tools CP-SAT solver with resource management."""
    
    # Time limits (72 hours, aligned with staged solver)
    max_time_seconds: int = 259200
    
    # Parallelism - reduces memory by limiting parallel workers
    num_workers: int = 8  # Default OR-Tools uses all cores which can be memory-heavy
    
    # Memory management
    # Note: OR-Tools doesn't have direct memory limits, but we can tune parameters
    random_seed: int = 42  # Deterministic for debugging
    
    # Search parameters
    log_search_progress: bool = True
    
    # CP-SAT specific tuning
    linearization_level: int = 1  # 0=none, 1=basic, 2=full (higher uses more memory)
    
    # Presolve (can reduce memory but takes time)
    cp_model_presolve: bool = True
    
    # Probing (uses memory but improves bounds)
    cp_model_probing_level: int = 2  # 0=disabled, 1=low, 2=medium, 3=high
    
    def apply_to_solver(self, solver) -> None:
        """Apply configuration to a CpSolver instance."""
        solver.parameters.max_time_in_seconds = self.max_time_seconds
        solver.parameters.num_workers = self.num_workers
        solver.parameters.random_seed = self.random_seed
        solver.parameters.log_search_progress = self.log_search_progress
        solver.parameters.linearization_level = self.linearization_level
        solver.parameters.cp_model_presolve = self.cp_model_presolve
        solver.parameters.cp_model_probing_level = self.cp_model_probing_level
    
    @classmethod
    def low_memory_config(cls, max_time: int = 259200) -> 'SolverConfig':
        """Create a configuration optimized for low memory usage."""
        return cls(
            max_time_seconds=max_time,
            num_workers=4,  # Fewer workers = less memory
            linearization_level=0,  # Disable linearization cuts (memory heavy)
            cp_model_probing_level=1,  # Lower probing
        )
    
    @classmethod
    def minimal_memory_config(cls, max_time: int = 259200) -> 'SolverConfig':
        """Create a configuration for extremely constrained memory (2 workers)."""
        return cls(
            max_time_seconds=max_time,
            num_workers=2,  # Minimal workers - much slower but far less memory
            linearization_level=0,  # No linearization
            cp_model_probing_level=0,  # Disable probing entirely
            cp_model_presolve=True,  # Keep presolve - actually reduces memory
        )
    
    @classmethod
    def balanced_config(cls, max_time: int = 259200) -> 'SolverConfig':
        """Create a balanced configuration."""
        return cls(
            max_time_seconds=max_time,
            num_workers=8,
            linearization_level=1,
            cp_model_probing_level=2,
        )
    
    @classmethod
    def high_performance_config(cls, max_time: int = 259200) -> 'SolverConfig':
        """Create a configuration optimized for speed (uses more memory)."""
        return cls(
            max_time_seconds=max_time,
            num_workers=0,  # 0 = use all cores
            linearization_level=2,
            cp_model_probing_level=3,
        )


def get_recommended_config(available_memory_mb: float = None) -> SolverConfig:
    """
    Get recommended solver configuration based on available memory.
    
    Args:
        available_memory_mb: Available memory in MB. If None, auto-detect.
    
    Returns:
        Appropriate SolverConfig
    """
    if available_memory_mb is None and PSUTIL_AVAILABLE:
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
    
    if available_memory_mb is None:
        # Can't detect, use balanced
        return SolverConfig.balanced_config()
    
    if available_memory_mb < 4000:  # Less than 4GB
        print(f"WARNING: Low available memory ({available_memory_mb:.0f}MB). Using low-memory config.")
        return SolverConfig.low_memory_config()
    elif available_memory_mb < 8000:  # Less than 8GB
        return SolverConfig.balanced_config()
    else:
        return SolverConfig.high_performance_config()


# ============== Diagnostic Callback ==============

class DiagnosticSolutionCallback:
    """
    Mixin for solution callbacks that adds diagnostic logging.
    
    Usage: Inherit alongside cp_model.CpSolverSolutionCallback
    """
    
    def __init__(self, logger: logging.Logger = None, resource_monitor: ResourceMonitor = None):
        self.diag_logger = logger or logging.getLogger("solver.callback")
        self.diag_monitor = resource_monitor
        self.diag_solution_times = []
    
    def log_solution_found(self, solution_num: int, objective: float, elapsed: float):
        """Log when a solution is found."""
        self.diag_solution_times.append((elapsed, objective))
        
        mem_info = ""
        if self.diag_monitor:
            snapshot = self.diag_monitor.get_snapshot()
            if snapshot:
                mem_info = f" | Memory: {snapshot.process_memory_mb:.0f}MB"
        
        self.diag_logger.info(
            f"Solution #{solution_num}: objective={objective:.0f}, time={elapsed:.1f}s{mem_info}"
        )


# ============== Utility Functions ==============

def log_system_info(logger: logging.Logger = None):
    """Log system information for debugging."""
    logger = logger or logging.getLogger("solver")
    
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {sys.platform}")
    
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        logger.info(f"Total memory: {mem.total / (1024**3):.1f} GB")
        logger.info(f"Available memory: {mem.available / (1024**3):.1f} GB")
        logger.info(f"CPU count: {psutil.cpu_count(logical=True)} (logical), {psutil.cpu_count(logical=False)} (physical)")
    
    try:
        from ortools.sat.python import cp_model
        logger.info(f"OR-Tools version: {cp_model.__version__ if hasattr(cp_model, '__version__') else 'unknown'}")
    except:
        pass
    
    logger.info("=" * 60)


def log_model_info(model, X: dict, logger: logging.Logger = None):
    """Log CP model statistics."""
    logger = logger or logging.getLogger("solver")
    
    proto = model.Proto()
    
    logger.info("MODEL STATISTICS")
    logger.info(f"  Variables: {len(X)}")
    logger.info(f"  Constraints: {len(proto.constraints)}")
    
    # Estimate memory (rough)
    estimated_mb = (len(X) * 8 + len(proto.constraints) * 100) / (1024 * 1024)
    logger.info(f"  Estimated base memory: {estimated_mb:.1f} MB")


def log_solve_result(status_name: str, objective: float, solve_time: float, 
                    games_scheduled: int, logger: logging.Logger = None):
    """Log solve results."""
    logger = logger or logging.getLogger("solver")
    
    logger.info("=" * 60)
    logger.info("SOLVE RESULT")
    logger.info("=" * 60)
    logger.info(f"Status: {status_name}")
    logger.info(f"Objective: {objective}")
    logger.info(f"Games scheduled: {games_scheduled}")
    logger.info(f"Solve time: {solve_time:.1f}s ({solve_time/60:.1f} min)")
    logger.info("=" * 60)
