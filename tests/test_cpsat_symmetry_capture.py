# tests/test_cpsat_symmetry_capture.py
"""
spec-035 Unit B (DoD-3) — no-mock tests for CP-SAT log capture + symmetry parsing.

Real CP-SAT, no mocks:
  1. A TINY hand-derivable symmetric model is solved with the capture wired and
     log_search_progress=True. We assert the captured run log contains CP-SAT's
     presolve [Symmetry] block AND that parse_symmetry_stats() returns integer
     num_generators / num_orbits.
  2. A log with no symmetry block -> parse_symmetry_stats returns the explicit
     {'present': False} sentinel and LOGS that at INFO via the 'solver' logger.

Hand-derived symmetry oracle for test 1
----------------------------------------
Model: assign N=8 fully-interchangeable items to K=3 colors.
  - one-hot:  for each item i,  sum_c x[i,c] == 1
  - capacity: for each color c, sum_i x[i,c] <= 3
The items carry no individual data, so ANY permutation of the 8 items is a model
symmetry (the constraints are invariant under relabelling items). The item-variable
set is therefore a single orbit. CP-SAT's presolve confirms this on ortools
9.15.6755: it reports `#generators: 8` and `1 orbits on 24 variables`. We assert
both are present as integers (and that the orbit count is exactly 1, matching the
single fully-interchangeable item group). The exact generator count can vary with
the presolved graph, so we assert it is a positive int (>0) rather than pinning 8.
"""

import logging
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ortools.sat.python import cp_model

from solver_diagnostics import (
    setup_logging,
    attach_cpsat_log_capture,
    parse_symmetry_stats,
    CPSAT_LOG_PREFIX,
)


def _build_symmetric_model(n_items: int = 8, n_colors: int = 3, cap: int = 3):
    """N fully-interchangeable items -> K colors, each color used at most `cap`.

    Items have no distinguishing data, so any item permutation is a symmetry.
    """
    model = cp_model.CpModel()
    x = {}
    for i in range(n_items):
        for c in range(n_colors):
            x[(i, c)] = model.NewBoolVar(f"x_{i}_{c}")
    for i in range(n_items):
        model.Add(sum(x[(i, c)] for c in range(n_colors)) == 1)
    for c in range(n_colors):
        model.Add(sum(x[(i, c)] for i in range(n_items)) <= cap)
    return model, x


def _get_log_file(logger: logging.Logger) -> Path:
    """Return the path backing the logger's FileHandler."""
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            return Path(h.baseFilename)
    raise AssertionError("solver logger has no FileHandler")


class TestCpsatSymmetryCapture:
    def test_symmetry_block_captured_and_parsed(self, tmp_path):
        # GIVEN the project 'solver' logger writing to a real file in tmp_path,
        logger = setup_logging(log_dir=str(tmp_path), run_id="symtest")
        log_file = _get_log_file(logger)

        # AND a tiny hand-derivable symmetric model,
        model, _x = _build_symmetric_model()

        # WHEN solved with the capture wired and log_search_progress=True,
        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        solver.parameters.num_workers = 1
        solver.parameters.max_time_in_seconds = 10
        attach_cpsat_log_capture(solver, logger)
        status = solver.Solve(model)

        # sanity: the solve actually ran
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

        # flush handlers so the file is complete
        for h in logger.handlers:
            h.flush()

        text = log_file.read_text(encoding="utf-8", errors="replace")

        # THEN CP-SAT's own presolve output reached the run log (DoD-3),
        assert f"{CPSAT_LOG_PREFIX} | Starting CP-SAT solver" in text, \
            "CP-SAT stdout was not captured into the run log"
        # AND the presolve symmetry block landed in the file.
        assert "[Symmetry]" in text, "presolve [Symmetry] block not in run log"

        # AND parse_symmetry_stats returns integer generators/orbits (oracle: 1 orbit).
        stats = parse_symmetry_stats(log_file)
        assert stats["present"] is True
        assert isinstance(stats["num_generators"], int)
        assert isinstance(stats["num_orbits"], int)
        assert stats["num_generators"] > 0
        # Hand oracle: 8 fully-interchangeable items => exactly one variable orbit.
        assert stats["num_orbits"] == 1
        # variables-in-orbit count is reported (24 = 8 items x 3 colors)
        assert stats["num_variables_in_orbits"] == 24

    def test_capture_does_not_break_solution_callback(self, tmp_path):
        # GIVEN the capture wired AND a real solution callback (the monitor/callback
        # path must keep working — DoD-8 / spec risk: capture must not break it),
        logger = setup_logging(log_dir=str(tmp_path), run_id="cbtest")
        model, _x = _build_symmetric_model()

        class _CountingCallback(cp_model.CpSolverSolutionCallback):
            def __init__(self):
                super().__init__()
                self.count = 0

            def on_solution_callback(self):
                self.count += 1

        solver = cp_model.CpSolver()
        solver.parameters.log_search_progress = True
        solver.parameters.num_workers = 1
        solver.parameters.max_time_in_seconds = 10
        attach_cpsat_log_capture(solver, logger)
        cb = _CountingCallback()

        # WHEN solved with BOTH the log_callback and a solution callback,
        status = solver.Solve(model, cb)

        # THEN the solution callback still fired (log_callback is a distinct hook).
        assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
        assert cb.count >= 1


class TestParseSymmetryStatsNotPresent:
    def test_no_symmetry_block_returns_sentinel_and_logs_info(self, tmp_path, caplog):
        # GIVEN a log file with NO [Symmetry] block,
        log_file = tmp_path / "no_symmetry.log"
        log_file.write_text(
            "2026-01-01 | INFO | solver | CPSAT | Starting CP-SAT solver v9.15\n"
            "2026-01-01 | INFO | solver | CPSAT | Starting presolve at 0.00s\n"
            "2026-01-01 | INFO | solver | CPSAT | Starting search at 0.01s\n",
            encoding="utf-8",
        )

        # WHEN parsed (capture INFO from the project 'solver' logger),
        with caplog.at_level(logging.INFO, logger="solver"):
            stats = parse_symmetry_stats(log_file)

        # THEN the explicit not-present sentinel is returned (never silent None),
        assert stats == {"present": False}
        # AND it was logged at INFO via the standard logger.
        assert any(
            rec.levelno == logging.INFO and "no [Symmetry] block" in rec.message
            for rec in caplog.records
        ), "not-present case must be logged at INFO via the 'solver' logger"

    def test_missing_file_returns_sentinel_and_logs_info(self, tmp_path, caplog):
        # GIVEN a path that does not exist,
        missing = tmp_path / "does_not_exist.log"

        # WHEN parsed,
        with caplog.at_level(logging.INFO, logger="solver"):
            stats = parse_symmetry_stats(missing)

        # THEN sentinel + INFO log (never crash, never silent None).
        assert stats == {"present": False}
        assert any(
            rec.levelno == logging.INFO and "not found" in rec.message
            for rec in caplog.records
        )
