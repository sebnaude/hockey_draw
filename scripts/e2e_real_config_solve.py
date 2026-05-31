#!/usr/bin/env python
# scripts/e2e_real_config_solve.py
"""REAL-2026-CONFIG e2e progression — emulates the spec-035 ultimate plan's stage
progression, but on the FULL production config (load_season_data(2026): the real
FORCED_GAMES (18), BLOCKED_GAMES (69, incl. the PHL premiership-weekend exemptions),
field unavailabilities, LOCKED_PAIRINGS) instead of the forced-free season_test,
and capped at ~5 min into solve per run instead of 30.

Same solve entry point as run_core_e2e.py (`main_staged.main_simple` via
`run._resolve_group_selection`), the ONLY difference being `year=2026` (real config)
and no forced-free assertion.

The 5 runs (same constraint-group steps the ultimate plan walked):
    1. core, exclude=ClubGameSpread
    2. core (full)
    3. core,bye_spacing
    4. core,spacing
    5. core,bye_spacing,spacing  (full sweep)

raw: --workers 8 (box OOMs at 10), week-1 NOT fixed, no locks, no slack.

Two modes:
  * default (driver): runs all 5 sequentially, each as a child subprocess killed at
    --minutes (default 5) with a psutil process-tree kill; writes a live progress file
    (scripts/_e2e_real_progress.txt) and a readout (scripts/e2e_real_readout.md).
  * --run-single: performs ONE real solve (used internally as the child).

Usage:
    python scripts/e2e_real_config_solve.py                  # all 5, 5 min each
    python scripts/e2e_real_config_solve.py --minutes 5
    python scripts/e2e_real_config_solve.py --only 2         # just run #2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REAL_YEAR = 2026

# The 5-stage progression (label, groups, exclude)
RUNS = [
    ("run1_core_noCGS", ["core"], ["ClubGameSpread"]),
    ("run2_core_full", ["core"], []),
    ("run3_core_bye", ["core", "bye_spacing"], []),
    ("run4_core_spacing", ["core", "spacing"], []),
    ("run5_full_sweep", ["core", "bye_spacing", "spacing"], []),
]


# --------------------------------------------------------------------------- #
# child: one real solve
# --------------------------------------------------------------------------- #
def run_single(groups, exclude, workers, run_id):
    import logging
    logger = logging.getLogger("solver")
    from solver_diagnostics import SolverConfig
    from run import _resolve_group_selection
    from main_staged import main_simple

    group_names, constraint_names = _resolve_group_selection(
        groups, exclude, no_symmetry_breakers=False
    )
    print(f"[e2e-real] year={REAL_YEAR} groups={groups} exclude={exclude} "
          f"workers={workers} -> {len(constraint_names)} constraints", flush=True)

    solver_config = SolverConfig.balanced_config()
    solver_config.num_workers = workers

    solution, data = main_simple(
        locked_keys=None,
        locked_weeks=set(),
        solver_config=solver_config,
        exclude_constraints=exclude,
        year=REAL_YEAR,
        relax_config=None,
        fix_round_1=False,
        constraint_slack=None,
        hint_solution=None,
        run_id=run_id,
        description=f"REAL-2026 e2e ({run_id}) groups={groups} exclude={exclude}",
        provenance={"spec": "spec-035-realconfig", "year": REAL_YEAR},
        constraint_names=constraint_names,
        groups_selected=group_names,
    )
    print(f"[e2e-real] {run_id}: solve returned (solution={'yes' if solution else 'none'})",
          flush=True)
    return solution


# --------------------------------------------------------------------------- #
# driver: sequential, killed at the cap
# --------------------------------------------------------------------------- #
def _tree_kill(proc):
    if proc.poll() is not None:
        return
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        victims = parent.children(recursive=True) + [parent]
        for v in victims:
            try:
                v.kill()
            except Exception:
                pass
        psutil.wait_procs(victims, timeout=20)
    except Exception:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, text=True)
        else:
            proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass


def _scan_log(run_id):
    """Best-effort: find the child's solver log and pull liveness/symmetry signals."""
    logdir = REPO_ROOT / "logs"
    matches = sorted(logdir.glob(f"solver_*_{run_id}.log"))
    if not matches:
        return {"log": None, "reached_search": False, "infeasible": False, "symmetry": None}
    log = matches[-1]
    txt = ""
    try:
        txt = log.read_text(encoding="utf-8", errors="replace")
    except Exception:
        pass
    reached = any(m in txt for m in ("#1 ", "#Bound", "#Done", "best:", "objective ",
                                     "CpSolverResponse", "Starting search"))
    infeasible = "INFEASIBLE" in txt
    sym = None
    for line in txt.splitlines():
        if "[Symmetry]" in line or "symmetry" in line.lower() and "gen" in line.lower():
            sym = line.strip()[:160]
            break
    return {"log": str(log.relative_to(REPO_ROOT)), "reached_search": reached,
            "infeasible": infeasible, "symmetry": sym}


def drive(minutes, workers, only):
    progress = REPO_ROOT / "scripts" / "_e2e_real_progress.txt"
    readout = REPO_ROOT / "scripts" / "e2e_real_readout.md"
    cap_s = minutes * 60.0
    rows = []
    plog = []

    def flush_progress(extra=""):
        progress.write_text("\n".join(plog + ([extra] if extra else [])), encoding="utf-8")

    selected = RUNS if not only else [RUNS[only - 1]]
    plog.append(f"# REAL-2026 e2e progression — {len(selected)} run(s), {minutes} min cap each, workers={workers}")
    plog.append(f"# started {datetime.now().isoformat(timespec='seconds')}")
    flush_progress()

    for idx, (label, groups, exclude) in enumerate(selected, 1):
        run_id = f"real_{label}"
        child_log = REPO_ROOT / "logs" / f"e2e_real_{label}.stdout.log"
        child_log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__)), "--run-single",
               "--groups", *groups, "--workers", str(workers), "--run-id", run_id]
        if exclude:
            cmd += ["--exclude", *exclude]
        line = f"[{idx}/{len(selected)}] {label}: groups={groups} exclude={exclude} -> launching"
        plog.append(f"{datetime.now().strftime('%H:%M:%S')} {line}")
        flush_progress()

        t0 = time.monotonic()
        with open(child_log, "w", encoding="utf-8") as fh:
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=fh,
                                    stderr=subprocess.STDOUT)
            killed = False
            try:
                proc.wait(timeout=cap_s)
            except subprocess.TimeoutExpired:
                killed = True
                _tree_kill(proc)
        elapsed = time.monotonic() - t0
        rc = proc.poll()
        sig = _scan_log(run_id)
        outcome = ("killed@cap (survived to cap)" if killed
                   else ("INFEASIBLE/early-exit" if sig["infeasible"] else f"exited rc={rc}"))
        row = {"label": label, "groups": groups, "exclude": exclude,
               "elapsed_s": round(elapsed), "killed_at_cap": killed, "rc": rc,
               "reached_search": sig["reached_search"], "infeasible": sig["infeasible"],
               "symmetry": sig["symmetry"], "child_stdout": str(child_log.relative_to(REPO_ROOT)),
               "solver_log": sig["log"], "outcome": outcome}
        rows.append(row)
        plog.append(f"{datetime.now().strftime('%H:%M:%S')}   -> {outcome}; "
                    f"elapsed={elapsed:.0f}s reached_search={sig['reached_search']} "
                    f"infeasible={sig['infeasible']} log={sig['log']}")
        flush_progress()

    # write the readout
    lines = ["# REAL-2026-config e2e progression — readout", "",
             f"Generated {datetime.now().isoformat(timespec='seconds')} · "
             f"year={REAL_YEAR} · workers={workers} · cap={minutes} min/run · raw (no slack, week-1 unfixed)",
             "",
             "| # | run | groups | exclude | elapsed | reached search | infeasible | outcome |",
             "|---|-----|--------|---------|---------|----------------|-----------|---------|"]
    for i, r in enumerate(rows, 1):
        lines.append(f"| {i} | {r['label']} | {','.join(r['groups'])} | "
                     f"{','.join(r['exclude']) or '—'} | {r['elapsed_s']}s | "
                     f"{r['reached_search']} | {r['infeasible']} | {r['outcome']} |")
    lines += ["", "## Per-run logs"]
    for r in rows:
        lines.append(f"- **{r['label']}**: solver log `{r['solver_log']}`, "
                     f"child stdout `{r['child_stdout']}`"
                     + (f", symmetry: `{r['symmetry']}`" if r['symmetry'] else ""))
    readout.write_text("\n".join(lines), encoding="utf-8")
    plog.append(f"{datetime.now().strftime('%H:%M:%S')} DONE — readout: {readout.relative_to(REPO_ROOT)}")
    flush_progress()
    print(f"[e2e-real] progression complete — readout {readout}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-single", action="store_true", help="(internal) one solve")
    ap.add_argument("--groups", nargs="+", default=["core"])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--minutes", type=float, default=5.0, help="kill cap per run (driver)")
    ap.add_argument("--only", type=int, default=0, help="run only the Nth profile (1-5)")
    args = ap.parse_args(argv)

    if args.run_single:
        rid = args.run_id or f"real_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_single(args.groups, args.exclude, args.workers, rid)
    else:
        drive(args.minutes, args.workers, args.only)


if __name__ == "__main__":
    main()
