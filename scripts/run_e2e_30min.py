#!/usr/bin/env python
# scripts/run_e2e_30min.py
"""spec-035 Unit C — 30-minute timeout wrapper around the raw-core e2e launcher.

DoD-4/DoD-7: each raw e2e run must "get through presolve and survive >=30 minutes
of search", then be "killed at the 30-minute mark regardless of whether a feasible
solution was found", with a CLEAN process-tree kill on Windows (no orphaned ortools
workers). The kill mechanism is, by spec-035 design, owned by Unit C (NOT the
launcher `run_core_e2e.py`, which deliberately has no cap). This wrapper is that
mechanism.

It runs `scripts/run_core_e2e.py` as a child process, waits up to `--minutes`
(default 30), and on timeout terminates the whole process tree
(`taskkill /F /T /PID` on Windows; `proc.kill()` elsewhere). The launcher writes
the CP-SAT log (incl. the `[Symmetry]` block — Unit B) and the profile sidecar
under `logs/`, both of which survive the kill because they are flushed as the
solve streams. Any args after `--` are forwarded verbatim to the launcher, so the
same wrapper drives:

    Run 1: python scripts/run_e2e_30min.py --run-id run1_noCGS -- --exclude ClubGameSpread
    Run 2: python scripts/run_e2e_30min.py --run-id run2_fullcore
    (Units D/E, once the launcher learns --groups:)
    Run 3: python scripts/run_e2e_30min.py --run-id run3_bye   -- --groups core,bye_spacing
    Run 4: python scripts/run_e2e_30min.py --run-id run4_spc   -- --groups core,spacing

Exit code 0 = ran to the cap and was killed at the mark (the expected "survived
30 min" outcome) OR the solve finished early with a solution; non-zero = the child
exited early with an error (e.g. presolve infeasibility) before the cap.
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
LAUNCHER = REPO_ROOT / "scripts" / "run_core_e2e.py"


def _tree_kill(proc: subprocess.Popen) -> None:
    """Kill the child process and every descendant (ortools workers).

    `taskkill /F /T` alone proved unreliable here (spec-035 Run 2): the venv
    python re-execs the real interpreter, and if the intermediate parent exits
    first the actual solver child can be reparented and survive a tree-kill from
    the original PID — leaving a multi-GB ortools orphan (`child_rc=None`). So we
    enumerate the live descendants via psutil and kill each explicitly, then fall
    back to taskkill / proc.kill, and finally VERIFY the tree is gone.
    """
    if proc.poll() is not None:
        return
    # 1. psutil-based recursive kill (reliable for reparented children).
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        victims = parent.children(recursive=True) + [parent]
        for v in victims:
            try:
                v.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(victims, timeout=20)
    except Exception:
        # 2. Fallback: OS tree-kill.
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, text=True)
        else:
            proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    # 3. Verify no orphan survived (best-effort; log if one did).
    try:
        import psutil
        if psutil.pid_exists(proc.pid) and proc.poll() is None:
            print(f"[e2e-30min] WARNING: PID {proc.pid} still alive after tree-kill")
    except Exception:
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="spec-035 Unit C 30-min e2e timeout wrapper.")
    ap.add_argument("--minutes", type=float, default=30.0,
                    help="Wall-clock cap in minutes (default 30 — the DoD-4 bar).")
    ap.add_argument("--run-id", type=str, default=None,
                    help="Run id forwarded to the launcher (also names the log glob).")
    ap.add_argument("forward", nargs=argparse.REMAINDER,
                    help="Args after `--` forwarded verbatim to run_core_e2e.py.")
    args = ap.parse_args(argv)

    run_id = args.run_id or f"core_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    forward = [a for a in args.forward if a != "--"]

    cmd = [sys.executable, str(LAUNCHER), "--run-id", run_id] + forward
    cap_s = args.minutes * 60.0

    print(f"[e2e-30min] run_id={run_id}")
    print(f"[e2e-30min] cap={args.minutes} min ({cap_s:.0f}s)")
    print(f"[e2e-30min] launching: {' '.join(cmd)}")
    started = time.monotonic()
    started_wall = datetime.now().isoformat(timespec="seconds")

    # cwd = REPO_ROOT so the launcher resolves season_test/logs against this worktree.
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
    killed_at_cap = False
    try:
        proc.wait(timeout=cap_s)
    except subprocess.TimeoutExpired:
        killed_at_cap = True
        print(f"[e2e-30min] cap reached ({args.minutes} min) — process-tree-killing PID {proc.pid}")
        _tree_kill(proc)

    elapsed = time.monotonic() - started
    rc = proc.poll()
    print(f"[e2e-30min] run_id={run_id} started={started_wall} "
          f"elapsed={elapsed:.0f}s killed_at_cap={killed_at_cap} child_rc={rc}")
    print(f"[e2e-30min] solve log glob: logs/solver_*_{run_id}.log")
    print(f"[e2e-30min] profile sidecar: logs/core_e2e_profile_{run_id}.json")

    # "Survived the cap and was killed at the mark" is the expected success.
    if killed_at_cap:
        return 0
    # Finished before the cap: 0 if the child exited cleanly, else its code.
    return 0 if rc == 0 else (rc or 1)


if __name__ == "__main__":
    sys.exit(main())
