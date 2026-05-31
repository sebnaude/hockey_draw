#!/usr/bin/env python
"""Trace which HARD atoms over-constrain the REAL 2026 config (year=2026) so the
convenor knows what to LOOSEN (apply --slack to) — this is a slack-targeting trace,
NOT a bug hunt. Mirrors scripts/bisect_core_feasibility.py but on the real production
config (the 18 forced / 69 blocked games / LOCKED_PAIRINGS), not forced-free season_test.

Driver mode (default): runs a fixed sequence of exclude-set probes, each as its own
child process (so ortools heap state doesn't accumulate), classifies each run's CP-SAT
log, and writes scripts/_realbisect_progress.txt + scripts/realconfig_infeasibility_trace.md.

Child mode (--probe): one solve, prints the verdict.

Verdicts:
  INFEASIBLE_INITIAL_COPY — proven infeasible during initial copy (~2s): a trivial
                            slack-0 contradiction (e.g. BalancedByeSpacing 2x<=1).
  INFEASIBLE_PRESOLVE     — proven infeasible during presolve/search (no feasible region).
  REACHED_SEARCH          — got past presolve into search without an infeasibility proof
                            within the cap (= removing the excluded atoms RELIEVED the
                            over-constraint; those atoms are loosening candidates).
  FEASIBLE/OPTIMAL        — found a feasible draw within the cap.
  UNKNOWN                 — cap hit mid-presolve (raise the cap).
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

REAL_YEAR = 2026

# Atom groups in `core` that can be excluded to localize the conflict.
GROUPS = {
    "home_away": ["AwayClubHomeWeekendsCount", "AwayClubPerOpponentAndAggregateHomeBalance"],
    # CoLocation is a helper-var CONSUMER of StackedWeekends; exclude together.
    "club_alignment": ["ClubVsClubStackedWeekends", "ClubVsClubStackedCoLocation"],
    "club_day": ["ClubDayParticipation", "ClubDayIntraClubMatchup", "ClubDayOpponentMatchup",
                 "ClubDaySameField", "ClubDayContiguousSlots"],
    "club_game_spread": ["ClubGameSpread"],
}

# The probe plan. Each entry: (label, groups_arg, exclude_atoms). `groups_arg` is what
# goes to --groups; exclude_atoms is the union of group atom-lists to drop.
def _ex(*group_keys):
    out = []
    for k in group_keys:
        out += GROUPS[k]
    return out


PROBES = [
    # Baseline: full core (expected INFEASIBLE — the thing we're localizing).
    ("P0_core_full",                ["core"], []),
    # Drop one group at a time from core — whichever drop REACHES_SEARCH is implicated.
    ("P1_core_no_club_alignment",   ["core"], _ex("club_alignment")),
    ("P2_core_no_club_day",         ["core"], _ex("club_day")),
    ("P3_core_no_home_away",        ["core"], _ex("home_away")),
    # The season_test minimal pair was ClubVsClubStackedWeekends x ClubDayParticipation.
    # Test the same pair-drop on real config: drop BOTH alignment+club_day.
    ("P4_core_no_align_no_clubday", ["core"], _ex("club_alignment", "club_day")),
    # bye_spacing: confirm the trivial slack-0 contradiction (initial-copy infeasible).
    ("P5_core_bye_spacing",         ["core", "bye_spacing"], []),
]


def _classify(logpath: str) -> str:
    with open(logpath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    low = text.lower()
    if "infeasible during initial copy" in low or "proven during initial copy" in low:
        return "INFEASIBLE_INITIAL_COPY"
    if "status: optimal" in low or "status: feasible" in low:
        return "FEASIBLE/OPTIMAL"
    if "proven infeasible" in low or "status: infeasible" in low:
        return "INFEASIBLE_PRESOLVE"
    if ("starting search" in low or "\n#1" in text or " #1 " in text):
        return "REACHED_SEARCH"
    return "UNKNOWN"


def run_probe(groups, exclude, max_time, workers, run_id):
    from run import _resolve_group_selection
    from main_staged import main_simple
    from solver_diagnostics import SolverConfig

    group_names, constraint_names = _resolve_group_selection(
        groups, exclude, no_symmetry_breakers=False
    )
    cfg = SolverConfig.balanced_config(max_time=max_time)
    cfg.num_workers = workers
    print(f"[realbisect] year={REAL_YEAR} groups={groups} exclude({len(exclude)})={exclude}")
    print(f"[realbisect] applying {len(constraint_names)} constraints; cap={max_time}s workers={workers}")
    try:
        main_simple(
            locked_keys=None, locked_weeks=set(), solver_config=cfg,
            exclude_constraints=exclude, year=REAL_YEAR, relax_config=None,
            fix_round_1=False, constraint_slack=None, hint_solution=None,
            run_id=run_id, description=f"realconfig bisect groups={groups} exclude={exclude}",
            constraint_names=constraint_names, groups_selected=group_names,
        )
    except Exception as exc:  # noqa: BLE001 — diagnostic; surface, don't swallow
        print(f"[realbisect] main_simple raised: {exc!r}")
    logs = sorted(glob.glob(os.path.join(REPO_ROOT, "logs", f"solver_*_{run_id}.log")))
    if not logs:
        print("[realbisect] NO LOG FOUND"); return "NO_LOG", None
    verdict = _classify(logs[-1])
    print(f"[realbisect] VERDICT: {verdict}  log={os.path.relpath(logs[-1], REPO_ROOT)}")
    return verdict, os.path.relpath(logs[-1], REPO_ROOT)


def drive(max_time, workers):
    progress = os.path.join(REPO_ROOT, "scripts", "_realbisect_progress.txt")
    trace = os.path.join(REPO_ROOT, "scripts", "realconfig_infeasibility_trace.md")
    rows = []
    plog = [f"# REAL-2026 infeasibility bisection — {len(PROBES)} probes, cap {max_time}s, workers {workers}",
            f"# started {datetime.now().isoformat(timespec='seconds')}"]
    def flush():
        with open(progress, "w", encoding="utf-8") as fh:
            fh.write("\n".join(plog))
    flush()

    for i, (label, groups, exclude) in enumerate(PROBES, 1):
        run_id = f"realbisect_{label}"
        child_log = os.path.join(REPO_ROOT, "logs", f"realbisect_{label}.stdout.log")
        cmd = [sys.executable, os.path.abspath(__file__), "--probe",
               "--groups", *groups, "--max-time", str(max_time),
               "--workers", str(workers), "--run-id", run_id]
        if exclude:
            cmd += ["--exclude", *exclude]
        plog.append(f"{datetime.now().strftime('%H:%M:%S')} [{i}/{len(PROBES)}] {label}: "
                    f"groups={groups} drop={exclude or '—'} -> running")
        flush()
        with open(child_log, "w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=fh, stderr=subprocess.STDOUT)
        # classify from the child's solver log
        logs = sorted(glob.glob(os.path.join(REPO_ROOT, "logs", f"solver_*_{run_id}.log")))
        verdict = _classify(logs[-1]) if logs else "NO_LOG"
        solver_log = os.path.relpath(logs[-1], REPO_ROOT) if logs else None
        rows.append((label, groups, exclude, verdict, solver_log))
        plog.append(f"{datetime.now().strftime('%H:%M:%S')}   -> {verdict}  (rc={rc})  log={solver_log}")
        flush()

    # write the trace doc
    lines = ["# REAL-2026-config infeasibility trace — what to loosen", "",
             f"Generated {datetime.now().isoformat(timespec='seconds')} · year={REAL_YEAR} · "
             f"workers={workers} · cap={max_time}s/probe · raw (no slack).", "",
             "Goal: localize which HARD `core` atoms over-constrain the **real** 2026 config "
             "(18 forced / 69 blocked games / LOCKED_PAIRINGS) so they can be given `--slack`. "
             "Not a bug hunt — a slack-targeting trace.", "",
             "| probe | groups | dropped atoms | verdict |",
             "|-------|--------|---------------|---------|"]
    for label, groups, exclude, verdict, _ in rows:
        lines.append(f"| {label} | {','.join(groups)} | {', '.join(exclude) or '—'} | **{verdict}** |")
    lines += ["", "## Reading", "",
              "- A probe that flips to **REACHED_SEARCH / FEASIBLE** when an atom group is dropped "
              "identifies that group as an over-constraint → a loosening (slack) candidate.",
              "- **INFEASIBLE_INITIAL_COPY** = a trivial slack-0 contradiction (loosen that atom's base slack).",
              "- **INFEASIBLE_PRESOLVE** with the group still present = the conflict is elsewhere / an interaction.",
              "", "## Per-probe solver logs"]
    for label, _, _, verdict, solver_log in rows:
        lines.append(f"- **{label}** → {verdict} · `{solver_log}`")
    with open(trace, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    plog.append(f"{datetime.now().strftime('%H:%M:%S')} DONE — trace: scripts/realconfig_infeasibility_trace.md")
    flush()
    print(f"[realbisect] done — {trace}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="(internal) run ONE solve")
    ap.add_argument("--groups", nargs="+", default=["core"])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--max-time", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()
    if args.probe:
        rid = args.run_id or f"realbisect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_probe(args.groups, args.exclude, args.max_time, args.workers, rid)
    else:
        drive(args.max_time, args.workers)


if __name__ == "__main__":
    main()
