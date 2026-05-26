#!/usr/bin/env python
"""spec-035 Unit C debugging — bisect the hard `core` set to find the atom(s)
that make the forced-free season_test model INFEASIBLE during presolve.

WHY: the full `core` (and `core --exclude ClubGameSpread`) is proven INFEASIBLE
during CP-SAT's *initial constraint copy* (~2s) on the forced-free 2026 base
config — so it never reaches presolve symmetry detection or search. Infeasibility
during initial copy means a HARD constraint (or an interaction of a few) is
over-constraining the model. Soft atoms (objective terms) cannot do this, so the
culprit is among the hard atoms. This harness runs `main_simple` against
`config/season_test` with an arbitrary set of excluded constraints and a SHORT
time cap, then classifies the outcome:

    INFEASIBLE_PRESOLVE  — proven infeasible during initial copy / presolve
    REACHED_SEARCH       — got past presolve into search (feasibility-OK chunk)
    FEASIBLE/OPTIMAL     — found a solution within the cap (definitely OK)
    UNKNOWN              — cap hit during presolve (inconclusive — raise the cap)

USAGE:
    python scripts/bisect_core_feasibility.py --exclude A B C --max-time 120
    python scripts/bisect_core_feasibility.py            # excludes only ClubGameSpread

Each probe is its own process (run via the launcher pattern) so ortools heap
state doesn't accumulate across probes.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _classify(logpath: str) -> str:
    """Read the run log and classify the CP-SAT outcome."""
    with open(logpath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    low = text.lower()
    # Proven infeasible (initial copy or presolve) — the signal we're hunting.
    if 'proven infeasible' in low or 'infeasible during initial copy' in low \
            or 'status: infeasible' in low:
        return 'INFEASIBLE_PRESOLVE'
    # A real solution was found.
    if 'status: optimal' in low or 'status: feasible' in low:
        return 'FEASIBLE/OPTIMAL'
    # Search actually started (bound/solution progress lines or explicit marker).
    if ('starting search' in low or '\n#1' in text or ' #1 ' in text
            or 'objective:' in low and 'objective: na' not in low):
        return 'REACHED_SEARCH'
    return 'UNKNOWN'


def probe(exclude: list[str], max_time: int, workers: int = 10) -> str:
    """Run one feasibility probe and return the classification string."""
    from run import _resolve_group_selection
    from main_staged import main_simple
    from solver_diagnostics import SolverConfig

    # ClubGameSpread is always excluded (the spec-035 target is core - ClubGameSpread);
    # any caller-supplied names are excluded on top.
    full_exclude = sorted(set(['ClubGameSpread'] + list(exclude or [])))
    group_names, constraint_names = _resolve_group_selection(
        ['core'], full_exclude, no_symmetry_breakers=False
    )

    run_id = f"bisect_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    cfg = SolverConfig.balanced_config(max_time=max_time)
    cfg.num_workers = workers

    print(f"[bisect] excluding ({len(full_exclude)}): {full_exclude}")
    print(f"[bisect] applying {len(constraint_names)} constraints; cap={max_time}s workers={workers}")

    try:
        main_simple(
            locked_keys=None, locked_weeks=set(), solver_config=cfg,
            exclude_constraints=full_exclude, year='test', relax_config=None,
            fix_round_1=False, constraint_slack=None, hint_solution=None,
            run_id=run_id, description=f"spec-035 bisect exclude={full_exclude}",
            constraint_names=constraint_names, groups_selected=group_names,
        )
    except Exception as exc:  # noqa: BLE001 — diagnostic harness; surface, don't swallow
        print(f"[bisect] main_simple raised: {exc!r}")

    logs = sorted(glob.glob(os.path.join(REPO_ROOT, 'logs', f'solver_*_{run_id}.log')))
    if not logs:
        print('[bisect] NO LOG FOUND — cannot classify'); return 'NO_LOG'
    verdict = _classify(logs[-1])
    print(f"[bisect] log: {os.path.relpath(logs[-1], REPO_ROOT)}")
    print(f"[bisect] VERDICT: {verdict}")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exclude', nargs='*', default=[],
                    help='Constraint canonical names to exclude (on top of ClubGameSpread).')
    ap.add_argument('--max-time', type=int, default=120, help='Solve time cap (seconds).')
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()
    verdict = probe(args.exclude, args.max_time, args.workers)
    # Exit non-zero on infeasible so a driver can branch on it.
    sys.exit(0 if verdict in ('REACHED_SEARCH', 'FEASIBLE/OPTIMAL') else 1)


if __name__ == '__main__':
    main()
