#!/usr/bin/env python
"""spec-034 — batched green-suite runner with honest coverage.

WHY BATCHES: running the whole `pytest tests/` in one process segfaults on
Windows/ortools (documented; repeated CP-SAT model builds in a single process
eventually crash the interpreter). Each batch runs in a FRESH subprocess so
ortools state is reset between batches, dodging the segfault. Coverage is
collected in `--parallel-mode` (each subprocess writes its own `.coverage.*`
data file) and then `coverage combine`d into one honest total.

USAGE (from the repo root, with the project venv's python):
    python scripts/run_green_suite.py            # full batched suite + coverage
    python scripts/run_green_suite.py --no-cov   # batches only, skip coverage
    python scripts/run_green_suite.py --list      # print the batch plan and exit

EXIT CODE: non-zero if ANY batch had a test failure/error (so CI can gate on it).
The DoD-2 floor (>=85%) is reported against the four surfaces that matter:
    constraints/atoms/*, constraints/registry.py, constraints/stages.py,
    analytics/tester.py
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(REPO_ROOT, 'tests')

# Surfaces the DoD-2 coverage floor is measured against.
FLOOR_INCLUDE = [
    'constraints/atoms/*',
    'constraints/registry.py',
    'constraints/stages.py',
    'analytics/tester.py',
]
FLOOR_PCT = 85.0

# How many top-level test files to run per subprocess. Small batches keep each
# process's ortools build count low (segfault safety) without spawning one
# process per file.
CHUNK = 8


def build_batches() -> list[list[str]]:
    """Return the ordered list of batches; each batch is a list of pytest targets
    (paths relative to the repo root)."""
    batches: list[list[str]] = []
    # Batch 0: the whole atoms package (cheap, fast — 250 tests in ~22s).
    batches.append(['tests/atoms'])
    # Remaining: every top-level tests/test_*.py, chunked.
    top = sorted(
        os.path.relpath(p, REPO_ROOT).replace(os.sep, '/')
        for p in glob.glob(os.path.join(TESTS_DIR, 'test_*.py'))
    )
    for i in range(0, len(top), CHUNK):
        batches.append(top[i:i + CHUNK])
    return batches


def run(cmd: list[str]) -> int:
    print('  $ ' + ' '.join(cmd), flush=True)
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--no-cov', action='store_true', help='skip coverage; just run batches')
    ap.add_argument('--list', action='store_true', help='print the batch plan and exit')
    args = ap.parse_args()

    batches = build_batches()

    if args.list:
        for i, b in enumerate(batches):
            print(f'batch {i}: {b}')
        return 0

    py = sys.executable
    use_cov = not args.no_cov

    if use_cov:
        run([py, '-m', 'coverage', 'erase'])

    failed_batches: list[int] = []
    for i, targets in enumerate(batches):
        print(f'\n=== batch {i}/{len(batches) - 1}: {targets} ===', flush=True)
        if use_cov:
            cmd = [py, '-m', 'coverage', 'run', '--parallel-mode', '-m', 'pytest']
        else:
            cmd = [py, '-m', 'pytest']
        cmd += targets + ['-q', '-p', 'no:cacheprovider']
        rc = run(cmd)
        # pytest exit code 5 == "no tests collected"; treat as benign for a batch.
        if rc not in (0, 5):
            failed_batches.append(i)

    print('\n' + '=' * 70)
    if use_cov:
        run([py, '-m', 'coverage', 'combine'])
        print('\n--- FULL coverage (constraints + analytics) ---', flush=True)
        run([py, '-m', 'coverage', 'report'])
        print('\n--- DoD-2 FLOOR surfaces (target >=85%) ---', flush=True)
        floor_rc = run([py, '-m', 'coverage', 'report',
                        '--include=' + ','.join(FLOOR_INCLUDE),
                        f'--fail-under={FLOOR_PCT:g}'])
        floor_ok = floor_rc == 0
    else:
        floor_ok = True

    print('\n' + '=' * 70)
    if failed_batches:
        print(f'RESULT: FAIL — batches with failures/errors: {failed_batches}')
        return 1
    if not floor_ok:
        print(f'RESULT: tests GREEN but coverage floor (<{FLOOR_PCT:g}%) NOT met on DoD-2 surfaces.')
        return 2
    print('RESULT: GREEN — all batches passed' + ('' if not use_cov else f' and floor >={FLOOR_PCT:g}% met') + '.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
