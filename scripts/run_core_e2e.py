#!/usr/bin/env python
# scripts/run_core_e2e.py
"""spec-035 Unit A — raw `--core` e2e launcher on the forced-free test config.

This is the single, auditable launch path for the spec-035 ultimate e2e solve.
It drives the SAME solve entry point `run.py generate` uses for a no-flag single
solve (`main_staged.main_simple`) against the constraint-free `season_test`
config, with the EXACT raw-core flag profile fixed in one place:

    groups       = ['core']     (spec-032 also unions the always-on
                                 `symmetry_breakers` trio — NOT suppressed here)
    workers      = 10           (and only 10)
    fix_round_1  = False        (week 1 is NOT fixed — a raw run)
    locked_weeks = none         (raw)
    forced_games = []           (inherent to season_test)
    exclude      = []            (default; the ONLY permitted per-run delta)

The launcher takes an optional `exclude` list threaded straight to the solve's
`--exclude` equivalent so the IDENTICAL profile produces both the DoD-2 run
(no exclude) and the DoD-2b run (`exclude=['ClubGameSpread']`) with no other
difference. The resolved flag set is recorded to a JSON sidecar under `logs/`
and logged via the project's standard `solver` logger so the profile is
auditable.

Data is loaded for the DoD-1 assertions via the documented programmatic path
``from config.season_test import get_season_data`` (Option (b) in spec-035 DoD-1
— avoids widening `load_season_data`'s `year: int` signature). The solve itself
is driven through ``main_simple(year='test', ...)`` which rebuilds the identical
data dict from the same `season_test` config (verified equivalent at build time:
48 teams, 787 timeslots, identical field availability, `forced_games == []`).

Usage (programmatic — preferred; Unit C calls this in the background and kills
at 30 min):

    from scripts.run_core_e2e import build_run_config, main
    cfg = build_run_config(exclude=['ClubGameSpread'])   # resolve + inspect
    main(exclude=['ClubGameSpread'])                      # actually solve

Usage (CLI):

    python scripts/run_core_e2e.py                        # full core (DoD-2)
    python scripts/run_core_e2e.py --exclude ClubGameSpread  # DoD-2b
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Compute repo root from __file__ so this script works from any cwd / worktree.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The project's standard logger (see solver_diagnostics.setup_logging). Reused
# here so launch/presolve-fail paths are logged at INFO, never printed/swallowed.
logger = logging.getLogger("solver")

# ---------------------------------------------------------------------------
# Fixed raw-core profile (DoD-2). Defined as module constants so there is a
# single source of truth and the tests can assert against them field-by-field.
# ---------------------------------------------------------------------------
CORE_GROUPS = ['core']
CORE_WORKERS = 10
CORE_FIX_ROUND_1 = False
CORE_LOCKED_WEEKS: list = []      # raw run — no locks
CORE_FORCED_GAMES: list = []      # inherent to season_test
# season_test is selected programmatically; `--year test` cannot be used (the
# int-typed `--year` cannot select it — see season_test.py docstring). The
# solve entry point still needs a `year` sentinel; 'test' resolves the config
# via load_season_data('test') -> config.season_test.
SEASON_YEAR_SENTINEL = 'test'


def build_run_config(exclude: Optional[list] = None,
                     workers: Optional[int] = None,
                     groups: Optional[list] = None) -> dict:
    """Resolve the raw-core flag profile into an auditable dict.

    `exclude` (default `[]`) is the ONLY permitted delta between the two
    spec-035 Unit-C runs. Everything else is fixed by the module constants above.

    `groups` (default `['core']` = `CORE_GROUPS`) is the spec-035 follow-on
    (Units D/E) lever: the same raw profile can be launched as `core`,
    `core,bye_spacing`, or `core,spacing` so the marginal liveness/symmetry
    effect of the spacing-family groups (peeled out of `core` by spec-032/033)
    can be measured. `symmetry_breakers` are still unioned in (spec-032). The
    group set is recorded in the sidecar so it is the only delta between the
    spacing-family runs and the full-core run.

    `workers` defaults to the DoD-2 value (`CORE_WORKERS` = 10). It is exposed
    as an override solely for the convenor-authorised DoD-6 resourcing path
    (2026-05-30): on a RAM-constrained box a 10-worker 110k-var solve OOMs
    before the 30-min liveness bar, so the convenor authorised dropping to 8
    workers. Worker count does NOT affect the model or its presolve `[Symmetry]`
    readout (symmetry is a property of the model, not the search threads), so
    the cross-run symmetry comparison stays valid; the resolved profile records
    the actual count used so any deviation from 10 is auditable.

    Returns a dict capturing the exact resolved profile — the same dict is
    written to the run's JSON sidecar (DoD-2: "records the exact resolved flag
    set ... so the profile is auditable") and asserted by the Unit A tests.
    """
    exclude_list = list(exclude) if exclude else []
    return {
        'groups': list(groups) if groups else list(CORE_GROUPS),
        'workers': int(workers) if workers else CORE_WORKERS,
        'fix_round_1': CORE_FIX_ROUND_1,
        'locked_weeks': list(CORE_LOCKED_WEEKS),
        'forced_games': list(CORE_FORCED_GAMES),
        'exclude': exclude_list,
        'no_symmetry_breakers': False,   # symmetry_breakers stay ON (spec-032)
        'year': SEASON_YEAR_SENTINEL,
        'config_module': 'config.season_test',
        'data_loader': 'config.season_test.get_season_data',
        'solve_entry_point': 'main_staged.main_simple',
        'mode': 'single-solve (no mode flag — spec-036 DEFAULT)',
    }


def _record_profile(cfg: dict, run_id: str) -> Path:
    """Write the resolved flag set to a JSON sidecar under logs/ and log it.

    Returns the path written. The sidecar makes the profile auditable
    independently of the solver log and lets Unit C prove the two runs differ
    only by `exclude`.
    """
    log_dir = REPO_ROOT / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    sidecar = log_dir / f"core_e2e_profile_{run_id}.json"
    payload = {
        'recorded_at': datetime.now().isoformat(timespec='seconds'),
        'run_id': run_id,
        'profile': cfg,
        # Unit C: the solve log (with the CP-SAT [Symmetry] block) is created by
        # main_simple under this glob. Exactly one file matches per run_id (the
        # launcher no longer double-inits logging — review M1).
        'solve_log_glob': f"logs/solver_*_{run_id}.log",
    }
    sidecar.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    logger.info("[spec-035] resolved raw-core profile (%s): %s", run_id, cfg)
    return sidecar


def _assert_season_test_is_forced_free() -> None:
    """DoD-1 guard: confirm season_test is forced-free before launching.

    Loads data via the documented Option-(b) path (`get_season_data()`), and
    fails loudly (logged at INFO via the project logger, then ValueError) if the
    config has drifted to non-empty forced games. The team-set / field-set
    reconciliation against season_2026 is covered by the Unit A test suite; here
    we guard the single inexpensive invariant that would silently corrupt the
    run (a forced game would make it not a *raw* run).
    """
    from config.season_test import get_season_data
    data = get_season_data()
    forced = data.get('forced_games', None)
    if forced:
        msg = (f"season_test has DRIFTED: forced_games is non-empty ({forced!r}); "
               f"the raw e2e run requires forced_games == []. Fix config/season_test.py.")
        logger.info("[spec-035] %s", msg)
        raise ValueError(msg)
    logger.info("[spec-035] season_test forced-free check passed (forced_games == []).")


def main(exclude: Optional[list] = None, run_id: Optional[str] = None,
         workers: Optional[int] = None, groups: Optional[list] = None):
    """Launch the raw-core single solve on the forced-free test config.

    This starts a REAL `main_simple` solve (long-running). Unit C runs this in
    the background and kills it at the 30-minute mark; the kill logic is NOT
    here by design (Unit C owns it). On any presolve/launch failure the path is
    logged at INFO via the project logger and re-raised — never swallowed.
    """
    from solver_diagnostics import SolverConfig

    resolved_run_id = run_id or f"core_e2e_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    # NOTE (spec-035 review M1): do NOT call setup_logging() here. main_simple()
    # calls it with this same run_id, and setup_logging timestamps + clears
    # handlers on every call — a second init here would create a SECOND log file,
    # leaving the solve output (incl. the CP-SAT [Symmetry] block Unit C parses)
    # in main_simple's file while the launcher's file holds only these preamble
    # lines. Letting main_simple own logging yields exactly ONE
    # `logs/solver_*_<run_id>.log` per run, which Unit C globs deterministically.
    # The profile sidecar below is the authoritative auditable record regardless.

    cfg = build_run_config(exclude=exclude, workers=workers, groups=groups)
    sidecar = _record_profile(cfg, resolved_run_id)
    logger.info("[spec-035] raw-core e2e launch; profile sidecar: %s", sidecar)
    print(f"[spec-035] raw-core e2e launch (run_id={resolved_run_id})")
    print(f"  Resolved profile: {cfg}")
    print(f"  Profile sidecar:  {sidecar}")

    # DoD-1: refuse to launch if season_test drifted away from forced-free.
    _assert_season_test_is_forced_free()

    # Build the solver config: workers=10 (and only 10), matching run.py's
    # custom-worker path (balanced base, num_workers overridden).
    solver_config = SolverConfig.balanced_config()
    solver_config.num_workers = cfg['workers']

    # spec-023/spec-032: resolve --groups core (+ always-on symmetry_breakers)
    # minus --exclude into the deduped canonical constraint set, exactly as
    # run.py does. This is the set main_simple filters each stage against.
    from run import _resolve_group_selection
    group_names, constraint_names = _resolve_group_selection(
        cfg['groups'], cfg['exclude'], no_symmetry_breakers=cfg['no_symmetry_breakers']
    )
    logger.info("[spec-035] resolved %d constraint(s) for groups=%s exclude=%s",
                len(constraint_names), group_names, cfg['exclude'])

    # Drive the SAME entry point run.py's no-flag single solve uses. year is the
    # 'test' sentinel so main_simple -> load_data('test') rebuilds the identical
    # season_test data dict (forced-free, 2026 base teams/fields).
    from main_staged import main_simple
    try:
        solution, data = main_simple(
            locked_keys=None,
            locked_weeks=set(cfg['locked_weeks']),    # empty — raw run
            solver_config=solver_config,
            exclude_constraints=cfg['exclude'],
            year=cfg['year'],
            relax_config=None,
            fix_round_1=cfg['fix_round_1'],           # False — week 1 not fixed
            constraint_slack=None,
            hint_solution=None,
            run_id=resolved_run_id,
            description=f"spec-035 raw-core e2e (exclude={cfg['exclude']})",
            provenance={'spec': 'spec-035', 'profile_sidecar': str(sidecar)},
            constraint_names=constraint_names,
            groups_selected=group_names,
        )
    except Exception as exc:
        # DoD-8 / AST sweep: presolve/launch failures are LOGGED at INFO via the
        # project logger, never silently swallowed, then re-raised for Unit C's
        # 30-min wrapper to observe.
        logger.info("[spec-035] raw-core e2e launch FAILED before/while solving: %s", exc)
        raise

    if solution:
        logger.info("[spec-035] raw-core e2e produced a solution (bonus — not required).")
    else:
        logger.info("[spec-035] raw-core e2e ended without a feasible solution "
                    "(acceptable — liveness, not feasibility, is the goal).")
    return solution, cfg


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="spec-035 raw-core e2e launcher (forced-free test config).")
    parser.add_argument(
        '--exclude', nargs='+', metavar='CONSTRAINT', default=None,
        help="Canonical atom name(s) to exclude (e.g. ClubGameSpread for DoD-2b). "
             "The ONLY permitted delta from the raw-core profile.")
    parser.add_argument(
        '--run-id', type=str, default=None,
        help="Optional run id (defaults to a timestamp).")
    parser.add_argument(
        '--workers', type=int, default=None,
        help="Worker count override (default 10 = DoD-2). Convenor-authorised "
             "DoD-6 resourcing lever only (e.g. 8 on a RAM-constrained box); "
             "does not affect the model or its symmetry readout.")
    parser.add_argument(
        '--groups', type=str, default=None,
        help="Comma-separated constraint group set (default 'core'). spec-035 "
             "follow-on (Units D/E): e.g. 'core,bye_spacing' or 'core,spacing'. "
             "symmetry_breakers are always unioned in.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    groups = [g.strip() for g in args.groups.split(',')] if args.groups else None
    main(exclude=args.exclude, run_id=args.run_id, workers=args.workers, groups=groups)
