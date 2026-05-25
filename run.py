#!/usr/bin/env python
# run.py
"""
Hockey Draw Scheduling System - Main Entry Point.

Usage:
    python run.py generate                  # Generate draw for default season
    python run.py generate --year 2026     # Generate draw for specific season
    python run.py generate --resume run_1   # Resume from checkpoint
    
    python run.py test draws/draw.json      # Test existing draw for violations
    python run.py analyze draws/draw.json   # Generate analytics report
    
    python run.py swap draw.json G001 G002  # Test swapping two games
    python run.py report draw.json --club Maitland  # Generate club report
    python run.py report draw.json --all    # Generate all reports
    
    python run.py import partial.xlsx 5     # Import first 5 weeks from Excel
    python run.py list-constraints          # List all constraints
    
    python run.py preseason --year 2026     # Generate pre-season config report

See docs/README.md for full documentation.
"""

import sys
import argparse
from collections import defaultdict
from pathlib import Path

# Add refactored directory to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description='Hockey Draw Scheduling System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py generate                 Generate draw with default settings
  python run.py generate --year 2026     Generate 2026 season draw
  python run.py generate --resume run_1  Resume from checkpoint
  
  python run.py test draw.json           Test draw for constraint violations
  python run.py analyze draw.json        Generate analytics report
  
  python run.py swap draw.json G001 G002 Test swapping two games
  python run.py report draw.json --club Maitland
  python run.py report draw.json --all   Generate all club/grade reports
  
  python run.py import partial.xlsx 5    Import first 5 weeks, save as JSON
  python run.py list-constraints         Show all available constraints
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate a new draw')
    gen_parser.add_argument('--year', type=int, required=True,
                            help='Season year (e.g., 2025, 2026). Required.')
    gen_parser.add_argument('--resume', nargs='*', metavar=('RUN_ID', 'STAGE'),
                            help='Resume from checkpoint')
    gen_parser.add_argument('--run-id', type=str,
                            help='Specify run ID for checkpoints')
    gen_parser.add_argument('--description', type=str, default='',
                            help='Short description saved to draw metadata (e.g. "optimising weeks 2-4")')
    gen_parser.add_argument('--locked', type=str, metavar='SOURCE',
                            help='Source for locked games: draw JSON (.json), '
                                 'checkpoint directory (e.g. checkpoints/latest), '
                                 'or solution pickle (.pkl). Use with --lock-weeks.')
    gen_parser.add_argument('--lock-weeks', type=str, default='',
                            help='Comma-separated list of weeks to lock (e.g. 1,2,3 or 1,5,7). Use with --locked')
    gen_parser.add_argument('--repair-locked', action='store_true',
                            help='Auto-repair locked keys that have stale round_no, day_slot, etc. '
                                 'Matches on (date, field, time) and fixes ancillary fields.')
    gen_parser.add_argument('--hint', type=str, metavar='FILE',
                            help='Path to a prior solution to use as solver hints (speeds up search). '
                                 'Accepts draw JSON (.json) or checkpoint pickle (.pkl). '
                                 'Hints are suggestions, not locks - solver can deviate.')
    gen_parser.add_argument('--workers', type=int, default=None,
                            help='Number of solver workers (default: auto based on memory)')
    gen_parser.add_argument('--low-memory', action='store_true',
                            help='Use low-memory solver configuration (4 workers, minimal linearization)')
    gen_parser.add_argument('--minimal-memory', action='store_true',
                            help='Use minimal-memory solver configuration (2 workers, no probing) - very slow but stable')
    gen_parser.add_argument('--high-performance', action='store_true',
                            help='Use high-performance config (all cores, full linearization)')
    gen_parser.add_argument('--exclude', nargs='+', metavar='CONSTRAINT',
                            help='Exclude specific atoms from the solve by canonical name '
                                 '(e.g. ClubGameSpread, EnsureBestTimeslotChoices).')
    gen_parser.add_argument('--stages', nargs='+', metavar='STAGE',
                            help='Run only specific SOLVER_STAGES entries by name '
                                 '(e.g. --stages critical_feasibility soft_optimisation).')
    gen_parser.add_argument('--staged', action='store_true',
                            help='DEFAULT_STAGES incremental staged solve. Applies the '
                                 'full constraint set one stage at a time, solving after '
                                 'each stage and carrying the prior solution as a HINT. '
                                 '(Default with no mode flag is a single full solve.)')
    gen_parser.add_argument('--severity', action='store_true',
                            help='Severity-grouped staged solve. Runs 5 stages by severity '
                                 'level: Level 1 (CRITICAL) -> Level 2 (HIGH) -> Level 3 (MEDIUM) '
                                 '-> Level 4 (LOW) -> Level 5 (VERY LOW). Each stage uses the '
                                 'prior solution as a HINT.')
    gen_parser.add_argument('--relax', action='store_true',
                            help='Enable severity-based constraint relaxation. If infeasible, '
                                 'automatically identifies problem severity group and relaxes slack variables.')
    gen_parser.add_argument('--relax-timeout', type=float, default=30.0,
                            help='Timeout per feasibility test during relaxation (default: 30 seconds)')
    gen_parser.add_argument('--fix-round-1', action='store_true',
                            help='Apply Round 1 symmetry breaking. Fixes which team pairings play '
                                 'in Round 1 using the circle method. This dramatically reduces '
                                 'search space by eliminating equivalent schedule orderings.')
    gen_parser.add_argument('--slack', type=int, default=None, metavar='N',
                            help='Relax constraints by adding N to their limits. '
                                 'Applies to all slack-aware constraints: '
                                 'EqualMatchUpSpacing, ClubGameSpread, BalancedByeSpacing, etc.')
    gen_parser.add_argument('--stages-config', type=str, metavar='FILE',
                            help='Path to a JSON file with a custom SOLVER_STAGES list. '
                                 'Replaces the in-config solver_stages.')
    gen_parser.add_argument('--stage-only', type=str, metavar='NAME',
                            help='Run only the named SOLVER_STAGES entry.')
    gen_parser.add_argument('--skip-stage', action='append', default=[], metavar='NAME',
                            help='Skip a SOLVER_STAGES entry. May be passed multiple times.')
    gen_parser.add_argument('--list-stages', action='store_true',
                            help='Print the configured SOLVER_STAGES and exit.')
    gen_parser.add_argument('--groups', nargs='+', metavar='NAME', default=None,
                            help='spec-023: apply the deduped UNION of these named constraint '
                                 'groups (e.g. --groups core soft, --groups severity_1). '
                                 'Resolves via the registry, minus any --exclude. No --groups '
                                 'means the "default" group (every production constraint). '
                                 'Use --list-groups to see available group names.')
    gen_parser.add_argument('--list-groups', action='store_true',
                            help='Print all known constraint group names (explicit tags + '
                                 'derived severity_N/default) and exit.')
    gen_parser.add_argument('--no-symmetry-breakers', action='store_true',
                            help='spec-032: suppress the always-on symmetry-breaker bundle '
                                 '(NIHCFillWFBeforeEF, NIHCFillEFBeforeSF, SoftLexMatchupOrdering). '
                                 'By default these tie-breakers are applied in EVERY solve '
                                 'regardless of --groups; this flag drops them from both the '
                                 '--groups path and the plain (no --groups) path.')
    gen_parser.add_argument('--regen-from', type=str, metavar='SOURCE',
                            help='Source draw JSON to freeze from (e.g. draws/2026/current.json). '
                                 'Presence of this flag enables regeneration mode: games outside '
                                 'the regen scope (--regen-grades / --regen-weeks) are pinned to '
                                 'their existing dates via LOCKED_PAIRINGS, while games inside the '
                                 'scope are solved fresh. Played weeks supplied via --lock-weeks '
                                 'are hard-locked (exact game key) and are never pinned — '
                                 '--lock-weeks and --regen-weeks must not overlap.')
    gen_parser.add_argument('--regen-grades', nargs='+', metavar='GRADE',
                            help='Grades to REGENERATE (re-solve freely). Games in grades NOT '
                                 'listed here are frozen (pinned to their date). Omit to leave '
                                 'all grades free along the grade axis. Union semantics: a game '
                                 'is FREE if its grade is in --regen-grades OR its week is in '
                                 '--regen-weeks (either axis frees it).')
    gen_parser.add_argument('--regen-weeks', type=str, metavar='SPEC',
                            help='Weeks to REGENERATE (re-solve freely), as a range or list '
                                 '(e.g. "10-22", "10,12,14", "10-12,15"). Future weeks NOT '
                                 'listed here are frozen (pinned to their date). Omit to leave '
                                 'all weeks free along the week axis. Union semantics: a game '
                                 'is FREE if its grade is in --regen-grades OR its week is in '
                                 '--regen-weeks. Must not overlap with --lock-weeks (FATAL).')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test draw for violations')
    test_parser.add_argument('draw_file', help='Path to draw JSON file')
    test_parser.add_argument('--year', type=int, required=True,
                            help='Season year (e.g., 2025, 2026). Required.')
    
    # Analyze command  
    analyze_parser = subparsers.add_parser('analyze', help='Generate analytics')
    analyze_parser.add_argument('draw_file', help='Path to draw JSON file')
    analyze_parser.add_argument('--year', type=int, required=True,
                               help='Season year (e.g., 2025, 2026). Required.')
    analyze_parser.add_argument('--output', '-o', type=str, help='Output file path')
    
    # Swap command
    swap_parser = subparsers.add_parser('swap', help='Test game swap')
    swap_parser.add_argument('draw_file', help='Path to draw JSON file')
    swap_parser.add_argument('game1', help='First game ID (e.g., G00001)')
    swap_parser.add_argument('game2', help='Second game ID')
    swap_parser.add_argument('--year', type=int, required=True,
                            help='Season year (e.g., 2025, 2026). Required.')
    swap_parser.add_argument('--save', type=str, help='Save modified draw to file')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate stakeholder reports')
    report_parser.add_argument('draw_file', help='Path to draw JSON file')
    report_parser.add_argument('--club', type=str, help='Generate report for specific club')
    report_parser.add_argument('--team', type=str, help='Generate report for specific team')
    report_parser.add_argument('--grade', type=str, help='Generate report for specific grade')
    report_parser.add_argument('--compliance', action='store_true', help='Generate compliance certificate')
    report_parser.add_argument('--all', action='store_true', help='Generate all reports')
    report_parser.add_argument('--html', action='store_true', help='Generate HTML report')
    report_parser.add_argument('--output', '-o', type=str, default='reports', help='Output directory')
    report_parser.add_argument('--year', type=int, required=True,
                              help='Season year (e.g., 2025, 2026). Required.')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import draw from Excel')
    import_parser.add_argument('excel_file', help='Path to Excel file')
    import_parser.add_argument('--lock-weeks', type=str,
                                help='Comma-separated list of weeks to lock (e.g. 1,2,3)')
    import_parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    import_parser.add_argument('--year', type=int, required=True,
                              help='Season year (e.g., 2025, 2026). Required.')
    
    # List constraints command
    list_parser = subparsers.add_parser('list-constraints', help='List all constraints')
    
    # Preseason report command
    preseason_parser = subparsers.add_parser('preseason', help='Generate pre-season configuration report')
    preseason_parser.add_argument('--year', type=int, required=True,
                                  help='Season year (e.g., 2025, 2026). Required.')
    preseason_parser.add_argument('--output', '-o', type=str,
                                  help='Output file path (optional)')
    
    # Diagnose command - find infeasibility and resolve
    diagnose_parser = subparsers.add_parser('diagnose', 
        help='Find blocking constraints and resolve infeasibility')
    diagnose_parser.add_argument('--year', type=int, required=True,
                                 help='Season year (e.g., 2025, 2026). Required.')
    diagnose_parser.add_argument('--stage', type=str, default='critical_feasibility',
                                 help='SOLVER_STAGES name to analyze (default: critical_feasibility). '
                                      'Severity-derived names (severity_1..severity_5) also work.')
    diagnose_parser.add_argument('--timeout', type=float, default=5.0,
                                 help='Timeout per feasibility test in seconds (default: 5)')
    diagnose_parser.add_argument('--resolve', action='store_true',
                                 help='Attempt iterative relaxation to find feasible solution')
    diagnose_parser.add_argument('--max-iterations', type=int, default=10,
                                 help='Max relaxation iterations (default: 10)')
    
    # Validate command - check draw keys against current timeslot data
    validate_parser = subparsers.add_parser('validate',
        help='Validate draw game keys against current season timeslot data')
    validate_parser.add_argument('source', type=str,
                                  help='Draw source: "current", path to JSON, or checkpoint dir')
    validate_parser.add_argument('--year', type=int, required=True,
                                  help='Season year (e.g., 2026). Required.')
    validate_parser.add_argument('--repair', action='store_true',
                                  help='Show suggested repairs for mismatched keys')

    # Migrate command - one-time migration to new directory structure
    migrate_parser = subparsers.add_parser('migrate',
        help='Migrate draws to new versioned directory structure')
    migrate_parser.add_argument('--year', type=int, required=True,
                                help='Season year to migrate')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Route to appropriate handler
    if args.command == 'generate':
        run_generate(args)
    elif args.command == 'test':
        run_test(args)
    elif args.command == 'analyze':
        run_analyze(args)
    elif args.command == 'swap':
        run_swap(args)
    elif args.command == 'report':
        run_report(args)
    elif args.command == 'import':
        run_import(args)
    elif args.command == 'list-constraints':
        run_list_constraints()
    elif args.command == 'preseason':
        run_preseason(args)
    elif args.command == 'diagnose':
        run_diagnose(args)
    elif args.command == 'validate':
        run_validate(args)
    elif args.command == 'migrate':
        run_migrate(args)


def _load_locked_keys(path: str, locked_weeks: set) -> list:
    """
    Load locked game keys from a draw JSON, checkpoint directory, or pickle file.

    Supports:
    - Draw JSON (.json): Uses DrawStorage.load_and_lock()
    - Checkpoint directory (e.g. checkpoints/latest): Loads solution.pkl from inside
    - Pickle file (.pkl): Loads solution dict directly

    Returns list of 11-tuple keys for model.Add(X[key] == 1).
    """
    import pickle

    weeks_label = ','.join(str(w) for w in sorted(locked_weeks))
    p = Path(path)

    # Case 1: Draw JSON file
    if p.suffix == '.json':
        from analytics.storage import DrawStorage
        _, locked_keys = DrawStorage.load_and_lock(path, locked_weeks)
        return locked_keys

    # Case 2: Checkpoint directory (contains solution.pkl)
    if p.is_dir():
        pkl_path = p / 'solution.pkl'
        if not pkl_path.exists():
            print(f"ERROR: No solution.pkl found in {path}")
            sys.exit(1)
        print(f"  Loading checkpoint from {pkl_path}")
        with open(pkl_path, 'rb') as f:
            solution = pickle.load(f)
        locked_keys = [
            key for key, val in solution.items()
            if val == 1 and len(key) >= 7 and key[6] in locked_weeks
        ]
        print(f"Loaded {len(locked_keys)} locked games from checkpoint (weeks {weeks_label})")
        return locked_keys

    # Case 3: Direct pickle file
    if p.suffix == '.pkl':
        print(f"  Loading solution pickle from {path}")
        with open(path, 'rb') as f:
            solution = pickle.load(f)
        locked_keys = [
            key for key, val in solution.items()
            if val == 1 and len(key) >= 7 and key[6] in locked_weeks
        ]
        print(f"Loaded {len(locked_keys)} locked games from pickle (weeks {weeks_label})")
        return locked_keys

    print(f"ERROR: Unknown locked file format: {path}")
    print(f"  Expected: .json (draw), .pkl (pickle), or directory (checkpoint)")
    sys.exit(1)


def _parse_week_spec(spec: str) -> set:
    """Parse a week specification string into a set of ints.

    Supports:
    - Comma-separated values:  "10,12,14"  → {10, 12, 14}
    - Closed ranges:           "10-22"     → {10, 11, ..., 22}
    - Combinations:            "10-12,15"  → {10, 11, 12, 15}

    Whitespace around tokens is stripped.  An empty string returns the
    empty set.

    Raises:
        ValueError: If a token cannot be parsed as an int or a valid A-B
            range, or if A > B in a range.
    """
    if not spec or not spec.strip():
        return set()

    result: set = set()
    for token in spec.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            # Could be a range like "10-22"
            parts = token.split('-', 1)
            try:
                a, b = int(parts[0].strip()), int(parts[1].strip())
            except ValueError:
                raise ValueError(
                    f"Invalid week range token {token!r}: expected A-B with integers"
                )
            if a > b:
                raise ValueError(
                    f"Invalid week range {token!r}: start {a} > end {b}"
                )
            result.update(range(a, b + 1))
        else:
            try:
                result.add(int(token))
            except ValueError:
                raise ValueError(
                    f"Invalid week token {token!r}: expected an integer"
                )
    return result


def resolve_regen_scope(games, regen_grades, regen_weeks, lock_weeks=None):
    """Partition games into (free_ids, frozen_ids) using the union free-scope rule.

    The rule (the one non-obvious thing — document it here):
        A game is FREE iff its grade ∈ regen_grades OR its week ∈ regen_weeks.
        Everything else is frozen (to be pinned via LOCKED_PAIRINGS).

    Rationale: regenerating a grade means ALL weeks of that grade move;
    regenerating weeks 10-22 means ALL grades in those weeks move; doing
    both frees the union.  This matches the convenor's use cases:
      - Roster change → --regen-grades 5th 6th (all weeks of those grades)
      - Re-time future weeks → --regen-weeks 10-22 (all grades in those weeks)
      - Both together frees the union.

    Args:
        games: Iterable of StoredGame (or any object with .game_id, .grade,
            .week attributes).
        regen_grades: Set/iterable of grade strings to REGENERATE (free).
            Empty → no game freed along the grade axis.
        regen_weeks: Set/iterable of week ints to REGENERATE (free).
            None or empty → no game freed along the week axis.
        lock_weeks: Set/iterable of hard-locked played weeks (from
            --lock-weeks).  These are excluded from both free and frozen sets
            because they are handled by the hard-lock path, not by pinning.
            Defaults to None (empty).

    Returns:
        (free_ids, frozen_ids): Two sets of game_id strings.
        - free_ids:   games inside the regen scope (solver re-decides them).
        - frozen_ids: games outside the regen scope (to be pinned).
        Note: games in hard-locked played weeks appear in NEITHER set.
    """
    regen_grades_set = set(regen_grades) if regen_grades else set()
    regen_weeks_set = set(regen_weeks) if regen_weeks else set()
    lock_weeks_set = set(lock_weeks) if lock_weeks else set()

    free_ids: set = set()
    frozen_ids: set = set()

    for game in games:
        # Hard-locked played weeks are handled by the locked-keys path, not pinning.
        if game.week in lock_weeks_set:
            continue
        # Union rule: free if grade ∈ regen_grades OR week ∈ regen_weeks.
        if game.grade in regen_grades_set or game.week in regen_weeks_set:
            free_ids.add(game.game_id)
        else:
            frozen_ids.add(game.game_id)

    return free_ids, frozen_ids


def _validate_regen_lock_weeks_overlap(regen_weeks: set, lock_weeks: set) -> None:
    """Validate that --regen-weeks and --lock-weeks do not overlap.

    A week cannot be both hard-locked (already played, pinned exactly) and
    re-solved (free) at the same time.  If they overlap, exit non-zero with a
    clear error message naming the overlapping weeks.

    Args:
        regen_weeks: Parsed set of ints from --regen-weeks.
        lock_weeks: Parsed set of ints from --lock-weeks.

    Raises:
        SystemExit(1): if the two sets intersect.
    """
    overlap = regen_weeks & lock_weeks
    if overlap:
        weeks_str = ', '.join(str(w) for w in sorted(overlap))
        print(
            f"ERROR: --regen-weeks and --lock-weeks overlap on week(s): {weeks_str}. "
            f"A week cannot be both hard-locked (played) and re-solved (free). "
            f"Remove the overlapping week(s) from one of the flags."
        )
        sys.exit(1)


def _validate_regen_weeks_in_source(regen_weeks: set, source_weeks: set) -> None:
    """Validate that every --regen-weeks value exists in the source draw (L3).

    A regen week absent from the source draw is almost always a typo or an
    off-by-one (e.g. requesting week 0 or week 30 against a 22-week draw).
    Silently accepting it would free nothing along the week axis for that
    value, masking the mistake.  FATAL with a clear message instead.

    Args:
        regen_weeks: Parsed set of ints from --regen-weeks.
        source_weeks: Set of week ints actually present in the source draw.

    Raises:
        SystemExit(1): if any regen week is not present in the source draw.
    """
    missing = regen_weeks - source_weeks
    if missing:
        missing_str = ', '.join(str(w) for w in sorted(missing))
        present_str = ', '.join(str(w) for w in sorted(source_weeks))
        print(
            f"ERROR: --regen-weeks references week(s) not present in the source "
            f"draw: {missing_str}. Source draw weeks are: {present_str}. "
            f"Check the week numbers against the source draw."
        )
        sys.exit(1)


def _build_regen_pins(source_draw, frozen_ids):
    """Build a LOCKED_PAIRINGS pin list for EXACTLY the frozen games (spec-026 DoD #1/#3).

    Pins are built directly from the frozen StoredGame objects (matched by
    game_id) rather than re-deriving via grade/week sets.  This is the cleanest
    approach per the Unit C INJECTION ORDER NOTE: it guarantees exactly one pin
    per frozen non-bye game and zero pins for free or hard-locked games.

    Re-deriving via ``extract_locked_pairings(freeze_grades=..., freeze_weeks=...)``
    would OVER-include under the union free-scope rule: e.g. if PHL is frozen
    across weeks 1-3 and week 1 of 6th is also frozen, the grade set {PHL, 6th}
    crossed with the week set {1,2,3} would wrongly pin freed 6th-grade games in
    weeks 2-3.  Building from the frozen game_ids avoids that entirely.

    Each pin keeps ``(teams, grade, date)`` and drops time/slot/field — the
    same shape ``extract_locked_pairings`` produces, so spec-025's generate_X
    pass treats auto-extracted and hand-authored pins identically.

    Args:
        source_draw: The source DrawStorage.
        frozen_ids: Set of game_id strings that must be pinned.

    Returns:
        A list of pin dicts (one per frozen non-bye game).
    """
    pins = []
    for g in source_draw.games:
        if g.game_id not in frozen_ids:
            continue
        # Bye / placeholder guard — mirrors extract_locked_pairings.
        if not g.team1 or not g.team2:
            continue
        pins.append({
            'teams': [g.team1, g.team2],
            'grade': g.grade,
            'date': g.date,
            'description': f"Regen pin: {g.team1} vs {g.team2} ({g.grade}) {g.date}",
        })
    return pins


def _select_regen_group():
    """Resolve the spec-027 `regen` constraint group (core-hard + regen-soft + soft).

    spec-027: the groups machinery (spec-023) and the `regen` group both landed,
    so this now resolves and returns the deduped, canonically-ordered list of the
    regen group's constraint names. A scoped regeneration applies this set instead
    of the full hard constraint set — the non-physical rules are softened (their
    `*RegenSoft` analogues emit penalties) so a frozen-but-retimed draw does not go
    INFEASIBLE. A normal (non-regen) run never calls this.

    Returns:
        list[str]: the resolved regen constraint canonical names (from
        ``constraints.registry.resolve_groups(['regen'])``).
    """
    from constraints.registry import resolve_groups
    return resolve_groups(['regen'])


def _compute_regen_state(args, locked_weeks):
    """Compute (regen_locked_pairings, regen_info, regen_groups) for a regen run.

    When ``--regen-from`` is given, freeze everything outside the regen scope
    (--regen-grades / --regen-weeks) by pinning those pairings to their dates,
    while the scope is solved fresh. Hard-locked played weeks (--lock-weeks)
    are NEVER pinned (they are pinned exactly by the locked-keys path).

    Returns ``(None, None, None)`` for a non-regen run (no --regen-from).

    Args:
        args: The parsed argparse Namespace.
        locked_weeks: Set of hard-locked played weeks (from --lock-weeks).

    Returns:
        (regen_locked_pairings, regen_info, regen_groups):
        - regen_locked_pairings: list of pin dicts (one per frozen game) or None.
        - regen_info: dict for the `regen` metadata block (DoD #6) or None.
        - regen_groups: resolved regen constraint group or None (DoD #3 guard).
    """
    if not getattr(args, 'regen_from', None):
        # Warn if regen scope flags were given without --regen-from (the enabler)
        # so the user isn't silently ignored.
        if getattr(args, 'regen_grades', None) or getattr(args, 'regen_weeks', None):
            print(
                "WARNING: --regen-grades/--regen-weeks ignored because "
                "--regen-from was not specified (it is what enables regen mode)."
            )
        return None, None, None

    from analytics.storage import DrawStorage

    regen_grades = set(getattr(args, 'regen_grades', None) or [])
    regen_weeks = _parse_week_spec(getattr(args, 'regen_weeks', None) or '')

    # FATAL: --regen-weeks and --lock-weeks cannot overlap.
    _validate_regen_lock_weeks_overlap(regen_weeks, locked_weeks)

    print(f"\n[*] REGENERATION MODE — source: {args.regen_from}")
    source_draw = DrawStorage.load(args.regen_from)
    source_weeks = {g.week for g in source_draw.games}

    # FATAL: every --regen-weeks value must exist in the source draw (L3).
    _validate_regen_weeks_in_source(regen_weeks, source_weeks)

    # Compute the frozen set (excludes hard-locked weeks via lock_weeks arg).
    free_ids, frozen_ids = resolve_regen_scope(
        source_draw.games, regen_grades, regen_weeks, lock_weeks=locked_weeks
    )
    regen_locked_pairings = _build_regen_pins(source_draw, frozen_ids)

    print(f"    Regen grades (free):  {sorted(regen_grades) or '(none)'}")
    print(f"    Regen weeks  (free):  {sorted(regen_weeks) or '(all)'}")
    print(f"    Free games:           {len(free_ids)}")
    print(f"    Frozen (pinned):      {len(frozen_ids)} "
          f"-> {len(regen_locked_pairings)} pins")
    print(f"    Hard-locked weeks:    {sorted(locked_weeks) or '(none)'}")

    # DoD #3: select the regen constraint group if spec-023/spec-027 landed.
    regen_groups = _select_regen_group()

    # Metadata block (DoD #6) — written into draw metadata by save_solver_output.
    regen_info = {
        'source_draw': args.regen_from,
        'regen_grades': sorted(regen_grades),
        'regen_weeks': sorted(regen_weeks),
        'frozen_pin_count': len(regen_locked_pairings),
        'hard_locked_weeks': sorted(locked_weeks),
    }

    return regen_locked_pairings, regen_info, regen_groups


def _resolve_solver_stages(args, season_config):
    """Resolve `solver_stages` from CLI flags + season config.

    Order of precedence:
      1. `--stages-config FILE` (overrides season + default).
      2. `season_config['solver_stages']` (if set).
      3. `DEFAULT_STAGES`.

    Then applies `--stage-only` and `--skip-stage` filters in that order.
    Returns the resolved list, or `None` to leave defaults in place.
    """
    import json as _json
    from constraints.stages import load_solver_stages, validate_solver_stages

    if getattr(args, 'stages_config', None):
        with open(args.stages_config, 'r', encoding='utf-8') as f:
            stages = _json.load(f)
        if not isinstance(stages, list):
            print(f"ERROR: --stages-config {args.stages_config} must contain a JSON list")
            sys.exit(1)
    else:
        stages = load_solver_stages(season_config or {})

    only = getattr(args, 'stage_only', None)
    if only:
        stages = [s for s in stages if s.get('name') == only]
        if not stages:
            print(f"ERROR: --stage-only {only!r} matched no configured stage")
            sys.exit(1)

    skip = list(getattr(args, 'skip_stage', None) or [])
    if skip:
        stages = [s for s in stages if s.get('name') not in skip]

    errors = validate_solver_stages(stages)
    if errors:
        print("ERROR: solver_stages validation failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    return stages


def _resolve_group_selection(groups_arg, exclude, no_symmetry_breakers=False):
    """spec-023/spec-032: resolve a --groups selection to (group_names, constraint_names).

    `groups_arg` is the raw `--groups` list (or None/empty when not passed);
    `exclude` is the `--exclude` list; `no_symmetry_breakers` (spec-032) drops the
    always-on tie-breaker bundle. Returns:
      - `group_names`: the requested group names, defaulting to ['default'] when
        none were passed (so metadata always records a selection).
      - `constraint_names`: the deduped UNION of those groups' canonical names,
        in canonical (registry insertion) order, MINUS any `--exclude`d name.

    spec-032: the `symmetry_breakers` group (the three NIHCFill/SoftLex
    tie-breakers) is ALWAYS unioned into the selection — even for `--groups core`,
    which does not itself contain them — so the tie-breakers shape every solve.
    `--no-symmetry-breakers` reverses this: it adds the three to the exclude set
    so they are dropped even from the `default` group (which contains them).

    Exits with an error if a requested group name is unknown. This single
    resolution is shared by both dispatch paths so --groups selects identically
    in --simple and --staged.
    """
    from constraints.registry import resolve_group, resolve_groups, list_group_names
    group_names = list(groups_arg) if groups_arg else ['default']
    known = set(list_group_names())
    unknown_groups = [g for g in group_names if g not in known]
    if unknown_groups:
        print(f"ERROR: unknown --groups name(s): {unknown_groups}")
        print(f"  Known groups: {', '.join(list_group_names())}")
        sys.exit(1)
    exclude_set = set(exclude or [])
    # spec-032: always-on symmetry breakers. When suppressed, exclude the three
    # (covers groups like `default` that already contain them). Otherwise add the
    # `symmetry_breakers` group to the union so groups that lack it (e.g. `core`)
    # still get the tie-breakers, in canonical registry order.
    selected_groups = list(group_names)
    if no_symmetry_breakers:
        exclude_set |= resolve_group('symmetry_breakers')
    elif 'symmetry_breakers' not in selected_groups:
        selected_groups = selected_groups + ['symmetry_breakers']
    constraint_names = [c for c in resolve_groups(selected_groups) if c not in exclude_set]
    return group_names, constraint_names


def run_generate(args):
    """Generate a new draw."""
    print("="*60)
    print(f"HOCKEY DRAW SCHEDULER - SEASON {args.year}")
    print("="*60)

    from main_staged import main_staged, load_data
    from solver_diagnostics import SolverConfig, get_recommended_config

    # --list-groups: print all known group names and exit (no solver setup).
    if getattr(args, 'list_groups', False):
        from constraints.registry import list_group_names
        print("Available constraint group names (spec-023):")
        for name in list_group_names():
            print(f"  {name}")
        return

    # --list-stages: print and exit before any solver setup. Also lists group
    # names (spec-023) so operators discover --groups selectors here.
    if getattr(args, 'list_stages', False):
        from constraints.stages import list_stages
        from constraints.registry import list_group_names
        # Only load season_config if the user didn't pass --stages-config.
        season_config = {} if getattr(args, 'stages_config', None) else load_data(args.year)
        stages = _resolve_solver_stages(args, season_config)
        print("Configured solver stages:")
        print(list_stages(stages))
        print("\nAvailable constraint group names (spec-023, for --groups):")
        for name in list_group_names():
            print(f"  {name}")
        return
    
    # Parse resume arguments
    resume_run_id = None
    resume_from = None
    if args.resume is not None:
        if len(args.resume) >= 1:
            resume_run_id = args.resume[0]
        if len(args.resume) >= 2:
            resume_from = args.resume[1]
    
    final_run_id = args.run_id or resume_run_id
    
    # Configure solver based on arguments
    solver_config = None
    if args.minimal_memory:
        print("\n[*] Using MINIMAL MEMORY configuration (2 workers, no probing)")
        solver_config = SolverConfig.minimal_memory_config()
    elif args.low_memory:
        print("\n[*] Using LOW MEMORY configuration (4 workers)")
        solver_config = SolverConfig.low_memory_config()
    elif args.high_performance:
        print("\n[*] Using HIGH PERFORMANCE configuration (all cores)")
        solver_config = SolverConfig.high_performance_config()
    elif args.workers is not None:
        label = "all cores" if args.workers == 0 else str(args.workers)
        print(f"\n[*] Using custom worker count: {label}")
        solver_config = SolverConfig.balanced_config()
        solver_config.num_workers = args.workers
    else:
        print("\n[*] Using auto-detected solver configuration")
        solver_config = get_recommended_config()
    
    print(f"  Workers: {solver_config.num_workers}")
    print(f"  Linearization level: {solver_config.linearization_level}")
    
    # Handle locked games
    locked_keys = None
    locked_weeks = set()
    if args.lock_weeks:
        locked_weeks = set(int(w.strip()) for w in args.lock_weeks.split(',') if w.strip())
    if args.locked and locked_weeks:
        locked_path = args.locked
        weeks_label = ','.join(str(w) for w in sorted(locked_weeks))
        print(f"\nLoading locked games from {locked_path} (weeks {weeks_label})...")
        locked_keys = _load_locked_keys(locked_path, locked_weeks)
        # Pass source info and repair flag to solver for locked key validation
        solver_config.locked_source = locked_path
        solver_config.repair_locked = getattr(args, 'repair_locked', False)
    
    # Load hint solution if provided
    # If --locked is a pickle/checkpoint and no --hint given, use locked source as hint too
    hint_solution = None
    hint_path = getattr(args, 'hint', None)
    if not hint_path and args.locked and locked_weeks:
        locked_p = Path(args.locked)
        if locked_p.suffix == '.pkl' or locked_p.is_dir():
            hint_path = args.locked
            print(f"\n[*] Using locked source as hint: {hint_path}")
    if hint_path:
        import pickle
        if hint_path.endswith('.pkl'):
            print(f"\nLoading hint solution from pickle: {hint_path}")
            with open(hint_path, 'rb') as f:
                hint_solution = pickle.load(f)
            hint_games = sum(1 for v in hint_solution.values() if v == 1)
            print(f"  Loaded {hint_games} scheduled games as hints")
        elif Path(hint_path).is_dir():
            pkl_path = Path(hint_path) / 'solution.pkl'
            if pkl_path.exists():
                print(f"\nLoading hint solution from checkpoint: {pkl_path}")
                with open(pkl_path, 'rb') as f:
                    hint_solution = pickle.load(f)
                hint_games = sum(1 for v in hint_solution.values() if v == 1)
                print(f"  Loaded {hint_games} scheduled games as hints")
            else:
                print(f"\n[!] No solution.pkl found in {hint_path}")
        elif hint_path.endswith('.json'):
            print(f"\nLoading hint solution from draw JSON: {hint_path}")
            from analytics.storage import DrawStorage
            draw = DrawStorage.load(hint_path)
            hint_solution = draw.to_X_dict()
            print(f"  Loaded {len(hint_solution)} scheduled games as hints")
        else:
            print(f"\n[!] Unknown hint file format: {hint_path} (expected .json, .pkl, or checkpoint dir)")

    # Check for Round 1 symmetry breaking
    fix_round_1 = getattr(args, 'fix_round_1', False)
    if fix_round_1:
        print("\n[*] Round 1 symmetry breaking ENABLED")
    
    # Build constraint slack overrides from --slack argument
    constraint_slack = None
    slack_value = getattr(args, 'slack', None)
    if slack_value is not None:
        # Apply slack to all slack-aware constraints
        constraint_slack = {
            'EqualMatchUpSpacingConstraint': slack_value,
            # spec-018: AwayAtMaitlandGrouping / MaitlandHomeGrouping slack keys
            # removed — those rules were deleted (--slack on them is a no-op).
            # spec-033 Unit A: ClubVsClubAlignment slack key removed — alignment
            # is now a fixed hard rule with no slack.
            # spec-024: MaximiseClubsPerTimeslotBroadmeadow / MinimiseClubsOnAFieldBroadmeadow
            # slack keys removed — those constraints were deleted.
            'ClubGameSpread': slack_value,
            # spec-033 Unit E: ClubNoConcurrentSlot is now soft + slack — the
            # hard ceiling is 1 overlap per (club, location, slot); --slack
            # raises it to 1 + N (the release valve when a venue has fewer
            # distinct slots than a club's games).
            'ClubNoConcurrentSlot': slack_value,
            # spec-008 Part B: bye spacing has its own slack key. Mirror the
            # CLI's --slack N here so a one-shot loosen affects both matchup
            # and bye spacing in step.
            'BalancedByeSpacing': slack_value,
        }
        print(f"\n[*] Constraint slack override: +{slack_value}")
    
    # spec-036: solve-mode dispatch. No mode flag => single full solve (default);
    # --staged => DEFAULT_STAGES incremental staged solve; --severity => severity-
    # grouped staged solve. --simple/--unified were removed (forward-only).
    use_staged = getattr(args, 'staged', False)
    severity_staged_flag = getattr(args, 'severity', False)
    if use_staged and severity_staged_flag:
        print("WARNING: --staged and --severity are mutually exclusive. Using --severity.")

    exclude = args.exclude or []

    # spec-023: resolve --groups into a deduped, canonically-ordered list of
    # WHOLE constraint canonical names, minus --exclude. No --groups => the
    # 'default' group (every production constraint), which equals today's full
    # DEFAULT_STAGES selection. This single resolution is shared by BOTH the
    # staged path (main_staged) and the simple/unified path (main_simple), so
    # --groups produces an identical selection in --simple and --staged.
    groups_arg = getattr(args, 'groups', None)
    # spec-027: a scoped regeneration (--regen-from) applies the `regen` group
    # (core-hard physical rules + regen-soft analogues + soft) unless the operator
    # explicitly passed --groups. This softens the non-physical rules so a
    # frozen-but-retimed draw does not go INFEASIBLE.
    regen_active = bool(getattr(args, 'regen_from', None))
    if regen_active and not groups_arg:
        groups_arg = ['regen']
        print("[*] Regeneration: defaulting to the 'regen' constraint group "
              "(core-hard + regen-soft + soft).")
    no_symmetry_breakers = getattr(args, 'no_symmetry_breakers', False)
    group_names, constraint_names = _resolve_group_selection(
        groups_arg, exclude, no_symmetry_breakers)
    # Asymmetry note (intentional, not a behaviour difference): the simple path
    # always receives the resolved `constraint_names` (default group when no
    # --groups), whereas the staged path receives None when no --groups so it
    # runs the legacy stage list untouched. These select IDENTICALLY: filtering
    # DEFAULT_STAGES' atoms down to the `default` group's names is a no-op
    # (DEFAULT_STAGES ⊆ default group), so staged-None == staged-default-group
    # atom-for-atom. Passing None just keeps the legacy staged path byte-for-byte
    # rather than routing it through the filter for a guaranteed no-op. The
    # equivalence is pinned by test_groups_cli_wiring.py
    # ::test_staged_none_equals_default_group_filter.
    #
    # spec-032 exception: with --no-symmetry-breakers and NO --groups, we must
    # FORCE the staged filter non-None to `default`-minus-symmetry. Otherwise the
    # plain path runs DEFAULT_STAGES untouched (which still contains the three
    # symmetry atoms), so the flag would silently do nothing on the staged path.
    # `constraint_names` already excludes the three here, so it is exactly the
    # default-minus-symmetry filter the staged path needs.
    if groups_arg or no_symmetry_breakers:
        staged_constraint_filter = constraint_names
    else:
        staged_constraint_filter = None

    relax_config = None
    if getattr(args, 'relax', False):
        relax_config = {
            'enabled': True,
            'timeout': getattr(args, 'relax_timeout', 30.0),
        }

    user_description = getattr(args, 'description', '') or ''

    # Build provenance info for metadata (sources of locked games and hints)
    provenance = {}
    if args.locked and locked_weeks:
        provenance['locked_source'] = args.locked
        provenance['locked_weeks'] = sorted(locked_weeks)
        provenance['locked_game_count'] = len(locked_keys) if locked_keys else 0
    if hint_path:
        provenance['hint_source'] = hint_path

    # ----- spec-026/spec-027: Regeneration mode wiring -----
    regen_locked_pairings, regen_info, regen_groups = _compute_regen_state(
        args, locked_weeks
    )
    # spec-027: a regen run applies its constraint set (the resolved `regen` group,
    # or whatever the operator selected via --groups) through the STAGED dispatcher
    # `apply_constraint_set` — the only path that dispatches the non-engine
    # `*RegenSoft` atoms. The single-solve path cannot add them, so a
    # regen run is routed staged with a single synthetic 'regen' stage = the
    # resolved selection. We build that stage from `constraint_names` (the regen
    # group resolved above); `regen_groups` is the same set, kept for metadata.
    regen_stage_override = None
    if regen_active:
        regen_stage_override = [{
            'name': 'regen',
            'description': ('spec-027 regeneration constraint group — core-hard '
                            'physical rules + regen-soft analogues + soft'),
            'atoms': list(constraint_names),
        }]
        # spec-036: regen always dispatches via the staged constraint dispatcher
        # (the only path that applies the non-engine *RegenSoft atoms), regardless
        # of any mode flag. Inform the operator that mode flags are ignored here.
        if use_staged or severity_staged_flag:
            print("[*] Regeneration uses the staged constraint dispatcher (the "
                  "single-solve path cannot apply the regen-soft atoms); the "
                  "--staged/--severity mode flag is ignored for this run.")

    # Resolve --stages-config / --stage-only / --skip-stage. Used for
    # SOLVER_STAGES dispatch in main_staged.
    resolved_stages = None
    needs_stage_overrides = bool(
        getattr(args, 'stages_config', None)
        or getattr(args, 'stage_only', None)
        or getattr(args, 'skip_stage', None)
    )
    if needs_stage_overrides:
        season_config = {} if getattr(args, 'stages_config', None) else load_data(args.year)
        resolved_stages = _resolve_solver_stages(args, season_config)

    # spec-036 dispatch: no mode flag + no regen => single full solve (DEFAULT).
    # Regen always goes staged; --staged/--severity also go staged.
    if not regen_active and not use_staged and not severity_staged_flag:
        from main_staged import main_simple
        solution, data = main_simple(
            locked_keys=locked_keys,
            locked_weeks=locked_weeks,
            solver_config=solver_config,
            exclude_constraints=exclude,
            year=args.year,
            relax_config=relax_config,
            fix_round_1=fix_round_1,
            constraint_slack=constraint_slack,
            hint_solution=hint_solution,
            run_id=final_run_id,
            description=user_description,
            provenance=provenance,
            constraint_names=constraint_names,
            groups_selected=group_names,
            regen_locked_pairings=regen_locked_pairings,
            regen_info=regen_info,
        )
    else:
        stages = getattr(args, 'stages', None)
        # spec-036: --severity drives severity-grouped staging; --staged (or no
        # flag, when reached via regen) drives DEFAULT_STAGES incremental.
        severity_staged = severity_staged_flag
        # spec-027: a regen run applies its single synthetic 'regen' stage (the
        # resolved regen group) via the staged dispatcher, never severity staging,
        # and passes the regen set as constraint_names so the stage is unfiltered.
        if regen_active:
            staged_stages = regen_stage_override
            severity_staged = False
            staged_filter = list(constraint_names)
        else:
            staged_stages = resolved_stages
            staged_filter = staged_constraint_filter
        solution, data = main_staged(
            run_id=final_run_id,
            resume_from=resume_from,
            locked_keys=locked_keys,
            locked_weeks=locked_weeks,
            solver_config=solver_config,
            year=args.year,
            stages_to_run=stages,
            relax_config=relax_config,
            fix_round_1=fix_round_1,
            constraint_slack=constraint_slack,
            severity_staged=severity_staged,
            hint_solution=hint_solution,
            exclude_constraints=exclude,
            description=user_description,
            provenance=provenance,
            solver_stages=staged_stages,
            constraint_names=staged_filter,
            groups_selected=group_names,
            regen_locked_pairings=regen_locked_pairings,
            regen_info=regen_info,
        )

    if solution:
        print("\n[OK] Draw generated successfully!")
        print(f"  Latest draw:    draws/{args.year}/current.json")
        print(f"  Latest Excel:   draws/{args.year}/current.xlsx")
        print(f"  All versions:   draws/{args.year}/versions/")
        print(f"  Changelog:      draws/{args.year}/CHANGELOG.md")
        print(f"  Checkpoints:    checkpoints/latest/")
        print("  Check the 'logs/' folder for detailed solver logs.")
    else:
        print("\n[X] Failed to generate draw.")
        sys.exit(1)


def run_test(args):
    """Test a draw for constraint violations."""
    print("="*60)
    print("DRAW VIOLATION TEST")
    print("="*60)
    
    from analytics.tester import DrawTester
    from analytics.storage import DrawStorage
    
    data = load_data_for_year(args.year)
    
    draw_path = resolve_draw_path(args.draw_file, args.year)
    print(f"\nLoading draw from {draw_path}...")
    draw = DrawStorage.load(draw_path)
    
    print("Running constraint checks...\n")
    tester = DrawTester(draw, data)
    report = tester.run_violation_check()
    
    print(report.full_report())
    
    if report.has_violations:
        sys.exit(1)


def run_analyze(args):
    """Generate analytics report for a draw."""
    print("="*60)
    print("DRAW ANALYTICS")
    print("="*60)
    
    from analytics.storage import DrawStorage, DrawAnalytics
    
    data = load_data_for_year(args.year)
    
    draw_path = resolve_draw_path(args.draw_file, args.year)
    print(f"\nLoading draw from {draw_path}...")
    draw = DrawStorage.load(draw_path)
    
    print("Generating analytics...")
    analytics = DrawAnalytics(draw, data)
    
    output_file = args.output or args.draw_file.replace('.json', '_analytics.xlsx')
    analytics.export_analytics_to_excel(output_file)
    
    print(f"\n[OK] Analytics exported to {output_file}")
    
    print("\nQuick Summary:")
    print(analytics.grade_summary().to_string(index=False))
    
    print("\nCompliance Check:")
    print(analytics.constraint_compliance_summary().to_string(index=False))


def run_swap(args):
    """Test swapping two games and check violations."""
    print("="*60)
    print("GAME SWAP TEST")
    print("="*60)
    
    from analytics.tester import DrawTester
    from analytics.storage import DrawStorage
    
    data = load_data_for_year(args.year)
    
    draw_path = resolve_draw_path(args.draw_file, args.year)
    print(f"\nLoading draw from {draw_path}...")
    draw = DrawStorage.load(draw_path)
    
    tester = DrawTester(draw, data)
    
    # Get original violation count
    original_report = tester.run_violation_check()
    original_count = len(original_report.violations)
    
    # Perform swap
    print(f"Swapping games {args.game1} and {args.game2}...")
    success = tester.swap_games(args.game1, args.game2)
    
    if not success:
        print(f"\n[X] Failed to swap games. Check that both IDs exist.")
        sys.exit(1)
    
    # Check new violations
    new_report = tester.run_violation_check()
    new_count = len(new_report.violations)
    
    print(f"\nSwap Results:")
    print(f"  Original violations: {original_count}")
    print(f"  New violations:      {new_count}")
    print(f"  Change:              {'+' if new_count > original_count else ''}{new_count - original_count}")
    
    if new_count > original_count:
        print("\n[WARNING] Swap introduces new violations:")
        # Show only new violations
        original_msgs = {v.message for v in original_report.violations}
        for v in new_report.violations:
            if v.message not in original_msgs:
                print(f"  [{v.severity}] {v.constraint}: {v.message}")
    elif new_count < original_count:
        print("\n[OK] Swap reduces violations!")
    else:
        print("\n[INFO] Swap has no effect on violations.")
    
    if args.save:
        # Save as a minor version update through the versioning system
        from analytics.versioning import DrawVersionManager
        version_manager = DrawVersionManager('draws', year=args.year)
        old_draw = DrawStorage.load(draw_path)
        modified_draw = tester.draw
        version_manager.save_modified_draw(
            modified_draw, old_draw,
            f"Game swap: {args.game1} <-> {args.game2}"
        )
        print(f"\n[SAVED] Modified draw saved as new version in draws/{args.year}/")


def run_report(args):
    """Generate stakeholder reports."""
    print("="*60)
    print("STAKEHOLDER REPORTS")
    print("="*60)
    
    from analytics.storage import DrawStorage
    from analytics.reports import (
        ClubReport, TeamReport, GradeReport, 
        ComplianceCertificate, generate_html_report, generate_all_reports
    )
    
    data = load_data_for_year(args.year)
    
    draw_path = resolve_draw_path(args.draw_file, args.year)
    print(f"\nLoading draw from {draw_path}...")
    draw = DrawStorage.load(draw_path)
    
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    if args.all:
        generate_all_reports(args.draw_file, data, args.output)
        return
    
    if args.club:
        print(f"Generating report for club: {args.club}")
        report = ClubReport(draw, data)
        output_path = Path(args.output) / f"{args.club}_report.xlsx"
        report.generate_for_club(args.club, str(output_path))
    
    if args.team:
        print(f"Generating report for team: {args.team}")
        report = TeamReport(draw, data)
        output_path = Path(args.output) / f"{args.team.replace(' ', '_')}_report.xlsx"
        report.generate(args.team, str(output_path))
    
    if args.grade:
        print(f"Generating report for grade: {args.grade}")
        report = GradeReport(draw, data)
        output_path = Path(args.output) / f"Grade_{args.grade}_report.xlsx"
        report.generate(args.grade, str(output_path))
    
    if args.compliance:
        print("Generating compliance certificate...")
        cert = ComplianceCertificate(draw, data)
        output_path = Path(args.output) / "compliance_certificate.xlsx"
        cert.generate(str(output_path))
        
        if cert.is_compliant():
            print("\n[OK] Draw is COMPLIANT - all constraints satisfied")
        else:
            print(f"\n[X] Draw is NON-COMPLIANT - {len(cert.report.violations)} violations")
    
    if args.html:
        print("Generating HTML report...")
        output_path = Path(args.output) / "draw_report.html"
        generate_html_report(draw, data, str(output_path))
    
    print(f"\n[OK] Reports generated in {args.output}/")


def run_import(args):
    """Import draw from Excel and save as JSON."""
    print("="*60)
    print("IMPORT DRAW FROM EXCEL")
    print("="*60)
    
    from analytics.storage import DrawStorage
    
    data = load_data_for_year(args.year)
    
    print(f"\nImporting from {args.excel_file}...")
    draw = DrawStorage.from_excel(args.excel_file, data)
    
    print(f"  Imported {draw.num_games} games across {draw.num_weeks} weeks")
    
    if args.lock_weeks:
        locked_weeks = set(int(w.strip()) for w in args.lock_weeks.split(',') if w.strip())
        locked, remaining = draw.lock_and_split(locked_weeks)
        weeks_label = ','.join(str(w) for w in sorted(locked_weeks))
        print(f"  Locked weeks {weeks_label}: {locked.num_games} games")
        print(f"  Remaining weeks: {remaining.num_games} games")
        
        # Save locked portion
        locked_output = args.output or args.excel_file.replace('.xlsx', f'_locked_{weeks_label}.json')
        locked.save(locked_output)
        print(f"\n[OK] Locked draw saved to {locked_output}")
    else:
        output = args.output or args.excel_file.replace('.xlsx', '.json')
        draw.save(output)
        print(f"\n[OK] Draw saved to {output}")


def run_list_constraints():
    """List all registered atoms / constraints, grouped by SOLVER_STAGES.

    Phase 7c: legacy STAGES / STAGES_AI dicts are gone. Output now reflects
    the configured `DEFAULT_STAGES` and the canonical-name registry.
    """
    print("="*60)
    print("REGISTERED CONSTRAINTS (SOLVER_STAGES)")
    print("="*60)

    from constraints.stages import load_solver_stages, list_stages
    from constraints.registry import CONSTRAINT_REGISTRY

    stages = load_solver_stages({})
    print(list_stages(stages))

    # Show tester-only / unstaged entries separately so users can see what's
    # in the registry that isn't currently dispatched.
    staged_atoms = {a for s in stages for a in s.get('atoms', [])}
    extras = [
        name for name, info in CONSTRAINT_REGISTRY.items()
        if name not in staged_atoms and not info.tester_only
    ]
    if extras:
        print("\n--- Registered but not in DEFAULT_STAGES ---")
        for name in sorted(extras):
            info = CONSTRAINT_REGISTRY[name]
            print(f"  - {name}  (severity {info.severity_level})")


def load_data_for_year(year: int) -> dict:
    """
    Load data for a specific season year.
    
    Args:
        year: The season year (e.g., 2025, 2026)
        
    Returns:
        Complete data dict ready for solver
        
    Raises:
        ValueError: If no configuration exists for the specified year
    """
    from config import load_season_data
    return load_season_data(year)


def resolve_draw_path(draw_file: str, year: int = None) -> str:
    """
    Resolve a draw file path, supporting special aliases.
    
    Supports:
    - "current" or "latest" -> draws/{year}/current.json
    - A version string like "v2.0" -> draws/{year}/versions/draw_v2.0.json
    - Any direct file path -> used as-is
    
    Args:
        draw_file: Path string or alias
        year: Season year (required for alias resolution)
        
    Returns:
        Resolved absolute or relative file path
    """
    if draw_file.lower() in ('current', 'latest'):
        if year is None:
            print("ERROR: --year is required when using 'current' or 'latest'")
            sys.exit(1)
        resolved = f"draws/{year}/current.json"
        if not Path(resolved).exists():
            print(f"ERROR: No current draw found at {resolved}")
            print(f"  Generate a draw first: python run.py generate --year {year}")
            sys.exit(1)
        return resolved
    
    # Check for version string like "v2.0" or "v1.1"
    import re
    version_match = re.match(r'^v?(\d+\.\d+)$', draw_file)
    if version_match and year:
        version_str = version_match.group(1)
        # Try versions/ subfolder first, then base path
        for check_path in [
            f"draws/{year}/versions/draw_v{version_str}.json",
            f"draws/{year}/draw_v{version_str}.json",
        ]:
            if Path(check_path).exists():
                return check_path
        print(f"ERROR: Version v{version_str} not found in draws/{year}/versions/")
        sys.exit(1)
    
    return draw_file


def run_preseason(args):
    """Generate pre-season configuration report."""
    print("="*60)
    print(f"PRE-SEASON CONFIGURATION CHECK - {args.year}")
    print("="*60)
    
    from analytics.preseason_report import run_preseason_check
    
    success = run_preseason_check(args.year, args.output)
    
    if success:
        print("\n[OK] Pre-season validation PASSED")
    else:
        print("\n[FAILED] Pre-season validation FAILED - see errors above")
        sys.exit(1)


def _diagnose_solve_stage(data, atoms, timeout):
    """Build a fresh model, apply only the given atoms via the engine, and solve.

    Returns ``(status_name, solve_time_seconds, num_atoms_applied)``.
    """
    from datetime import datetime as _dt

    from ortools.sat.python import cp_model
    from constraints.stages import apply_solver_stage
    from constraints.unified import UnifiedConstraintEngine
    from utils import generate_X

    model = cp_model.CpModel()
    test_data = dict(data)
    test_data['penalties'] = {}
    X, conflicts = generate_X(model, test_data)
    if isinstance(test_data.get('games'), dict):
        test_data['games'] = list(test_data['games'].keys())
    test_data['team_conflicts'] = conflicts

    engine = UnifiedConstraintEngine(model, X, test_data, skip_constraints=set())
    engine.build_groupings()

    if atoms:
        local_stage = {'name': 'diagnose', 'atoms': list(atoms)}
        apply_solver_stage(
            local_stage,
            model=model, X=X, data=test_data, engine=engine,
            applied_engine_keys=set(), applied_atoms=set(),
        )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout
    solver.parameters.num_workers = 1

    start = _dt.now()
    status = solver.Solve(model)
    elapsed = (_dt.now() - start).total_seconds()
    return solver.status_name(status), elapsed, len(atoms)


def _diagnose_group_atoms(atoms):
    """Group atoms by engine skip-key for cluster-level removal testing.

    Atoms that share an engine key (e.g. all PHL atoms map to
    ``PHLAndSecondGradeTimes``) are removed as a unit — the engine
    applies a cluster atomically, so per-atom removal is meaningless
    inside a cluster. Atoms with no engine key (non-engine legacy class
    fallbacks) form singleton groups so they remove individually.
    """
    from collections import defaultdict
    from constraints.stages import atom_to_engine_key

    groups = defaultdict(list)
    for atom in atoms:
        key = atom_to_engine_key(atom) or atom
        groups[key].append(atom)
    return dict(groups)


def run_diagnose(args):
    """Find blocking atoms / clusters and report which removal makes the stage feasible.

    Phase 7c-bis: the command now drives the unified engine via
    SOLVER_STAGES instead of the deleted ``STAGES`` / ``STAGES_AI``
    dicts. ``--stage`` accepts any name from
    ``constraints.stages.load_solver_stages()`` or ``severity_N`` from
    ``severity_solver_stages()``. Removal testing operates at engine-key
    granularity (atoms inside a cluster apply together), so the
    "blocking" report names a cluster when the blocker is one of the
    atomized groups.
    """
    print("="*60)
    print(f"INFEASIBILITY DIAGNOSIS - {args.year} SEASON")
    print("="*60)

    from main_staged import load_data
    from constraints.stages import load_solver_stages, severity_solver_stages

    print(f"\nLoading {args.year} season data...")
    data = load_data(args.year)

    # Resolve stage: union of configured SOLVER_STAGES and the registry-derived
    # severity stages so users can target either.
    stages = list(load_solver_stages(data)) + list(severity_solver_stages())
    stage = next((s for s in stages if s['name'] == args.stage), None)
    if stage is None:
        print(f"[ERROR] Unknown stage: {args.stage}")
        print(f"Available stages: {sorted({s['name'] for s in stages})}")
        sys.exit(1)

    atoms = list(stage.get('atoms', []))
    print(f"\nStage: {args.stage}")
    print(f"Atoms ({len(atoms)}):")
    for a in atoms:
        print(f"  - {a}")

    # First test: all atoms together.
    print(f"\n[1] Testing ALL {len(atoms)} atoms together (timeout {args.timeout}s)...")
    status, elapsed, _ = _diagnose_solve_stage(data, atoms, args.timeout)
    print(f"    Status: {status} ({elapsed:.1f}s)")
    if status in ('OPTIMAL', 'FEASIBLE'):
        print("\n[OK] Stage is feasible — no blocking atoms.")
        sys.exit(0)
    if status != 'INFEASIBLE':
        print(f"\n[INCONCLUSIVE] Solver returned {status}. Increase --timeout for a definitive answer.")
        sys.exit(1)

    # Removal testing — group by engine key so atomized clusters are tested as a unit.
    print(f"\n[2] Stage is INFEASIBLE. Testing cluster removal at timeout {args.timeout}s...")
    groups = _diagnose_group_atoms(atoms)
    blocking = []
    for key, members in groups.items():
        remaining = [a for a in atoms if a not in set(members)]
        sub_status, sub_elapsed, _ = _diagnose_solve_stage(data, remaining, args.timeout)
        unblocks = sub_status in ('OPTIMAL', 'FEASIBLE', 'UNKNOWN')
        marker = 'unblocks' if unblocks else 'still blocked'
        print(f"    Without {key} ({len(members)} atom(s)): {sub_status} ({sub_elapsed:.1f}s) — {marker}")
        if unblocks:
            blocking.append(key)

    if blocking:
        print(f"\n[FOUND] Blocking cluster(s): {blocking}")
        if args.resolve:
            print("\n[INFO] --resolve is not yet wired to the engine path. Use the "
                  "reported clusters as input to `run.py generate --year YYYY "
                  "--exclude <atom>` for ad-hoc relaxation.")
        else:
            print("  Use --exclude on `run.py generate` to drop a blocking atom.")
        sys.exit(1)

    print("\n[INFO] No single cluster removal restores feasibility. The blocker "
          "is interactive across multiple clusters; try --exclude on subsets via "
          "`run.py generate`.")
    sys.exit(1)


def run_validate(args):
    """Validate draw game keys against current season timeslot data."""
    from config import load_season_data
    from analytics.storage import DrawStorage
    from utils import validate_draw_keys, repair_locked_keys

    print("="*60)
    print(f"DRAW KEY VALIDATION - {args.year} SEASON")
    print("="*60)

    data = load_season_data(args.year)
    timeslots = data['timeslots']

    # Resolve source
    source = args.source
    if source == 'current':
        source = f'draws/{args.year}/current.json'

    p = Path(source)
    if not p.exists():
        print(f"ERROR: Source not found: {source}")
        sys.exit(1)

    # Load draw
    if p.suffix == '.json':
        draw = DrawStorage.load(source)
        keys = [game.to_key() for game in draw.games]
        print(f"Loaded {len(keys)} games from {source}")
    elif p.suffix == '.pkl':
        import pickle
        with open(source, 'rb') as f:
            solution = pickle.load(f)
        keys = [key for key, val in solution.items() if val == 1 and len(key) >= 11]
        print(f"Loaded {len(keys)} scheduled games from pickle {source}")
    elif p.is_dir():
        import pickle
        pkl_path = p / 'solution.pkl'
        if not pkl_path.exists():
            print(f"ERROR: No solution.pkl in {source}")
            sys.exit(1)
        with open(pkl_path, 'rb') as f:
            solution = pickle.load(f)
        keys = [key for key, val in solution.items() if val == 1 and len(key) >= 11]
        print(f"Loaded {len(keys)} scheduled games from checkpoint {source}")
    else:
        print(f"ERROR: Unknown format: {source}")
        sys.exit(1)

    valid_keys, issues = validate_draw_keys(keys, timeslots, label=source)

    if not issues:
        print(f"\nAll {len(valid_keys)} game keys match current timeslot data.")
        sys.exit(0)

    print(f"\n{len(issues)} game(s) have mismatched keys:")

    # Group by mismatch type
    by_field = defaultdict(list)
    for issue in issues:
        fields = list(issue['field_diffs'].keys())
        key_label = ', '.join(fields) if fields else 'unknown'
        by_field[key_label].append(issue)

    for field_label, field_issues in sorted(by_field.items()):
        print(f"\n  --- {field_label} ({len(field_issues)} games) ---")
        for issue in field_issues[:10]:
            print(f"    {issue['reason']}")
        if len(field_issues) > 10:
            print(f"    ... and {len(field_issues) - 10} more")

    if args.repair:
        print(f"\n--- Repair suggestions ---")
        repaired, log = repair_locked_keys(keys, timeslots)
        repairable = sum(1 for r in log if r.get('repaired'))
        unrepairable = sum(1 for r in log if not r.get('repaired'))
        print(f"  Repairable: {repairable}, Unrepairable: {unrepairable}")
        for r in log[:10]:
            if r.get('repaired'):
                orig = r['original']
                fixed = r['repaired']
                diffs = ', '.join(f"{k}: {v['draw']}->{v['timeslot']}"
                                 for k, v in r['field_diffs'].items())
                print(f"    {orig[0]} vs {orig[1]} ({orig[2]}) {orig[7]}: {diffs}")

    print(f"\nValid: {len(valid_keys)}, Mismatched: {len(issues)}")
    sys.exit(1 if issues else 0)


def run_migrate(args):
    """Migrate draws from legacy flat structure to versioned directory structure."""
    print("="*60)
    print(f"MIGRATING DRAWS - {args.year} SEASON")
    print("="*60)
    
    from analytics.versioning import DrawVersionManager
    
    manager = DrawVersionManager('draws', year=args.year)
    manager.migrate_legacy_draws()
    
    print(f"\n[OK] Migration complete for {args.year}")
    print(f"  Versions folder: draws/{args.year}/versions/")
    print(f"  Current draw:    draws/{args.year}/current.json")


if __name__ == "__main__":
    main()
