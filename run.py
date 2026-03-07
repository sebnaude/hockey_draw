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
import os
import argparse
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
    gen_parser.add_argument('--simple', action='store_true',
                            help='Use simple single-solve instead of staged')
    gen_parser.add_argument('--run-id', type=str,
                            help='Specify run ID for checkpoints')
    gen_parser.add_argument('--locked', type=str, metavar='DRAW_FILE',
                            help='Path to draw with locked games')
    gen_parser.add_argument('--lock-weeks', type=int, default=0,
                            help='Lock games up to this week (use with --locked)')
    gen_parser.add_argument('--workers', type=int, default=None,
                            help='Number of solver workers (default: auto based on memory)')
    gen_parser.add_argument('--low-memory', action='store_true',
                            help='Use low-memory solver configuration (4 workers, minimal linearization)')
    gen_parser.add_argument('--minimal-memory', action='store_true',
                            help='Use minimal-memory solver configuration (2 workers, no probing) - very slow but stable')
    gen_parser.add_argument('--high-performance', action='store_true',
                            help='Use high-performance config (all cores, full linearization)')
    gen_parser.add_argument('--ai', action='store_true',
                            help='Use AI-enhanced constraint implementations instead of originals')
    gen_parser.add_argument('--exclude', nargs='+', metavar='CONSTRAINT',
                            help='Exclude specific constraints from simple mode solve. '
                                 'Use class names (e.g. EnsureBestTimeslotChoices or EnsureBestTimeslotChoicesAI)')
    
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
    import_parser.add_argument('--lock-weeks', type=int, help='Lock games up to this week')
    import_parser.add_argument('--output', '-o', type=str, help='Output JSON file')
    import_parser.add_argument('--year', type=int, required=True,
                              help='Season year (e.g., 2025, 2026). Required.')
    
    # List constraints command
    list_parser = subparsers.add_parser('list-constraints', help='List all constraints')
    list_parser.add_argument('--ai', action='store_true',
                            help='Show AI-enhanced constraint implementations')
    
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
    diagnose_parser.add_argument('--stage', type=str, default='stage1_required',
                                 help='Stage to analyze (default: stage1_required)')
    diagnose_parser.add_argument('--timeout', type=float, default=5.0,
                                 help='Timeout per feasibility test in seconds (default: 5)')
    diagnose_parser.add_argument('--resolve', action='store_true',
                                 help='Attempt iterative relaxation to find feasible solution')
    diagnose_parser.add_argument('--max-iterations', type=int, default=10,
                                 help='Max relaxation iterations (default: 10)')
    diagnose_parser.add_argument('--ai', action='store_true',
                                 help='Use AI constraint implementations')
    
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
        run_list_constraints(use_ai=args.ai)
    elif args.command == 'preseason':
        run_preseason(args)
    elif args.command == 'diagnose':
        run_diagnose(args)


def run_generate(args):
    """Generate a new draw."""
    print("="*60)
    print(f"HOCKEY DRAW SCHEDULER - SEASON {args.year}")
    print("="*60)
    
    from main_staged import main_staged, load_data
    from solver_diagnostics import SolverConfig, get_recommended_config
    
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
    elif args.workers:
        print(f"\n[*] Using custom worker count: {args.workers}")
        solver_config = SolverConfig.balanced_config()
        solver_config.num_workers = args.workers
    else:
        print("\n[*] Using auto-detected solver configuration")
        solver_config = get_recommended_config()
    
    print(f"  Workers: {solver_config.num_workers}")
    print(f"  Linearization level: {solver_config.linearization_level}")
    
    # Handle locked games
    locked_keys = None
    if args.locked and args.lock_weeks:
        from analytics.storage import DrawStorage
        print(f"\nLoading locked games from {args.locked} (weeks 1-{args.lock_weeks})...")
        _, locked_keys = DrawStorage.load_and_lock(args.locked, args.lock_weeks)
    
    if args.simple:
        from main_staged import main_simple
        exclude = args.exclude or []
        solution, data = main_simple(locked_keys=locked_keys, solver_config=solver_config, exclude_constraints=exclude, use_ai=args.ai, year=args.year)
    else:
        solution, data = main_staged(
            run_id=final_run_id, 
            resume_from=resume_from,
            locked_keys=locked_keys,
            solver_config=solver_config,
            year=args.year
        )
    
    if solution:
        print("\n✅ Draw generated successfully!")
        print("  Check the 'draws/' folder for output files.")
        print("  Check the 'logs/' folder for detailed solver logs.")
    else:
        print("\n❌ Failed to generate draw.")
        sys.exit(1)


def run_test(args):
    """Test a draw for constraint violations."""
    print("="*60)
    print("DRAW VIOLATION TEST")
    print("="*60)
    
    from analytics.tester import DrawTester
    from analytics.storage import DrawStorage
    
    data = load_data_for_year(args.year)
    
    print(f"\nLoading draw from {args.draw_file}...")
    draw = DrawStorage.load(args.draw_file)
    
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
    
    print(f"\nLoading draw from {args.draw_file}...")
    draw = DrawStorage.load(args.draw_file)
    
    print("Generating analytics...")
    analytics = DrawAnalytics(draw, data)
    
    output_file = args.output or args.draw_file.replace('.json', '_analytics.xlsx')
    analytics.export_analytics_to_excel(output_file)
    
    print(f"\n✅ Analytics exported to {output_file}")
    
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
    
    print(f"\nLoading draw from {args.draw_file}...")
    draw = DrawStorage.load(args.draw_file)
    
    tester = DrawTester(draw, data)
    
    # Get original violation count
    original_report = tester.run_violation_check()
    original_count = len(original_report.violations)
    
    # Perform swap
    print(f"Swapping games {args.game1} and {args.game2}...")
    success = tester.swap_games(args.game1, args.game2)
    
    if not success:
        print(f"\n❌ Failed to swap games. Check that both IDs exist.")
        sys.exit(1)
    
    # Check new violations
    new_report = tester.run_violation_check()
    new_count = len(new_report.violations)
    
    print(f"\nSwap Results:")
    print(f"  Original violations: {original_count}")
    print(f"  New violations:      {new_count}")
    print(f"  Change:              {'+' if new_count > original_count else ''}{new_count - original_count}")
    
    if new_count > original_count:
        print("\n⚠️ Swap introduces new violations:")
        # Show only new violations
        original_msgs = {v.message for v in original_report.violations}
        for v in new_report.violations:
            if v.message not in original_msgs:
                print(f"  [{v.severity}] {v.constraint}: {v.message}")
    elif new_count < original_count:
        print("\n✅ Swap reduces violations!")
    else:
        print("\n⚖️ Swap has no effect on violations.")
    
    if args.save:
        tester.save_modified_draw(args.save)
        print(f"\n💾 Modified draw saved to {args.save}")


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
    
    print(f"\nLoading draw from {args.draw_file}...")
    draw = DrawStorage.load(args.draw_file)
    
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
            print("\n✅ Draw is COMPLIANT - all constraints satisfied")
        else:
            print(f"\n❌ Draw is NON-COMPLIANT - {len(cert.report.violations)} violations")
    
    if args.html:
        print("Generating HTML report...")
        output_path = Path(args.output) / "draw_report.html"
        generate_html_report(draw, data, str(output_path))
    
    print(f"\n✅ Reports generated in {args.output}/")


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
        locked, remaining = draw.lock_and_split(args.lock_weeks)
        print(f"  Locked weeks 1-{args.lock_weeks}: {locked.num_games} games")
        print(f"  Remaining weeks: {remaining.num_games} games")
        
        # Save locked portion
        locked_output = args.output or args.excel_file.replace('.xlsx', f'_locked_{args.lock_weeks}.json')
        locked.save(locked_output)
        print(f"\n✅ Locked draw saved to {locked_output}")
    else:
        output = args.output or args.excel_file.replace('.xlsx', '.json')
        draw.save(output)
        print(f"\n✅ Draw saved to {output}")


def run_list_constraints(use_ai=False):
    """List all available constraints."""
    mode_label = "AI-ENHANCED" if use_ai else "ORIGINAL"
    print("="*60)
    print(f"AVAILABLE CONSTRAINTS ({mode_label})")
    print("="*60)
    
    from main_staged import STAGES, STAGES_AI
    stages = STAGES_AI if use_ai else STAGES
    
    for stage_id, stage_info in stages.items():
        print(f"\n{stage_info['name']} ({stage_id})")
        print("-" * 40)
        print(f"Description: {stage_info['description']}")
        print(f"Required: {'Yes' if stage_info.get('required') else 'No'}")
        print(f"Time Limit: {stage_info['max_time_seconds'] // 60} minutes")
        print("Constraints:")
        for constraint_cls in stage_info['constraints']:
            doc = constraint_cls.__doc__ or "No description"
            doc_line = doc.strip().split('\n')[0]
            print(f"  • {constraint_cls.__name__}")
            print(f"    {doc_line}")


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


def run_diagnose(args):
    """
    Find blocking constraints and resolve infeasibility.
    
    This command helps identify which constraint(s) are causing
    infeasibility and can automatically relax them to find a solution.
    """
    print("="*60)
    print(f"INFEASIBILITY DIAGNOSIS - {args.year} SEASON")
    print("="*60)
    
    from main_staged import load_data, STAGES, STAGES_AI
    from infeasibility_resolver import (
        InfeasibilityResolver, 
        ConstraintSlackRegistry,
        get_constraint_names_from_stage
    )
    
    # Load data
    print(f"\nLoading {args.year} season data...")
    data = load_data(args.year)
    
    # Get stage configuration
    stages = STAGES_AI if args.ai else STAGES
    if args.stage not in stages:
        print(f"[ERROR] Unknown stage: {args.stage}")
        print(f"Available stages: {list(stages.keys())}")
        sys.exit(1)
    
    stage_config = stages[args.stage]
    constraint_names = get_constraint_names_from_stage(stage_config)
    
    print(f"\nStage: {args.stage} ({stage_config['name']})")
    print(f"Constraints to test: {len(constraint_names)}")
    for name in constraint_names:
        print(f"  • {name}")
    
    # Create resolver
    registry = ConstraintSlackRegistry()
    resolver = InfeasibilityResolver(
        data, 
        registry, 
        timeout_per_test=args.timeout,
        verbose=True
    )
    
    if args.resolve:
        # Attempt iterative relaxation
        print(f"\nAttempting iterative resolution (max {args.max_iterations} iterations)...")
        success, relaxed = resolver.resolve_iteratively(
            constraint_names, 
            max_iterations=args.max_iterations
        )
        
        print("\n" + resolver.get_resolution_report())
        
        if success:
            print("\n[SUCCESS] Found feasible configuration!")
            print(f"Relaxed constraints: {relaxed}")
            sys.exit(0)
        else:
            print("\n[FAILED] Could not find feasible configuration")
            sys.exit(1)
    else:
        # Just find the blocking constraint
        print("\nSearching for blocking constraint...")
        blocking = resolver.find_blocking_constraint(constraint_names)
        
        if blocking:
            print(f"\n[FOUND] Blocking constraint: {blocking}")
            state = registry.get_state(blocking)
            if state and state.can_relax():
                print(f"  This constraint can be relaxed (Level {state.severity_level})")
                print(f"  Use --resolve to attempt automatic relaxation")
            else:
                print(f"  This is a Level 1 constraint - cannot be relaxed")
                print(f"  Check your data/config for conflicts")
            sys.exit(1)
        else:
            print("\n[OK] All constraints are feasible together!")
            sys.exit(0)


if __name__ == "__main__":
    main()
