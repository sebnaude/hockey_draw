# example_draw_workflow.py
"""
Example workflow demonstrating the new draw analytics and testing tools.

This script shows how to:
1. Save a draw in the pliable JSON format
2. Generate comprehensive analytics
3. Test game modifications for constraint violations
4. Generate violation reports

Run this after solving a draw to see the full workflow.
"""

from pathlib import Path
import sys

# Ensure imports work
sys.path.insert(0, str(Path(__file__).parent))

from draw_analytics import DrawStorage, DrawAnalytics, load_draw
from draw_tester import DrawTester, test_draw, what_if_move_game


def example_save_and_analyze(X_solution: dict, data: dict):
    """
    Example 1: Save a draw and generate analytics.
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Save Draw & Generate Analytics")
    print("="*60)
    
    # 1. Create DrawStorage from solution
    draw = DrawStorage.from_X_solution(X_solution, description="Season 2025 Draw v1")
    print(f"✓ Created draw with {draw.num_games} games across {draw.num_weeks} weeks")
    
    # 2. Save to JSON (pliable format)
    draw.save("draw_2025.json")
    print("✓ Saved to draw_2025.json")
    
    # 3. Generate analytics
    analytics = DrawAnalytics(draw, data)
    
    # 4. Export comprehensive Excel report
    analytics.export_analytics_to_excel("draw_analytics_2025.xlsx")
    print("✓ Exported analytics to draw_analytics_2025.xlsx")
    
    # 5. Quick compliance check
    compliance = analytics.constraint_compliance_summary()
    print("\nCompliance Summary:")
    print(compliance.to_string(index=False))
    
    return draw


def example_query_draw(draw: DrawStorage, data: dict):
    """
    Example 2: Query the draw for specific information.
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Query Draw Data")
    print("="*60)
    
    # Query games for a specific team
    team_games = draw.get_games_by_team("Maitland PHL")
    print(f"\nMaitland PHL has {len(team_games)} games:")
    for game in team_games[:5]:
        print(f"  Week {game.week}: vs {game.team2 if game.team1 == 'Maitland PHL' else game.team1} "
              f"at {game.field_location} ({game.day} {game.time})")
    
    # Query games for a specific week
    week_5_games = draw.get_games_by_week(5)
    print(f"\nWeek 5 has {len(week_5_games)} games")
    
    # Filter with multiple criteria
    maitland_away_games = draw.filter_games(
        team="Maitland",
        field_location="Newcastle International Hockey Centre"
    )
    print(f"\nMaitland teams have {len(maitland_away_games)} games at Broadmeadow")
    
    # Get club season schedule
    analytics = DrawAnalytics(draw, data)
    maitland_schedule = analytics.club_season_schedule("Maitland")
    print(f"\nMaitland club season schedule ({len(maitland_schedule)} entries)")


def example_test_modifications(draw_path: str, data: dict):
    """
    Example 3: Test game modifications for violations.
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Test Game Modifications")
    print("="*60)
    
    # Load the draw and create tester
    tester = DrawTester.from_file(draw_path, data)
    
    # First, check current state
    report = tester.run_violation_check()
    print("\nCurrent draw status:")
    print(report.summary())
    
    # Find a specific game
    games = tester.find_game(team="Souths PHL", week=3)
    if games:
        game = games[0]
        print(f"\nFound game: {game.game_id} - {game.team1} vs {game.team2}")
        print(f"  Current: Week {game.week}, {game.day} slot {game.day_slot}")
        
        # Try moving it to a different slot
        print(f"\nTesting: Move this game to week 4, slot 5...")
        tester.move_game(
            game.game_id,
            new_week=4,
            new_day_slot=5
        )
        
        # Check for violations after the move
        new_report = tester.run_violation_check()
        print("\nAfter modification:")
        print(new_report.summary())
        
        if new_report.has_violations:
            print("\nViolations caused by this move:")
            for v in new_report.violations[:5]:
                print(f"  • [{v.severity}] {v.constraint}: {v.message}")
        
        # Reset to original
        tester.reset()
        print("\n✓ Reset to original draw")


def example_what_if_analysis(draw_path: str, data: dict):
    """
    Example 4: What-if analysis for game changes.
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: What-If Analysis")
    print("="*60)
    
    tester = DrawTester.from_file(draw_path, data)
    
    # Scenario: Club wants to move Friday game to Saturday
    print("\nScenario: A club wants to move their Friday night game to Saturday afternoon")
    
    # Find Friday games
    friday_games = [g for g in tester.draw.games if g.day == "Friday"]
    
    if friday_games:
        game = friday_games[0]
        print(f"\nOriginal game: {game.game_id}")
        print(f"  {game.team1} vs {game.team2}")
        print(f"  {game.day} {game.time} at {game.field_name}")
        
        # Test moving to Saturday
        tester.move_game(
            game.game_id,
            new_day="Saturday",
            new_time="14:00",
            new_day_slot=3
        )
        
        report = tester.run_violation_check()
        
        print(f"\nWhat-if result (move to Saturday 14:00):")
        if report.has_violations:
            print("  ❌ This move would cause violations:")
            for v in report.violations:
                print(f"     • {v.constraint}: {v.message}")
        else:
            print("  ✅ This move would be valid!")


def example_full_violation_report(draw_path: str, data: dict):
    """
    Example 5: Generate full violation report.
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Full Violation Report")
    print("="*60)
    
    report = test_draw(draw_path, data)
    print(report.full_report())


def example_compare_before_after():
    """
    Example 6: Compare draws before and after modification.
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Compare Draw Modifications")
    print("="*60)
    
    print("""
    # Load original
    tester = DrawTester.from_file("draw_original.json", data)
    original_report = tester.run_violation_check()
    
    # Make modifications
    tester.move_game("G00123", new_week=5, new_day_slot=2)
    tester.swap_games("G00045", "G00067")
    
    # Check violations
    modified_report = tester.run_violation_check()
    
    # Compare
    print(f"Original violations: {len(original_report.violations)}")
    print(f"Modified violations: {len(modified_report.violations)}")
    
    # Save if acceptable
    if not modified_report.has_violations:
        tester.save_modified_draw("draw_modified.json")
    """)


# ============== Main Demo ==============

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DRAW ANALYTICS & TESTING WORKFLOW DEMO")
    print("="*60)
    
    print("""
This script demonstrates the new draw management tools:

1. DrawStorage - Pliable JSON format for draw storage
   - Save/load draws easily
   - Query games by team, week, grade, etc.
   - Convert to/from X solution format

2. DrawAnalytics - Comprehensive analytics generation
   - Games per team/grade cross-tabs
   - Home/away balance analysis
   - Team matchup matrices
   - Club season schedules
   - Field usage analysis
   - Export to multi-sheet Excel

3. DrawTester - Modification testing
   - Move games to test scenarios
   - Swap game timeslots
   - Run constraint violation checks
   - Generate violation reports

USAGE:
------
    
# After solving:
from draw_analytics import DrawStorage, DrawAnalytics
from draw_tester import DrawTester

# Save the solution
draw = DrawStorage.from_X_solution(X_solution, "Season 2025")
draw.save("draws/season_2025.json")

# Generate analytics
analytics = DrawAnalytics(draw, data)
analytics.export_analytics_to_excel("analytics/season_2025.xlsx")

# Test a modification
tester = DrawTester.from_file("draws/season_2025.json", data)
game = tester.find_game(team="Maitland PHL", week=3)[0]
tester.move_game(game.game_id, new_week=4, new_day_slot=2)
report = tester.run_violation_check()
print(report.full_report())
    """)
