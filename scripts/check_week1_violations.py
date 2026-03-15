"""Verify no team plays multiple games in Week 1."""

import json
from collections import defaultdict

# Load draw
d = json.load(open('draws/2026/draw_v1.0.json'))

# Get Week 1 games
week1_games = [g for g in d['games'] if g['week'] == 1]

# Count games per team
team_game_counts = defaultdict(list)
for g in week1_games:
    team_game_counts[g['team1']].append(f"vs {g['team2']} @ {g['field_name']} {g['time']}")
    team_game_counts[g['team2']].append(f"vs {g['team1']} @ {g['field_name']} {g['time']}")

print(f"Week 1: {len(week1_games)} games")
print("\nTeams with multiple games in Week 1 (VIOLATIONS):")

violations = False
for team, games in sorted(team_game_counts.items()):
    if len(games) > 1:
        print(f"  ⚠️ {team}: {len(games)} games:")
        for g in games:
            print(f"      {g}")
        violations = True

if not violations:
    print("  ✅ None - all teams play exactly once (or have a bye)")

# Check for byes (teams not playing in Week 1)
# Need to load all teams from the draw
all_teams = set()
for g in d['games']:
    all_teams.add(g['team1'])
    all_teams.add(g['team2'])

playing_teams = set(team_game_counts.keys())
bye_teams = all_teams - playing_teams

print(f"\nTeams with byes in Week 1 ({len(bye_teams)}):")
for team in sorted(bye_teams):
    print(f"  - {team}")
