"""Create draw_v4.0.json using locked week1_schedule.csv + weeks 2+ from draw_v1.0.json"""
import pandas as pd
import json
from typing import Any, Dict, List, Set

# Read locked week 1
df = pd.read_csv('draws/week1_schedule.csv')
df = df.dropna(subset=['Field'])  # Only actual games
df = df[df['Field'].isin(['EF', 'SF', 'WF'])]  # Only valid fields

# Read original draw for weeks 2+
with open('draws/2026/draw_v1.0.json', 'r') as f:
    original = json.load(f)

# Get original metadata
new_draw: Dict[str, Any] = {
    'metadata': original.get('metadata', {}),
    'teams': original.get('teams', []),
    'games': []
}

# Time to day_slot mapping (derived from original draw)
time_to_slot = {
    '08:30': 1,
    '10:00': 2,
    '11:30': 3,
    '13:00': 4,
    '14:30': 5,
    '16:00': 6,
    '17:30': 7,
    '19:00': 8,
}

# Create week 1 games from CSV
# NOTE: Solver uses SHORT field names (EF, SF, WF) - do NOT convert to long names!
week1_games: List[Dict[str, Any]] = []
for idx, row in df.iterrows():
    time_str = row['Time'].strip()
    field_short = row['Field'].strip()  # Keep EF/SF/WF as-is
    game: Dict[str, Any] = {
        'game_id': f'week1_{idx}',
        'team1': row['Team1'].strip(),
        'team2': row['Team2'].strip(),
        'grade': row['Grade'].strip(),
        'week': 1,
        'round_no': 1,
        'date': '2026-03-22',  # Week 1 date (from original draw)
        'day': 'Sunday',
        'time': time_str,
        'day_slot': time_to_slot.get(time_str, 1),
        'field_name': field_short,  # Short name for solver compatibility
        'field_location': 'Newcastle International Hockey Centre'
    }
    week1_games.append(game)

print(f'Created {len(week1_games)} games from locked week 1')

# Get weeks 2+ from original
weeks_2_plus = [g for g in original['games'] if g['week'] != 1]
print(f'Keeping {len(weeks_2_plus)} games from weeks 2+')

# Combine
new_draw['games'] = week1_games + weeks_2_plus
print(f'Total games: {len(new_draw["games"])}')

# Verify NOT PLAYING teams
week1_teams: Set[str] = set()
for g in week1_games:
    week1_teams.add(g['team1'])
    week1_teams.add(g['team2'])

all_teams: Set[str] = set(new_draw.get('teams', []))
if not all_teams:
    # Get from all games
    for g in original['games']:
        all_teams.add(g['team1'])
        all_teams.add(g['team2'])

not_playing: Set[str] = all_teams - week1_teams
print(f'Teams NOT PLAYING in week 1: {sorted([t for t in not_playing if "Maitland 4th" in t or "Wests 5th" in t])}')

# Save
with open('draws/2026/draw_v4.0.json', 'w') as f:
    json.dump(new_draw, f, indent=2)
print('Saved to draws/2026/draw_v4.0.json')
