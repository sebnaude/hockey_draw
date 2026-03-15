"""Verify PHLAndSecondGradeTimes violations."""
import json
from collections import defaultdict

draw = json.load(open('draws/2026/draw_v2.0.json'))

# Build team to club mapping
team_to_club = {}
for g in draw['games']:
    for team in [g['team1'], g['team2']]:
        if 'Souths' in team: team_to_club[team] = 'Souths'
        elif 'Wests' in team: team_to_club[team] = 'Wests'
        elif 'Tigers' in team: team_to_club[team] = 'Tigers'
        elif 'Norths' in team: team_to_club[team] = 'Norths'
        elif 'Maitland' in team: team_to_club[team] = 'Maitland'
        elif 'Gosford' in team: team_to_club[team] = 'Gosford'
        elif 'Uni' in team or 'University' in team: team_to_club[team] = 'University'
        elif 'Colts' in team: team_to_club[team] = 'Colts'
        elif 'Crusaders' in team: team_to_club[team] = 'Crusaders'
        elif 'Port' in team: team_to_club[team] = 'Port Stephens'

# Check flagged cases
cases = [
    (5, 'Sunday', 4, 'Souths'),
    (25, 'Sunday', 3, 'Wests'),
    (13, 'Sunday', 4, 'Souths'),
    (13, 'Sunday', 4, 'Wests'),
    (19, 'Sunday', 6, 'Tigers'),
]

for week, day, slot, club in cases:
    print(f'=== Week {week}, {day} slot {slot} ({club}) ===')
    found_phl = False
    found_2nd = False
    for g in draw['games']:
        if g['week'] == week and g['day'] == day and g['day_slot'] == slot:
            clubs = [team_to_club.get(g['team1'], '?'), team_to_club.get(g['team2'], '?')]
            if club in clubs:
                print(f"  {g['grade']}: {g['team1']} vs {g['team2']} @ {g['time']} {g['field_location']}")
                if g['grade'] == 'PHL': found_phl = True
                if g['grade'] == '2nd': found_2nd = True
    if found_phl and found_2nd:
        print("  >>> VIOLATION CONFIRMED: PHL and 2nd at same slot!")
    print()
