#!/usr/bin/env python
"""Apply Round 1 changes."""
import json

with open('draws/2026/draw_v1.0.json') as f:
    draw = json.load(f)

changes = []

# 1. Switch Norths/Tigers PHL (11:30) with Norths/Tigers 2nd (13:00)
for g in draw['games']:
    if g['round_no'] != 1:
        continue
    
    # PHL at 11:30 -> move to 13:00
    if g['grade'] == 'PHL' and 'Norths PHL' in [g['team1'], g['team2']] and 'Tigers PHL' in [g['team1'], g['team2']]:
        changes.append(f"Norths vs Tigers PHL: {g['time']} -> 13:00")
        g['time'] = '13:00'
        g['day_slot'] = 4
    
    # 2nd at 13:00 -> move to 11:30
    if g['grade'] == '2nd' and 'Norths 2nd' in [g['team1'], g['team2']] and 'Tigers 2nd' in [g['team1'], g['team2']]:
        changes.append(f"Norths vs Tigers 2nd: {g['time']} -> 11:30")
        g['time'] = '11:30'
        g['day_slot'] = 3

# 2. Add Port Stephens 6th vs Tigers Black 6th on SF at 14:30
new_game1 = {
    'game_id': f'G{draw["num_games"]:05d}',
    'team1': 'Port Stephens 6th',
    'team2': 'Tigers Black 6th',
    'grade': '6th',
    'week': 1,
    'round_no': 1,
    'date': '2026-03-22',
    'day': 'Sunday',
    'time': '14:30',
    'day_slot': 5,
    'field_name': 'SF',
    'field_location': 'Newcastle International Hockey Centre'
}
draw['games'].append(new_game1)
changes.append(f"Added: Port Stephens 6th vs Tigers Black 6th at 14:30 SF")

# 3. Add Uni 4th vs Crusaders 4th on SF at 13:00
new_game2 = {
    'game_id': f'G{draw["num_games"]+1:05d}',
    'team1': 'Crusaders 4th',
    'team2': 'Uni 4th',
    'grade': '4th',
    'week': 1,
    'round_no': 1,
    'date': '2026-03-22',
    'day': 'Sunday',
    'time': '13:00',
    'day_slot': 4,
    'field_name': 'SF',
    'field_location': 'Newcastle International Hockey Centre'
}
draw['games'].append(new_game2)
changes.append(f"Added: Crusaders 4th vs Uni 4th at 13:00 SF")

# 4. Add Maitland 3rd vs Souths 3rd on WF at 17:30
new_game3 = {
    'game_id': f'G{draw["num_games"]+2:05d}',
    'team1': 'Maitland 3rd',
    'team2': 'Souths 3rd',
    'grade': '3rd',
    'week': 1,
    'round_no': 1,
    'date': '2026-03-22',
    'day': 'Sunday',
    'time': '17:30',
    'day_slot': 7,
    'field_name': 'WF',
    'field_location': 'Newcastle International Hockey Centre'
}
draw['games'].append(new_game3)
changes.append(f"Added: Maitland 3rd vs Souths 3rd at 17:30 WF")

draw['num_games'] = len(draw['games'])

with open('draws/2026/draw_v1.0.json', 'w') as f:
    json.dump(draw, f, indent=2)

print("Changes made:")
for c in changes:
    print(f"  {c}")
print(f"\nTotal games: {draw['num_games']}")
