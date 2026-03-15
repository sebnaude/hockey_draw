#!/usr/bin/env python
"""Apply Round 1 changes - fix 3rd and 4th grade."""
import json

with open('draws/2026/draw_v1.0.json') as f:
    draw = json.load(f)

changes = []

for g in draw['games']:
    if g['round_no'] != 1:
        continue
    
    # 1. Change Crusaders 4th vs Uni 4th -> Wests Red 4th vs Crusaders 4th
    if g['grade'] == '4th' and g['team1'] == 'Crusaders 4th' and g['team2'] == 'Uni 4th':
        changes.append(f"Changed: Crusaders 4th vs Uni 4th -> Crusaders 4th vs Wests Red 4th")
        g['team2'] = 'Wests Red 4th'
    
    # 2. Change Maitland 3rd vs Souths 3rd -> Crusaders 3rd vs Maitland 3rd at 17:30 WF
    if g['grade'] == '3rd' and g['team1'] == 'Maitland 3rd' and g['team2'] == 'Souths 3rd':
        changes.append(f"Changed: Maitland 3rd vs Souths 3rd -> Crusaders 3rd vs Maitland 3rd")
        g['team1'] = 'Crusaders 3rd'
        g['team2'] = 'Maitland 3rd'

# 3. Add Souths 3rd vs Wests 3rd on WF at 08:30
new_game1 = {
    'game_id': f'G{draw["num_games"]:05d}',
    'team1': 'Souths 3rd',
    'team2': 'Wests 3rd',
    'grade': '3rd',
    'week': 1,
    'round_no': 1,
    'date': '2026-03-22',
    'day': 'Sunday',
    'time': '08:30',
    'day_slot': 1,
    'field_name': 'WF',
    'field_location': 'Newcastle International Hockey Centre'
}
draw['games'].append(new_game1)
changes.append(f"Added: Souths 3rd vs Wests 3rd at 08:30 WF")

# 4. Add Norths 3rd vs Port Stephens 3rd at 17:30 SF
new_game2 = {
    'game_id': f'G{draw["num_games"]+1:05d}',
    'team1': 'Norths 3rd',
    'team2': 'Port Stephens 3rd',
    'grade': '3rd',
    'week': 1,
    'round_no': 1,
    'date': '2026-03-22',
    'day': 'Sunday',
    'time': '17:30',
    'day_slot': 7,
    'field_name': 'SF',
    'field_location': 'Newcastle International Hockey Centre'
}
draw['games'].append(new_game2)
changes.append(f"Added: Norths 3rd vs Port Stephens 3rd at 17:30 SF")

draw['num_games'] = len(draw['games'])

with open('draws/2026/draw_v1.0.json', 'w') as f:
    json.dump(draw, f, indent=2)

print("Changes made:")
for c in changes:
    print(f"  {c}")
print(f"\nTotal games: {draw['num_games']}")
