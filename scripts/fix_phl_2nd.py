#!/usr/bin/env python
"""Reorganize Round 1 PHL and 2nd grade matchups."""
import json

with open('draws/2026/draw_v1.0.json') as f:
    draw = json.load(f)

# Current state in Round 1:
# EF 11:30: Norths PHL vs Souths PHL
# EF 13:00: Tigers PHL vs Wests PHL
# WF 10:00: Souths 2nd vs Wests 2nd
# WF 11:30: Port Stephens 4th vs Uni Seapigs 4th
# WF 13:00: Norths 2nd vs Tigers 2nd

# Target state:
# EF 11:30: Tigers PHL vs Norths PHL (Tigers/Norths pair on EF)
# EF 13:00: Norths 2nd vs Tigers 2nd (Tigers/Norths 2nd back-to-back)
# WF 10:00: Souths 2nd vs Wests 2nd (Wests/Souths 2nd stays)
# WF 11:30: Wests PHL vs Souths PHL (Wests/Souths pair on WF)
# WF 13:00: Port Stephens 4th vs Uni Seapigs 4th (displaced game)

changes = []

for g in draw['games']:
    if g['round_no'] != 1:
        continue
    
    # 1. Change Norths PHL vs Souths PHL at 11:30 EF -> Tigers PHL vs Norths PHL
    if g['team1'] == 'Norths PHL' and g['team2'] == 'Souths PHL':
        changes.append(f"EF 11:30: {g['team1']} vs {g['team2']} -> Tigers PHL vs Norths PHL")
        g['team1'] = 'Norths PHL'
        g['team2'] = 'Tigers PHL'
        # Keep time 11:30, field EF
    
    # 2. Change Tigers PHL vs Wests PHL at 13:00 EF -> Wests PHL vs Souths PHL at 11:30 WF
    elif g['team1'] == 'Tigers PHL' and g['team2'] == 'Wests PHL':
        changes.append(f"EF 13:00: {g['team1']} vs {g['team2']} -> Wests PHL vs Souths PHL at 11:30 WF")
        g['team1'] = 'Souths PHL'
        g['team2'] = 'Wests PHL'
        g['time'] = '11:30'
        g['day_slot'] = 3
        g['field_name'] = 'WF'
    
    # 3. Move Norths 2nd vs Tigers 2nd from 13:00 WF -> 13:00 EF
    elif g['team1'] == 'Norths 2nd' and g['team2'] == 'Tigers 2nd':
        changes.append(f"WF 13:00: {g['team1']} vs {g['team2']} -> moved to 13:00 EF")
        g['field_name'] = 'EF'
        # Keep time 13:00
    
    # 4. Move Port Stephens 4th vs Uni Seapigs 4th from 11:30 WF -> 13:00 WF
    elif g['team1'] == 'Port Stephens 4th' and g['team2'] == 'Uni Seapigs 4th':
        changes.append(f"WF 11:30: {g['team1']} vs {g['team2']} -> moved to 13:00 WF")
        g['time'] = '13:00'
        g['day_slot'] = 4
        # Keep field WF

with open('draws/2026/draw_v1.0.json', 'w') as f:
    json.dump(draw, f, indent=2)

print("Changes made:")
for c in changes:
    print(f"  {c}")
print(f"\nTotal games: {draw['num_games']}")
