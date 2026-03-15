#!/usr/bin/env python
"""Apply edits to Round 1."""
import json

with open('draws/2026/draw_v1.0.json') as f:
    draw = json.load(f)

# 1. Remove the game we just added (Port Stephens 6th vs Tigers Black 6th)
before = len(draw['games'])
draw['games'] = [g for g in draw['games'] if not (g['team1'] == 'Port Stephens 6th' and g['team2'] == 'Tigers Black 6th' and g['round_no'] == 1)]
after = len(draw['games'])
if before != after:
    print(f"Removed Port Stephens 6th vs Tigers Black 6th game")

for g in draw['games']:
    if g['round_no'] != 1:
        continue
    
    # 2. Move Maitland 6th vs Souths 6th from 19:00 WF to 16:00 WF
    if g['team1'] == 'Maitland 6th' and g['team2'] == 'Souths 6th':
        print(f"Moving {g['team1']} vs {g['team2']} from {g['time']} to 16:00")
        g['time'] = '16:00'
        g['day_slot'] = 6
    
    # 3. Move Colts Green 5th vs Tigers 5th from 08:30 WF to 10:00 EF
    if g['team1'] == 'Colts Green 5th' and g['team2'] == 'Tigers 5th':
        print(f"Moving {g['team1']} vs {g['team2']} from {g['time']} {g['field_name']} to 10:00 EF")
        g['time'] = '10:00'
        g['day_slot'] = 2
        g['field_name'] = 'EF'
    
    # 4. Swap times: Norths 2nd vs Tigers 2nd (11:30) <-> Port Stephens 4th vs Uni Seapigs 4th (13:00)
    if g['team1'] == 'Norths 2nd' and g['team2'] == 'Tigers 2nd':
        print(f"Moving {g['team1']} vs {g['team2']} from {g['time']} to 13:00")
        g['time'] = '13:00'
        g['day_slot'] = 4
    
    if g['team1'] == 'Port Stephens 4th' and g['team2'] == 'Uni Seapigs 4th':
        print(f"Moving {g['team1']} vs {g['team2']} from {g['time']} to 11:30")
        g['time'] = '11:30'
        g['day_slot'] = 3

draw['num_games'] = len(draw['games'])

with open('draws/2026/draw_v1.0.json', 'w') as f:
    json.dump(draw, f, indent=2)

print(f"Done. Total games: {draw['num_games']}")
