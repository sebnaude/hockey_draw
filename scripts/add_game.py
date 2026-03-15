#!/usr/bin/env python
"""Add a game to a draw."""
import json
import sys

def add_game(draw_path, team1, team2, grade, time, field, round_no=1, week=1, date='2026-03-22'):
    with open(draw_path) as f:
        draw = json.load(f)
    
    # Find day_slot from time
    time_to_slot = {
        '08:30': 1, '10:00': 2, '11:30': 3, '13:00': 4,
        '14:30': 5, '16:00': 6, '17:30': 7, '19:00': 8
    }
    day_slot = time_to_slot.get(time, 2)
    
    new_game = {
        'game_id': f'G{draw["num_games"]:05d}',
        'team1': team1,
        'team2': team2,
        'grade': grade,
        'week': week,
        'round_no': round_no,
        'date': date,
        'day': 'Sunday',
        'time': time,
        'day_slot': day_slot,
        'field_name': field,
        'field_location': 'Newcastle International Hockey Centre'
    }
    
    draw['games'].append(new_game)
    draw['num_games'] += 1
    
    with open(draw_path, 'w') as f:
        json.dump(draw, f, indent=2)
    
    print(f"Added: {team1} vs {team2} at {time} on {field}")
    print(f"Total games: {draw['num_games']}")

if __name__ == '__main__':
    # Usage: python add_game.py draw_path team1 team2 grade time field
    add_game(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
