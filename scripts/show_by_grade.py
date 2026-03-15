#!/usr/bin/env python
"""Show round grouped by grade."""
import json
import sys
from collections import defaultdict

def show_by_grade(draw_path, round_no):
    with open(draw_path) as f:
        draw = json.load(f)
    
    games = [g for g in draw['games'] if g['round_no'] == round_no]
    
    print(f"ROUND {round_no} - {games[0]['date'] if games else 'N/A'} - BY GRADE")
    print("=" * 70)
    
    grade_order = ['PHL', '2nd', '3rd', '4th', '5th', '6th']
    
    by_grade = defaultdict(list)
    for g in games:
        by_grade[g['grade']].append(g)
    
    for grade in grade_order:
        if grade in by_grade:
            grade_games = sorted(by_grade[grade], key=lambda x: x['time'])
            print(f"\n=== {grade} Grade ({len(grade_games)} games) ===")
            for g in grade_games:
                print(f"  {g['time']}  {g['team1']:20} vs {g['team2']:20}  {g['field_name']:4}")

if __name__ == '__main__':
    draw_path = sys.argv[1] if len(sys.argv) > 1 else 'draws/2026/draw_v1.0.json'
    round_no = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    show_by_grade(draw_path, round_no)
