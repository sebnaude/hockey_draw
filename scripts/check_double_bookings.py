#!/usr/bin/env python
"""Check for double-booked field slots in a draw."""
import json
import sys
from collections import defaultdict

def main():
    draw_file = sys.argv[1] if len(sys.argv) > 1 else 'draws/2026/draw_v2.0.json'
    
    with open(draw_file) as f:
        d = json.load(f)
    
    games = d['games']
    print(f"Loaded {len(games)} games")
    
    # Group by field/week/slot
    by_slot = defaultdict(list)
    for g in games:
        key = (g['field_name'], g['field_location'], g['week'], g['day_slot'])
        by_slot[key].append(g)
    
    print("\nDouble-booked slots:")
    count = 0
    for k, v in sorted(by_slot.items()):
        if len(v) > 1:
            count += 1
            field, loc, week, slot = k
            print(f"\n  {field} at {loc} - Week {week}, Slot {slot}: {len(v)} games")
            for g in v:
                print(f"    {g['team1']} vs {g['team2']} ({g['grade']}) - {g['date']} {g['time']}")
    
    print(f"\nTotal double-booked slots: {count}")

if __name__ == '__main__':
    main()
