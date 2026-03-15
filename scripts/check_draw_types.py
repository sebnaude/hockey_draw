import json

d = json.load(open('draws/2026/draw_v1.0.json'))
g = d['games'][0]

print("Sample game key components:")
print(f"  team1: {repr(g['team1'])} ({type(g['team1']).__name__})")
print(f"  team2: {repr(g['team2'])} ({type(g['team2']).__name__})")
print(f"  grade: {repr(g['grade'])} ({type(g['grade']).__name__})")
print(f"  day: {repr(g['day'])} ({type(g['day']).__name__})")
print(f"  day_slot: {repr(g['day_slot'])} ({type(g['day_slot']).__name__})")
print(f"  time: {repr(g['time'])} ({type(g['time']).__name__})")
print(f"  week: {repr(g['week'])} ({type(g['week']).__name__})")
print(f"  date: {repr(g['date'])} ({type(g['date']).__name__})")
print(f"  round_no: {repr(g['round_no'])} ({type(g['round_no']).__name__})")
print(f"  field_name: {repr(g['field_name'])} ({type(g['field_name']).__name__})")
print(f"  field_location: {repr(g['field_location'])} ({type(g['field_location']).__name__})")

# Check first week 1 game (our locked games)
print("\nWeek 1 games loaded from draw:")
week1_games = [g for g in d['games'] if g['week'] == 1]
print(f"  Total week 1 games: {len(week1_games)}")
for g in week1_games[:3]:
    print(f"  - {g['team1']} vs {g['team2']} @ {g['field_name']} {g['time']}")
