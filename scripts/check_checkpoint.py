"""Check latest checkpoint for Friday night games."""
import pickle
from pathlib import Path

# Get latest checkpoint
cp_dirs = sorted(Path('checkpoints/run_46').iterdir(), key=lambda x: x.stat().st_mtime)
latest = cp_dirs[-1]
print(f'Using checkpoint: {latest.name}')

with open(latest / 'solution.pkl', 'rb') as f:
    solution = pickle.load(f)

# Extract assigned games (value == 1)
games = []
for key, val in solution.items():
    if val == 1:
        home, away, grade, day, slot, time, week, date, rnd, field, venue = key
        games.append({
            'home': home, 'away': away, 'grade': grade,
            'day': day, 'slot': slot, 'time': time,
            'week': week, 'date': date, 'round': rnd,
            'field': field, 'venue': venue
        })

print(f'Total assigned games: {len(games)}')
print()

# Friday night games
print('=== FRIDAY NIGHT GAMES ===')
friday_games = [g for g in games if g['day'] == 'Friday']
print(f'Total Friday night games: {len(friday_games)}')
print()
for g in sorted(friday_games, key=lambda x: (x['date'], x['time'])):
    print(f"  R{g['round']} {g['date']} {g['time']} -- {g['home']} vs {g['away']} ({g['grade']}) -- {g['field']} {g['venue']}")
