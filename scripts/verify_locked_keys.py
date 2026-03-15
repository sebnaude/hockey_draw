"""
Verify that locked game keys from draw match keys in solver.X.

This script simulates the locking mechanism:
1. Load the draw file and generate locked keys (via DrawStorage.load_and_lock)
2. Build the solver data and generate X dictionary keys  
3. Check which locked keys exist in X
"""

import sys
sys.path.insert(0, '.')

from analytics.storage import DrawStorage
from config import season_2026
from utils import build_season_data, generate_X
from ortools.sat.python import cp_model

# Load locked games from draw
draw_path = 'draws/2026/draw_v4.0.json'
lock_weeks = 1

print(f"Loading locked games from {draw_path} (weeks 1-{lock_weeks})...")
locked_draw, locked_keys = DrawStorage.load_and_lock(draw_path, lock_weeks)

print(f"\nLocked keys ({len(locked_keys)} total):")
for i, key in enumerate(locked_keys[:5]):
    print(f"  Key {i+1}: {key[:3]}... @ {key[5]} day_slot={key[4]} week={key[6]}")
if len(locked_keys) > 5:
    print(f"  ... and {len(locked_keys) - 5} more")

# Build season data (same as solver)
print(f"\nBuilding season data for {season_2026.SEASON_CONFIG['year']}...")
data = build_season_data(season_2026.SEASON_CONFIG)

# Generate X dictionary (same as solver)
print("\nGenerating X dictionary...")
model = cp_model.CpModel()
X, Y, conflicts, unavailable = generate_X(model, data)

print(f"\nX dictionary has {len(X)} keys")
print("Sample X key format:")
sample_key = list(X.keys())[0]
print(f"  {sample_key[:3]}... @ {sample_key[5]} day_slot={sample_key[4]} week={sample_key[6]}")

# Check which locked keys exist in X
print("\n" + "="*60)
print("VERIFICATION: Checking locked keys against X...")
print("="*60)

found = 0
not_found = 0
not_found_keys = []

for key in locked_keys:
    if key in X:
        found += 1
    else:
        not_found += 1
        not_found_keys.append(key)

print(f"\n✓ Found in X: {found}/{len(locked_keys)}")
print(f"✗ NOT found in X: {not_found}/{len(locked_keys)}")

if not_found_keys:
    print("\nKeys NOT found in X (first 10):")
    for key in not_found_keys[:10]:
        t1, t2, grade, day, day_slot, time, week, date, round_no, field_name, field_loc = key
        print(f"  {t1} vs {t2} ({grade}) @ {field_name} {day} {time} week={week}")
        
    # Debug: find a similar key for one of the not-found keys
    if not_found_keys:
        test_key = not_found_keys[0]
        t1, t2, grade = test_key[0], test_key[1], test_key[2]
        print(f"\nLooking for any X key with same teams: {t1} vs {t2} ({grade})...")
        matching = [k for k in X.keys() if k[0] == t1 and k[1] == t2 and k[2] == grade]
        if matching:
            print(f"  Found {len(matching)} X keys for this game")
            sample = matching[0]
            print(f"  Sample X key: day={sample[3]}, slot={sample[4]}, time={sample[5]}, week={sample[6]}, date={sample[7]}, round={sample[8]}, field={sample[9]}, loc={sample[10]}")
            print(f"  Locked key:   day={test_key[3]}, slot={test_key[4]}, time={test_key[5]}, week={test_key[6]}, date={test_key[7]}, round={test_key[8]}, field={test_key[9]}, loc={test_key[10]}")
        else:
            print(f"  No X keys found for this game at all!")

if found == len(locked_keys):
    print("\n✅ ALL locked keys found in X - locking mechanism will work correctly!")
else:
    print(f"\n⚠️ WARNING: {not_found} locked keys not in X - these games won't be locked!")
