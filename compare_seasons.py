#!/usr/bin/env python3
"""Compare 2025 vs 2026 season configurations."""

from config import load_season_data
from collections import Counter

d25 = load_season_data(2025)
d26 = load_season_data(2026)

t25 = Counter([t.grade for t in d25['teams']])
t26 = Counter([t.grade for t in d26['teams']])

print("=" * 60)
print("SEASON COMPARISON: 2025 vs 2026")
print("=" * 60)

print("\n=== Teams by Grade ===")
print(f"{'Grade':<10} {'2025':>6} {'2026':>6} {'Diff':>6}")
print("-" * 30)
total_25 = total_26 = 0
for g in ['PHL', '2nd', '3rd', '4th', '5th', '6th']:
    c25 = t25.get(g, 0)
    c26 = t26.get(g, 0)
    total_25 += c25
    total_26 += c26
    diff = c26 - c25
    diff_str = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "0"
    print(f"{g:<10} {c25:>6} {c26:>6} {diff_str:>6}")
print("-" * 30)
print(f"{'TOTAL':<10} {total_25:>6} {total_26:>6} {total_26 - total_25:>+6}")

print("\n=== Games by Grade ===")
# Calculate games per grade: n*(n-1) for round-robin
print(f"{'Grade':<10} {'2025 Games':>12} {'2026 Games':>12} {'Diff':>8}")
print("-" * 45)
total_g25 = total_g26 = 0
for g in ['PHL', '2nd', '3rd', '4th', '5th', '6th']:
    n25 = t25.get(g, 0)
    n26 = t26.get(g, 0)
    g25 = n25 * (n25 - 1) if n25 > 1 else 0  # full round-robin games
    g26 = n26 * (n26 - 1) if n26 > 1 else 0
    total_g25 += g25
    total_g26 += g26
    diff = g26 - g25
    diff_str = f"+{diff}" if diff > 0 else str(diff) if diff < 0 else "0"
    print(f"{g:<10} {g25:>12} {g26:>12} {diff_str:>8}")
print("-" * 45)
print(f"{'TOTAL':<10} {total_g25:>12} {total_g26:>12} {total_g26 - total_g25:>+8}")

print("\n=== Timeslots ===")
print(f"2025: {len(d25['timeslots'])} timeslots")
print(f"2026: {len(d26['timeslots'])} timeslots")
print(f"Diff: {len(d26['timeslots']) - len(d25['timeslots']):+d}")

print("\n=== Rounds ===")
print(f"2025: {d25.get('num_rounds', {}).get('max', '?')} max rounds")
print(f"2026: {d26.get('num_rounds', {}).get('max', '?')} max rounds")

print("\n=== Variable Estimation ===")
# Variables = games * timeslots (roughly, ignoring filtering)
v25_est = total_g25 * len(d25['timeslots'])
v26_est = total_g26 * len(d26['timeslots'])
print(f"2025: ~{v25_est:,} potential variables (games × slots)")
print(f"2026: ~{v26_est:,} potential variables (games × slots)")
print(f"Diff: {v26_est - v25_est:+,}")

print("\n=== Clubs with Team Changes ===")
clubs_25 = Counter([t.club.name for t in d25['teams']])
clubs_26 = Counter([t.club.name for t in d26['teams']])
all_clubs = sorted(set(clubs_25.keys()) | set(clubs_26.keys()))
for club in all_clubs:
    c25 = clubs_25.get(club, 0)
    c26 = clubs_26.get(club, 0)
    if c25 != c26:
        print(f"  {club}: {c25} -> {c26} teams ({c26-c25:+d})")

print()
