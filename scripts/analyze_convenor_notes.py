"""Analyze checkpoint run_217/simple_solve_intermediate_110 against convenor notes."""
import pickle, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.storage import DrawStorage

# Load checkpoint
with open('checkpoints/run_217/simple_solve_intermediate_110/solution.pkl', 'rb') as f:
    solution = pickle.load(f)

draw = DrawStorage.from_X_solution(solution, description='run_217 intermediate 110')

print(f"Total games: {draw.num_games}, Total weeks: {draw.num_weeks}")
print(f"="*100)

# Helper functions
LOCKED_WEEKS = {1, 2, 4, 5, 6}

def locked_label(week):
    return " [LOCKED]" if week in LOCKED_WEEKS else ""

def print_game(g):
    print(f"  {g.game_id} | {g.team1} vs {g.team2} | {g.grade} | {g.date} {g.day} {g.time} | {g.field_name} @ {g.field_location} | Week {g.week}{locked_label(g.week)}")

def games_on_date(date_str):
    return [g for g in draw.games if g.date == date_str]

def games_in_week(week):
    return [g for g in draw.games if g.week == week]

def team_in_game(g, team_substr):
    return team_substr.lower() in g.team1.lower() or team_substr.lower() in g.team2.lower()

def games_for_team(team_substr, grade=None):
    results = []
    for g in draw.games:
        if team_in_game(g, team_substr):
            if grade is None or g.grade.lower() == grade.lower():
                results.append(g)
    return sorted(results, key=lambda g: (g.week, g.date, g.time))

def games_between(team1_sub, team2_sub, grade=None):
    results = []
    for g in draw.games:
        if (team_in_game(g, team1_sub) and team_in_game(g, team2_sub)):
            if grade is None or g.grade.lower() == grade.lower():
                results.append(g)
    return sorted(results, key=lambda g: (g.week, g.date, g.time))

# Print all unique dates and weeks for reference
dates_weeks = sorted(set((g.date, g.week, g.day) for g in draw.games))
print("\nAll dates in draw:")
for d, w, day in dates_weeks:
    print(f"  {d} ({day}) = Week {w}{locked_label(w)}")

# Print all unique teams
teams = sorted(set(g.team1 for g in draw.games) | set(g.team2 for g in draw.games))
print(f"\nAll teams ({len(teams)}):")
for t in teams:
    print(f"  {t}")

print("\n" + "="*100)

# ============== A ==============
print("\n[A] Short Week May 24 - A League Grand Final (late games blocked)")
print("-"*60)
may24_games = games_on_date('2026-05-24')
if may24_games:
    for g in sorted(may24_games, key=lambda g: g.time):
        print_game(g)
    late_times = [g for g in may24_games if g.time >= '16:00' and 'Newcastle' in g.field_location]
    if late_times:
        print("  ** FAIL: Late games (4pm+) found at NIHC on May 24!")
    else:
        print("  ** PASS: No late NIHC games on May 24")
else:
    print("  No games on May 24")

# ============== B ==============
print("\n[B] Norths 80th weekend - Wests vs Norths PHL on Jun 12 Friday")
print("-"*60)
jun12_games = games_on_date('2026-06-12')
jun14_games = games_on_date('2026-06-14')
print("Jun 12 (Friday) games:")
if jun12_games:
    for g in sorted(jun12_games, key=lambda g: g.time):
        print_game(g)
else:
    print("  No games on Jun 12")

norths_wests_phl = [g for g in jun12_games if team_in_game(g, 'norths') and team_in_game(g, 'wests') and g.grade == 'PHL']
if norths_wests_phl:
    print("  ** PASS: Norths vs Wests PHL on Friday Jun 12")
else:
    print("  ** FAIL: No Norths vs Wests PHL on Friday Jun 12")
    # Check if it's elsewhere
    nw_phl = games_between('norths', 'wests', 'PHL')
    print("  All Norths vs Wests PHL games:")
    for g in nw_phl:
        print_game(g)

print("\nJun 14 (Sunday) Norths games:")
norths_jun14 = [g for g in jun14_games if team_in_game(g, 'norths')]
for g in sorted(norths_jun14, key=lambda g: (g.grade, g.time)):
    print_game(g)

# ============== C ==============
print("\n[C] Crusaders club day Jun 14 - back-to-back on same field")
print("-"*60)
crus_jun14 = [g for g in jun14_games if team_in_game(g, 'crusader')]
print("Crusaders games on Jun 14:")
for g in sorted(crus_jun14, key=lambda g: (g.field_name, g.time)):
    print_game(g)
# Check for back-to-back same field
from collections import defaultdict
field_games = defaultdict(list)
for g in crus_jun14:
    field_games[(g.field_name, g.field_location)].append(g)
for (fn, fl), gg in field_games.items():
    if len(gg) >= 2:
        print(f"  ** Back-to-back on {fn} @ {fl}: {len(gg)} games")
    else:
        print(f"  Only 1 game on {fn} @ {fl}")

# ============== D ==============
print("\n[D] Norths PHL vs Wests on their (80th) weekend - same as B")
print("-"*60)
print("  See check B above.")

# ============== E ==============
print("\n[E] Week 9 (May 17, Masters SC) - PHL games and times")
print("-"*60)
week9_games = games_in_week(9)
phl_week9 = [g for g in week9_games if g.grade == 'PHL']
print("PHL games in week 9:")
if phl_week9:
    for g in sorted(phl_week9, key=lambda g: (g.date, g.time)):
        print_game(g)
else:
    print("  No PHL games in week 9")
print("All week 9 games:")
for g in sorted(week9_games, key=lambda g: (g.date, g.time)):
    print_game(g)

# ============== F ==============
print("\n[F] PHL games on State Champ weekends (May 17, Jun 21)")
print("-"*60)
for date_str, label in [('2026-05-17', 'May 17'), ('2026-06-21', 'Jun 21')]:
    phl_games = [g for g in games_on_date(date_str) if g.grade == 'PHL']
    print(f"PHL games on {label}:")
    if phl_games:
        for g in sorted(phl_games, key=lambda g: g.time):
            print_game(g)
    else:
        print(f"  No PHL games on {label}")
    # Also check Friday before
    # May 15 Friday, Jun 19 Friday
    fri_map = {'2026-05-17': '2026-05-15', '2026-06-21': '2026-06-19'}
    fri = fri_map[date_str]
    phl_fri = [g for g in games_on_date(fri) if g.grade == 'PHL']
    print(f"PHL games on Friday {fri}:")
    if phl_fri:
        for g in phl_fri:
            print_game(g)
    else:
        print(f"  No PHL games on {fri}")

# ============== G ==============
print("\n[G] Week 22 (Aug 16) - no games before 12:30")
print("-"*60)
aug16_games = games_on_date('2026-08-16')
week22_games = games_in_week(22)
print("All week 22 games:")
for g in sorted(week22_games, key=lambda g: (g.date, g.time)):
    print_game(g)
early_games = [g for g in week22_games if g.time < '12:30' and 'Newcastle' in g.field_location]
if early_games:
    print("  ** FAIL: Games before 12:30 at NIHC on week 22!")
else:
    print("  ** PASS: No early NIHC games in week 22")

# ============== H ==============
print("\n[H] Week 19 - Gosford vs Wests PHL on Friday night")
print("-"*60)
week19_games = games_in_week(19)
print("Week 19 PHL games:")
phl_w19 = [g for g in week19_games if g.grade == 'PHL']
for g in sorted(phl_w19, key=lambda g: (g.date, g.time)):
    print_game(g)
gos_wests_phl = [g for g in phl_w19 if team_in_game(g, 'gosford') and team_in_game(g, 'wests')]
if gos_wests_phl:
    for g in gos_wests_phl:
        if g.day == 'Friday':
            print("  ** PASS: Gosford vs Wests PHL on Friday night")
        else:
            print(f"  ** FAIL: Gosford vs Wests PHL is on {g.day} not Friday")
else:
    print("  ** FAIL: No Gosford vs Wests PHL in week 19")
    print("  All Gosford vs Wests PHL games:")
    for g in games_between('gosford', 'wests', 'PHL'):
        print_game(g)

# ============== I ==============
print("\n[I] Double-ups: same opponent in consecutive weeks (PHL & 2nd)")
print("-"*60)
for grade_name in ['PHL', '2nd']:
    grade_games = [g for g in draw.games if g.grade == grade_name]
    # Build matchup-by-week
    matchup_weeks = defaultdict(list)
    for g in grade_games:
        pair = tuple(sorted([g.team1, g.team2]))
        matchup_weeks[pair].append(g.week)

    for pair, weeks in sorted(matchup_weeks.items()):
        weeks_sorted = sorted(set(weeks))
        for i in range(len(weeks_sorted) - 1):
            if weeks_sorted[i+1] - weeks_sorted[i] == 1:
                print(f"  {grade_name}: {pair[0]} vs {pair[1]} in consecutive weeks {weeks_sorted[i]} & {weeks_sorted[i+1]}")
                # Show the actual games
                for g in grade_games:
                    p = tuple(sorted([g.team1, g.team2]))
                    if p == pair and g.week in (weeks_sorted[i], weeks_sorted[i+1]):
                        print_game(g)

# ============== J ==============
print("\n[J] West Field Finisher - last timeslot of day at NIHC on WF")
print("-"*60)
nihc_games = [g for g in draw.games if 'Newcastle' in g.field_location]
# Group by date
by_date = defaultdict(list)
for g in nihc_games:
    by_date[g.date].append(g)

for date_str in sorted(by_date.keys()):
    gg = by_date[date_str]
    # Find last timeslot
    max_time = max(g.time for g in gg)
    last_slot_games = [g for g in gg if g.time == max_time]
    fields_used = set(g.field_name for g in last_slot_games)
    if len(fields_used) == 1:
        field = fields_used.pop()
        week = last_slot_games[0].week
        status = "OK" if field == 'WF' else "** FAIL (on EF)"
        print(f"  {date_str} (Week {week}{locked_label(week)}): Last slot {max_time} - single field: {field} - {status}")
        if field != 'WF':
            for g in last_slot_games:
                print_game(g)
    # If multiple fields at last slot, that's fine

# ============== K ==============
print("\n[K] Week 14 (Jun 19-21) - U16 SC: no non-PHL Fri/Sat, PHL Sunday afternoon")
print("-"*60)
print("Jun 19 (Friday) games:")
jun19 = games_on_date('2026-06-19')
for g in sorted(jun19, key=lambda g: g.time):
    print_game(g)
non_phl_fri = [g for g in jun19 if g.grade != 'PHL']
if non_phl_fri:
    print("  ** FAIL: Non-PHL games on Friday Jun 19")
else:
    print("  ** PASS: Only PHL (or none) on Friday Jun 19")

print("Jun 20 (Saturday) games:")
jun20 = games_on_date('2026-06-20')
if jun20:
    for g in jun20:
        print_game(g)
    print("  ** FAIL: Games on Saturday Jun 20")
else:
    print("  No games - PASS")

print("Jun 21 (Sunday) games:")
jun21 = games_on_date('2026-06-21')
for g in sorted(jun21, key=lambda g: (g.grade, g.time)):
    print_game(g)
phl_sun = [g for g in jun21 if g.grade == 'PHL']
if phl_sun:
    print(f"  PHL on Sunday: {len(phl_sun)} games")
else:
    print("  No PHL on Sunday Jun 21")

# ============== L ==============
print("\n[L] Week 9 - Sunday PHL times (Masters SC, times TBD)")
print("-"*60)
may17_phl = [g for g in games_on_date('2026-05-17') if g.grade == 'PHL']
if may17_phl:
    print("PHL games on Sunday May 17:")
    for g in sorted(may17_phl, key=lambda g: g.time):
        print_game(g)
else:
    print("  No PHL games on Sunday May 17")
# Also check Friday May 15
may15_phl = [g for g in games_on_date('2026-05-15') if g.grade == 'PHL']
if may15_phl:
    print("PHL games on Friday May 15:")
    for g in may15_phl:
        print_game(g)

# ============== M ==============
print("\n[M] Week 13 - Norths 80th (Jun 12-14): PHL Fri, lower grades Sun")
print("-"*60)
print("Already covered in B. Summary:")
print(f"  Friday Jun 12: {len(jun12_games)} games")
print(f"  Sunday Jun 14: {len(jun14_games)} games")
# Check lower grades on Sunday
lower_jun14 = [g for g in jun14_games if g.grade not in ('PHL',)]
print(f"  Lower grade games on Sunday Jun 14: {len(lower_jun14)}")
for g in sorted(lower_jun14, key=lambda g: (g.grade, g.time)):
    print_game(g)

# ============== N ==============
print("\n[N] Gosford not playing weekend after Opens State Championships (Jun 21)")
print("-"*60)
gos_jun21 = [g for g in jun21 if team_in_game(g, 'gosford')]
if gos_jun21:
    print("  ** FAIL: Gosford has games on Jun 21:")
    for g in gos_jun21:
        print_game(g)
else:
    print("  ** PASS: No Gosford games on Jun 21")
# Also check Jun 19 Friday
gos_jun19 = [g for g in jun19 if team_in_game(g, 'gosford')]
if gos_jun19:
    print("  Gosford on Friday Jun 19:")
    for g in gos_jun19:
        print_game(g)

# ============== O ==============
print("\n[O] Souths vs Tigers PHL - 3 times in 4 weeks (Week 9, 10, 13)?")
print("-"*60)
st_phl = games_between('souths', 'tigers', 'PHL')
print("All Souths vs Tigers PHL games:")
for g in st_phl:
    print_game(g)
weeks = sorted(set(g.week for g in st_phl))
print(f"  Weeks: {weeks}")
# Check for clustering
for i in range(len(weeks)):
    for j in range(i+1, len(weeks)):
        span = weeks[j] - weeks[i] + 1
        count = j - i + 1
        if count >= 3 and span <= 5:
            print(f"  ** WARNING: {count} meetings in {span}-week span (weeks {weeks[i]}-{weeks[j]})")

# ============== P ==============
print("\n[P] Week 8 (May 10) - Red & Blue Derby: Norths vs Souths PHL + lower grades")
print("-"*60)
may10_games = games_on_date('2026-05-10')
ns_may10 = [g for g in may10_games if team_in_game(g, 'norths') and team_in_game(g, 'souths')]
print("Norths vs Souths on May 10:")
if ns_may10:
    for g in sorted(ns_may10, key=lambda g: (g.grade, g.time)):
        print_game(g)
else:
    print("  No Norths vs Souths on May 10")
    # Where is it?
    ns_phl = games_between('norths', 'souths', 'PHL')
    print("  All Norths vs Souths PHL games:")
    for g in ns_phl:
        print_game(g)

# Also check Friday May 8
may8_games = games_on_date('2026-05-08')
ns_may8 = [g for g in may8_games if team_in_game(g, 'norths') and team_in_game(g, 'souths')]
if ns_may8:
    print("Norths vs Souths on Friday May 8:")
    for g in ns_may8:
        print_game(g)

# ============== Q ==============
print("\n[Q] Week 10 (May 24) - No Souths PHL due to U18 State Champs")
print("-"*60)
souths_may24 = [g for g in may24_games if team_in_game(g, 'souths') and g.grade == 'PHL']
# Also check Friday May 22
may22_games = games_on_date('2026-05-22')
souths_may22 = [g for g in may22_games if team_in_game(g, 'souths') and g.grade in ('PHL', '2nd')]
print("Souths PHL on May 24 (Sunday):")
if souths_may24:
    for g in souths_may24:
        print_game(g)
    print("  ** FAIL: Souths PHL playing on May 24")
else:
    print("  ** PASS: No Souths PHL on May 24")
print("Souths PHL/2nd on May 22 (Friday):")
if souths_may22:
    for g in souths_may22:
        print_game(g)
    print("  ** FAIL: Souths PHL/2nd playing on May 22")
else:
    print("  ** PASS: No Souths PHL/2nd on May 22")

# ============== R ==============
print("\n[R] Week 11 (May 31) - 5:30pm WF match")
print("-"*60)
may31_games = games_on_date('2026-05-31')
wf_530 = [g for g in may31_games if g.time == '17:30' and g.field_name == 'WF']
print("All May 31 games:")
for g in sorted(may31_games, key=lambda g: (g.time, g.field_name)):
    print_game(g)
if wf_530:
    print("  5:30pm WF games found - convenor wants to move to earlier slot")
    for g in wf_530:
        print_game(g)
else:
    print("  No 5:30pm WF games on May 31")
# Also check any 5:30 games
all_530 = [g for g in may31_games if g.time == '17:30']
if all_530:
    print("  All 5:30pm games on May 31:")
    for g in all_530:
        print_game(g)

# ============== S ==============
print("\n[S] Week 13 (Jun 14) - Moane/Stanbury: Maitland vs Souths PHL")
print("-"*60)
ms_jun14 = [g for g in jun14_games if team_in_game(g, 'maitland') and team_in_game(g, 'souths') and g.grade == 'PHL']
if ms_jun14:
    print("  Maitland vs Souths PHL on Jun 14:")
    for g in ms_jun14:
        print_game(g)
    print("  ** PASS: Moane/Stanbury on Jun 14")
else:
    print("  ** FAIL: No Maitland vs Souths PHL on Jun 14")
    ms_phl = games_between('maitland', 'souths', 'PHL')
    print("  All Maitland vs Souths PHL games:")
    for g in ms_phl:
        print_game(g)

# ============== T ==============
print("\n[T] Week 15 (Jun 28) - Tigers across grades")
print("-"*60)
jun28_games = games_on_date('2026-06-28')
# Also check Friday Jun 26
jun26_games = games_on_date('2026-06-26')
tigers_jun28 = [g for g in jun28_games if team_in_game(g, 'tigers')]
tigers_jun26 = [g for g in jun26_games if team_in_game(g, 'tigers')]
print("Tigers games Jun 28 (Sunday):")
for g in sorted(tigers_jun28, key=lambda g: (g.grade, g.time)):
    print_game(g)
if tigers_jun26:
    print("Tigers games Jun 26 (Friday):")
    for g in tigers_jun26:
        print_game(g)
# Check specific issue: PHL in Maitland, 3rd at 3pm, 2nd in Newcastle 4pm, 4th at 1pm
print("  Issue noted: PHL in Maitland with 3rd at 3pm but 2nd in Newcastle at 4pm and 4th at 1pm")

# ============== U ==============
print("\n[U] Week 16 (Jul 5) - Crusaders in Maitland issues")
print("-"*60)
jul5_games = games_on_date('2026-07-05')
jul3_games = games_on_date('2026-07-03')
crus_jul5 = [g for g in jul5_games if team_in_game(g, 'crusader')]
print("Crusaders games Jul 5 (Sunday):")
for g in sorted(crus_jul5, key=lambda g: (g.grade, g.time)):
    print_game(g)
if jul3_games:
    crus_jul3 = [g for g in jul3_games if team_in_game(g, 'crusader')]
    if crus_jul3:
        print("Crusaders games Jul 3 (Friday):")
        for g in crus_jul3:
            print_game(g)

# ============== V ==============
print("\n[V] Week 19 (Jul 24-26) - PHL Friday night confirmation")
print("-"*60)
jul24_games = games_on_date('2026-07-24')
print("PHL games Jul 24 (Friday):")
phl_jul24 = [g for g in jul24_games if g.grade == 'PHL']
if phl_jul24:
    for g in phl_jul24:
        print_game(g)
else:
    print("  No PHL games on Friday Jul 24")
print("All week 19 PHL games:")
for g in sorted(phl_w19, key=lambda g: (g.date, g.time)):
    print_game(g)

# ============== W ==============
print("\n[W] Week 20 (Aug 2) - Tigers vs Souths PHL & 2nd to Taree")
print("-"*60)
aug2_games = games_on_date('2026-08-02')
aug2_fri = games_on_date('2026-07-31')
ts_aug2 = [g for g in aug2_games if team_in_game(g, 'tigers') and team_in_game(g, 'souths')]
print("Tigers vs Souths on Aug 2:")
if ts_aug2:
    for g in sorted(ts_aug2, key=lambda g: (g.grade, g.time)):
        print_game(g)
    print("  Note: Convenor wants these to go to Taree (external venue, not in config)")
else:
    print("  No Tigers vs Souths on Aug 2")
    # Where are they?
    ts_phl = games_between('tigers', 'souths', 'PHL')
    print("  All Tigers vs Souths PHL:")
    for g in ts_phl:
        print_game(g)

# ============== X ==============
print("\n[X] Week 22 (Aug 16) - no stand-alone lower grade in Maitland")
print("-"*60)
week22_mait = [g for g in week22_games if 'Maitland' in g.field_location]
if week22_mait:
    print("  Maitland Park games in week 22:")
    for g in sorted(week22_mait, key=lambda g: (g.grade, g.time)):
        print_game(g)
    lower = [g for g in week22_mait if g.grade not in ('PHL', '2nd')]
    phl_2nd = [g for g in week22_mait if g.grade in ('PHL', '2nd')]
    if lower and not phl_2nd:
        print("  ** FAIL: Stand-alone lower grade games at Maitland (no PHL/2nd)")
    elif lower and phl_2nd:
        print("  ** OK: Lower grades at Maitland but PHL/2nd also present")
    else:
        print("  ** OK: Only PHL/2nd at Maitland")
else:
    print("  ** PASS: No games at Maitland Park in week 22")

# ============== Y ==============
print("\n[Y] Week 23 (Aug 21-23) - PHL Friday night at Central Coast")
print("-"*60)
aug21_games = games_on_date('2026-08-21')
aug23_games = games_on_date('2026-08-23')
phl_aug21 = [g for g in aug21_games if g.grade == 'PHL']
print("PHL games on Friday Aug 21:")
if phl_aug21:
    for g in phl_aug21:
        print_game(g)
    cc_phl = [g for g in phl_aug21 if 'Central Coast' in g.field_location or 'Coast' in g.field_location]
    if cc_phl:
        print("  ** PASS: PHL on Friday at Central Coast")
    else:
        print("  ** FAIL: PHL on Friday but NOT at Central Coast")
else:
    print("  No PHL on Friday Aug 21")
print("All week 23 PHL games:")
week23_phl = [g for g in games_in_week(23) if g.grade == 'PHL']
for g in sorted(week23_phl, key=lambda g: (g.date, g.time)):
    print_game(g)

# ============== Z ==============
print("\n[Z] Week 24 (Aug 30) - Norths 3rd & 4th different locations")
print("-"*60)
aug30_games = games_on_date('2026-08-30')
aug28_games = games_on_date('2026-08-28')
norths_aug30 = [g for g in aug30_games if team_in_game(g, 'norths')]
print("Norths games Aug 30:")
for g in sorted(norths_aug30, key=lambda g: (g.grade, g.time)):
    print_game(g)
n3 = [g for g in norths_aug30 if g.grade == '3rd']
n4 = [g for g in norths_aug30 if g.grade == '4th']
if n3 and n4:
    n3_locs = set(g.field_location for g in n3)
    n4_locs = set(g.field_location for g in n4)
    if n3_locs != n4_locs:
        print(f"  ** WARNING: 3rd at {n3_locs}, 4th at {n4_locs} - DIFFERENT locations")
    else:
        print(f"  ** OK: 3rd and 4th at same location(s): {n3_locs}")

# ============== AA ==============
print("\n[AA] Wallis/Markwell cup - suggest Jul 19 (Friday night PHL)")
print("-"*60)
jul17_games = games_on_date('2026-07-17')
jul19_games = games_on_date('2026-07-19')
phl_jul17 = [g for g in jul17_games if g.grade == 'PHL']
print("PHL games on Friday Jul 17:")
if phl_jul17:
    for g in phl_jul17:
        print_game(g)
else:
    print("  No PHL games on Friday Jul 17")
print("PHL games on Sunday Jul 19:")
phl_jul19 = [g for g in jul19_games if g.grade == 'PHL']
for g in sorted(phl_jul19, key=lambda g: g.time):
    print_game(g)

# ============== BB ==============
print("\n[BB] Week 6 (Apr 26) - ANZAC Sunday: no PHL")
print("-"*60)
apr26_games = games_on_date('2026-04-26')
phl_apr26 = [g for g in apr26_games if g.grade == 'PHL']
print("PHL games on Apr 26:")
if phl_apr26:
    for g in phl_apr26:
        print_game(g)
    print("  ** FAIL: PHL games on ANZAC Sunday")
else:
    print("  ** PASS: No PHL games on Apr 26")
# Also check Friday Apr 24
apr24_games = games_on_date('2026-04-24')
phl_apr24 = [g for g in apr24_games if g.grade == 'PHL']
if phl_apr24:
    print("PHL games on Friday Apr 24:")
    for g in phl_apr24:
        print_game(g)

# ============== CC ==============
print("\n[CC] Gosford vs Wests PHL Friday night week 19 at Central Coast")
print("-"*60)
print("See check H. Specifically checking Central Coast venue:")
gw_w19_fri = [g for g in jul24_games if team_in_game(g, 'gosford') and team_in_game(g, 'wests') and g.grade == 'PHL']
if gw_w19_fri:
    for g in gw_w19_fri:
        print_game(g)
        if 'Central Coast' in g.field_location or 'Coast' in g.field_location:
            print("  ** PASS: At Central Coast")
        else:
            print(f"  ** FAIL: At {g.field_location}, not Central Coast")
else:
    print("  ** FAIL: No Gosford vs Wests PHL on Friday Jul 24")
    # Check all week 19
    gw_w19 = [g for g in week19_games if team_in_game(g, 'gosford') and team_in_game(g, 'wests') and g.grade == 'PHL']
    if gw_w19:
        print("  Found in week 19 but not on Friday:")
        for g in gw_w19:
            print_game(g)

print("\n" + "="*100)
print("ANALYSIS COMPLETE")
