#!/usr/bin/env python3
"""Analyze PHL team feasibility: can each team reach 20 games given constraints?"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import load_season_data
from utils import max_games_per_grade

data = load_season_data(2026)
teams = data['teams']
phl_teams = [t for t in teams if t.grade == 'PHL']
print(f"PHL teams ({len(phl_teams)}):")
for t in phl_teams:
    print(f"  {t.name} (club: {t.club})")

num_rounds = data['num_rounds']
print(f"\nnum_rounds: {num_rounds}")
print(f"PHL required games: {num_rounds.get('PHL', '???')}")

# Load the locked draw to see what games are locked
draw_path = 'draws/2026/current.json'
with open(draw_path) as f:
    draw = json.load(f)

locked_weeks = {1, 2, 4, 5, 6}

# Count locked PHL games per team
locked_games_per_team = {}
for t in phl_teams:
    locked_games_per_team[t.name] = 0

for g in draw['games']:
    if g['grade'] == 'PHL' and g['week'] in locked_weeks:
        for t in [g['team1'], g['team2']]:
            if t in locked_games_per_team:
                locked_games_per_team[t] = locked_games_per_team.get(t, 0) + 1

print("\nLocked PHL games per team (weeks 1,2,4,5,6):")
for team, count in sorted(locked_games_per_team.items()):
    print(f"  {team}: {count} locked games")

# Check which weeks have PHL games in the locked draw
phl_weeks_in_draw = {}
for g in draw['games']:
    if g['grade'] == 'PHL':
        week = g['week']
        if week not in phl_weeks_in_draw:
            phl_weeks_in_draw[week] = []
        phl_weeks_in_draw[week].append(f"{g['team1']} vs {g['team2']} ({g['day']} {g['date']})")

print(f"\nPHL games in draw by week:")
for wk in sorted(phl_weeks_in_draw.keys()):
    marker = " [LOCKED]" if wk in locked_weeks else ""
    print(f"  Week {wk}{marker}: {len(phl_weeks_in_draw[wk])} games")
    for g in phl_weeks_in_draw[wk]:
        print(f"    {g}")

# Identify all playable weeks from timeslots
timeslots = data['timeslots']
all_weeks = sorted(set(t.week for t in timeslots if t.day))
print(f"\nAll playable weeks: {all_weeks}")
print(f"Total playable weeks: {len(all_weeks)}")

unlocked_weeks = sorted(set(all_weeks) - locked_weeks)
print(f"Unlocked weeks: {unlocked_weeks}")
print(f"Total unlocked weeks: {len(unlocked_weeks)}")

# Check which weeks have PHL timeslots (Friday and/or Sunday)
phl_game_times = data.get('phl_game_times', {})
print(f"\nPHL timeslots by week (from generated timeslots):")

# Gather PHL-eligible timeslots per week
phl_timeslots_by_week = {}
for t in timeslots:
    if not t.day:
        continue
    # Check if this timeslot matches PHL_GAME_TIMES
    venue = t.field.location
    field = t.field.name
    day = t.day
    time_str = t.time

    if venue in phl_game_times:
        if field in phl_game_times[venue]:
            if day in phl_game_times[venue][field]:
                times_list = phl_game_times[venue][field][day]
                # Compare times
                from datetime import time as tm
                if isinstance(time_str, str):
                    from datetime import datetime as dt
                    t_time = dt.strptime(time_str, '%H:%M').time()
                else:
                    t_time = time_str
                if t_time in times_list:
                    wk = t.week
                    if wk not in phl_timeslots_by_week:
                        phl_timeslots_by_week[wk] = {'Friday': [], 'Sunday': []}
                    phl_timeslots_by_week[wk].setdefault(day, []).append(
                        f"{venue}:{field} {time_str}"
                    )

for wk in sorted(phl_timeslots_by_week.keys()):
    marker = " [LOCKED]" if wk in locked_weeks else ""
    fri = len(phl_timeslots_by_week[wk].get('Friday', []))
    sun = len(phl_timeslots_by_week[wk].get('Sunday', []))
    print(f"  Week {wk}{marker}: {fri} Fri slots, {sun} Sun slots")

# Now analyze blocked games for PHL
blocked_games = data.get('blocked_games', [])
print("\nBLOCKED_GAMES affecting PHL:")
for bg in blocked_games:
    grade = bg.get('grade', '')
    grades = bg.get('grades', [])
    if grade == 'PHL' or 'PHL' in grades:
        print(f"  {bg.get('description', bg)}")
    elif bg.get('club') in ['Gosford', 'Souths', 'Maitland'] and not grade:
        # Club-level blocks affect PHL too
        print(f"  [Club-level] {bg.get('description', bg)}")

# FORCED_GAMES with lesse constraint (limiting PHL games)
forced_games = data.get('forced_games', [])
print("\nFORCED_GAMES with 'lesse' constraint affecting PHL:")
for fg in forced_games:
    if fg.get('constraint') == 'lesse' and fg.get('grade') == 'PHL':
        print(f"  {fg.get('description', fg)}")

# Now calculate: for each PHL team, how many unlocked weeks can they play?
# A team can't play in a week if:
# 1. All their PHL vars in that week are blocked/eliminated
# 2. They're blocked by BLOCKED_GAMES
# 3. NoDoubleBooking means they can only play once per week (Fri OR Sun, not both)

# Key insight: In weeks 9 (May 15/17) and 14 (Jun 19/21):
# - Sunday at NIHC: max 1 PHL game (FORCED_GAMES lesse)
# - Sunday at Gosford: blocked (whole_day May 17, Jun 21 at Central Coast)
# - Sunday at Maitland: open for PHL
# - Friday at Gosford: forced game exists
# - A team playing Friday can't also play Sunday (NoDoubleBooking per week)

# For Gosford PHL specifically:
# - They play 8 Friday games at Gosford
# - On those Fridays, they can't play Sunday (NoDoubleBooking)
# - Plus blocked Jun 21 (recovery weekend)
# - Plus other blocked dates

print("\n" + "="*60)
print("FEASIBILITY ANALYSIS: Can each PHL team reach required games?")
print("="*60)

required = num_rounds.get('PHL', 20)
print(f"Required PHL games per team: {required}")

# For each team, count available weeks
# Method: locked_games + max_possible_from_unlocked >= required?
for team in phl_teams:
    locked = locked_games_per_team.get(team.name, 0)
    needed = required - locked

    # Count how many unlocked weeks this team can play in
    # Check blocked games
    blocked_dates = set()
    blocked_fridays = set()
    for bg in blocked_games:
        bg_grade = bg.get('grade', '')
        bg_grades = bg.get('grades', [])
        bg_club = bg.get('club', '')
        bg_teams = bg.get('teams', [])
        bg_date = bg.get('date', '')
        bg_day = bg.get('day', '')

        # Does this block affect this team?
        affects_team = False
        if bg_club and bg_club == team.club:
            if not bg_grade and not bg_grades:
                affects_team = True  # Club-wide block
            elif bg_grade == 'PHL' or 'PHL' in bg_grades:
                affects_team = True
        if bg_teams and team.name in bg_teams:
            affects_team = True
        # Grade-only blocks (no club/team specified) affect ALL teams in that grade
        if not bg_club and not bg_teams and (bg_grade == 'PHL' or 'PHL' in bg_grades):
            affects_team = True

        if affects_team and bg_date:
            if bg_day == 'Friday':
                blocked_fridays.add(bg_date)
            elif bg_day == 'Sunday' or not bg_day:
                blocked_dates.add(bg_date)

    # Map dates to weeks
    date_to_week = {}
    for t in timeslots:
        if t.day and t.date:
            date_str = str(t.date) if not isinstance(t.date, str) else t.date
            date_to_week[date_str] = t.week

    # Check Gosford's Friday games (they lose those Sundays due to NoDoubleBooking)
    friday_weeks = set()
    if team.club == 'Gosford':
        # Gosford plays every Gosford Friday game
        for fg in forced_games:
            if fg.get('grade') == 'PHL' and fg.get('day') == 'Friday' and \
               fg.get('field_location') == 'Central Coast Hockey Park':
                fg_date = fg.get('date', '')
                wk = date_to_week.get(fg_date)
                if wk and wk not in locked_weeks:
                    friday_weeks.add(wk)

    # Check: which teams play in Gosford Friday games? It's always Gosford + 1 opponent
    # The opponent also loses that Sunday
    # But we need to figure out who the opponent is -- we don't know yet for unlocked weeks
    # For now, note that in each Gosford Friday, one non-Gosford team also can't play Sunday

    # Count weeks where team is blocked on BOTH Friday and Sunday
    fully_blocked_weeks = set()
    for wk in unlocked_weeks:
        # Get dates for this week
        wk_dates = {}
        for t in timeslots:
            if t.week == wk and t.day:
                date_str = str(t.date) if not isinstance(t.date, str) else t.date
                wk_dates.setdefault(t.day, set()).add(date_str)

        # Check if Sunday is blocked for this team
        sunday_blocked = False
        for sd in wk_dates.get('Sunday', set()):
            if sd in blocked_dates:
                sunday_blocked = True

        friday_blocked = False
        for fd in wk_dates.get('Friday', set()):
            if fd in blocked_fridays:
                friday_blocked = True

        # If team plays Friday, they can't play Sunday (NoDoubleBooking)
        if wk in friday_weeks:
            # Gosford plays Friday, so Sunday is unavailable for them
            # But they DO get one game that week (the Friday game)
            # So this week still counts as "playable" for them
            pass

        if sunday_blocked and (friday_blocked or not wk_dates.get('Friday', set())):
            fully_blocked_weeks.add(wk)

    # For Gosford: Friday weeks count as playable (they play the Friday game)
    # But we need to check: in SC weekends (9, 14), does the Sunday limit matter?

    # Special analysis for weeks 9 and 14
    # Week 9: May 15 (Fri), May 17 (Sun) - Masters SC
    # Week 14: Jun 19 (Fri), Jun 21 (Sun) - U16 Girls SC

    available_unlocked = len(unlocked_weeks) - len(fully_blocked_weeks)

    # For Gosford: they play Friday in some weeks but can't play Sunday same week
    # However, they DO get a game that week (Friday counts)
    # The issue is whether having a Friday game = 1 game that week, not 0

    surplus = available_unlocked - needed

    print(f"\n{team.name} ({team.club}):")
    print(f"  Locked games: {locked}")
    print(f"  Needed from unlocked: {needed}")
    print(f"  Unlocked weeks: {len(unlocked_weeks)}")
    print(f"  Fully blocked weeks: {fully_blocked_weeks}")
    print(f"  Available unlocked weeks: {available_unlocked}")
    print(f"  Surplus: {surplus} {'OK' if surplus >= 0 else '*** INFEASIBLE ***'}")
    if friday_weeks:
        print(f"  Friday game weeks (Gosford): {sorted(friday_weeks)}")
    if blocked_dates:
        print(f"  Blocked dates (Sun): {sorted(blocked_dates)}")
    if blocked_fridays:
        print(f"  Blocked dates (Fri): {sorted(blocked_fridays)}")

# Special deep-dive: weeks 9 and 14 Sunday limits
print("\n" + "="*60)
print("DEEP DIVE: Weeks 9 and 14 (SC weekends)")
print("="*60)

for wk, desc in [(9, "Masters SC"), (14, "U16 Girls SC")]:
    print(f"\nWeek {wk} ({desc}):")
    # What PHL timeslots exist this week?
    if wk in phl_timeslots_by_week:
        fri = phl_timeslots_by_week[wk].get('Friday', [])
        sun = phl_timeslots_by_week[wk].get('Sunday', [])
        print(f"  Friday PHL slots: {len(fri)} -> {fri}")
        print(f"  Sunday PHL slots: {len(sun)} -> {sun}")
    else:
        print(f"  No PHL timeslots this week!")

    # FORCED_GAMES lesse constraint limits Sunday to 1 game at NIHC
    # But what about Maitland Sunday? Is that also limited?
    print(f"  Sunday at NIHC: max 1 PHL game (lesse constraint)")
    print(f"  Sunday at Gosford: BLOCKED (whole_day)")
    print(f"  Sunday at Maitland: check...")

    # Check if Maitland Sunday is available
    for t in timeslots:
        if t.week == wk and t.day == 'Sunday' and t.field.location == 'Maitland Park':
            print(f"    Maitland slot: {t.time} {t.field.name}")

# The key question: how many PHL games can be played on Sunday in weeks 9 and 14?
# If max 1 at NIHC + any at Maitland, that's potentially 2 games
# But with 6 teams (3 games needed), 1 game = 2 teams play, 4 teams have bye
# With Friday games: Gosford + opponent play Friday, so 2 more teams covered
# That leaves 2 teams with no game in that week

print("\n" + "="*60)
print("SUMMARY: PHL scheduling capacity per team")
print("="*60)
print(f"6 PHL teams, 3 games per Sunday normally")
print(f"Weeks 9 & 14: limited Sunday + Friday at Gosford")
print(f"  Friday: 1 game (Gosford + opponent) = 2 teams play")
print(f"  Sunday NIHC: max 1 game = 2 teams play")
print(f"  Sunday Maitland: check if available")
print(f"  = max 4-6 teams play per SC weekend")
print(f"  = 0-2 teams have bye that week")
