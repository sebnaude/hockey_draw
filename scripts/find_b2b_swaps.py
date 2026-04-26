"""Find feasible swaps to fix remaining 5th grade b2b byes, allowing spacing violations."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from collections import defaultdict
from config.season_2026 import BLOCKED_GAMES

with open('draws/2026/current.json') as f:
    draw = json.load(f)

LOCKED_WEEKS = {1, 2, 3, 4, 5, 6}
SWAP_ONLY = {10, 22}

blocked_team_dates = defaultdict(set)
for bg in BLOCKED_GAMES:
    if 'teams' in bg and 'date' in bg:
        for t in bg['teams']:
            blocked_team_dates[t].add(bg['date'])

grade_weeks = defaultdict(set)
team_weeks = defaultdict(set)
week_date = {}

for g in draw['games']:
    grade_weeks[g['grade']].add(g['week'])
    team_weeks[(g['team1'], g['grade'])].add(g['week'])
    team_weeks[(g['team2'], g['grade'])].add(g['week'])
    if g['day'] == 'Sunday':
        week_date[g['week']] = g['date']

fifth_all = sorted(grade_weeks['5th'])


def check_b2b(tw):
    b2b = []
    for t, weeks in tw.items():
        byes = sorted(set(fifth_all) - weeks)
        for i in range(len(byes) - 1):
            idx1 = fifth_all.index(byes[i])
            idx2 = fifth_all.index(byes[i+1])
            if idx2 == idx1 + 1:
                b2b.append((t, byes[i], byes[i+1]))
    return b2b


def is_blocked(team, week):
    date = week_date.get(week)
    return date in blocked_team_dates.get(team, set())


# Current state (Moves 1&2 already applied to file)
tw = {}
for (t, g), weeks in team_weeks.items():
    if g == '5th':
        tw[t] = set(weeks)

current_b2b = check_b2b(tw)
print(f'Current 5th b2b: {current_b2b}')
print()


def try_swap(game_a_id, a_t1, a_t2, a_w, game_b_id, b_t1, b_t2, b_w):
    """Simulate swap: game_a moves from a_w to b_w, game_b moves from b_w to a_w."""
    new_tw = {t: set(w) for t, w in tw.items()}
    # Teams in game_a: was at a_w, now at b_w
    for t in [a_t1, a_t2]:
        new_tw[t] = (new_tw[t] - {a_w}) | {b_w}
    for t in [b_t1, b_t2]:
        new_tw[t] = (new_tw[t] - {b_w}) | {a_w}
    nb = check_b2b(new_tw)
    return nb, new_tw


print('=' * 80)
print('COLTS GREEN 5TH W13&15 — swap options')
print('=' * 80)

cg_games = [g for g in draw['games'] if g['grade'] == '5th'
            and g['week'] not in LOCKED_WEEKS
            and 'Colts Green 5th' in (g['team1'], g['team2'])]

w13_5th = [g for g in draw['games'] if g['grade'] == '5th' and g['week'] == 13]
w15_5th = [g for g in draw['games'] if g['grade'] == '5th' and g['week'] == 15]

# Try swap CG game <-> W13 game
print('\nSwap CG game <-> W13 game:')
for cg_g in cg_games:
    cg_opp = cg_g['team2'] if cg_g['team1'] == 'Colts Green 5th' else cg_g['team1']
    cg_w = cg_g['week']
    if cg_w in (13, 15):
        continue
    if cg_w in SWAP_ONLY:
        continue
    if is_blocked('Colts Green 5th', 13) or is_blocked(cg_opp, 13):
        continue

    for w13g in w13_5th:
        t1, t2 = w13g['team1'], w13g['team2']
        if cg_w in tw[t1] or cg_w in tw[t2]:
            continue
        if is_blocked(t1, cg_w) or is_blocked(t2, cg_w):
            continue
        if 13 in tw[cg_opp]:
            continue

        nb, new_tw = try_swap(
            cg_g['game_id'], 'Colts Green 5th', cg_opp, cg_w,
            w13g['game_id'], t1, t2, 13
        )
        delta = len(nb) - len(current_b2b)
        fixed = [x for x in current_b2b if x not in nb]
        created = [x for x in nb if x not in current_b2b]
        if ('Colts Green 5th', 13, 15) in fixed:
            print(f'  {cg_g["game_id"]} (CG vs {cg_opp} W{cg_w}) <-> {w13g["game_id"]} ({t1} vs {t2} W13)')
            print(f'    delta={delta}, fixes={fixed}, creates={created}')

print('\nSwap CG game <-> W15 game: (CG blocked at W15)')
# CG can play W15? Let's check blocked
if not is_blocked('Colts Green 5th', 15):
    print('  Actually not blocked — checking options')
else:
    print('  Skipped')


print()
print('=' * 80)
print('MAITLAND 5TH W22&23 — swap options')
print('=' * 80)

mait_games = [g for g in draw['games'] if g['grade'] == '5th'
              and g['week'] not in LOCKED_WEEKS
              and 'Maitland 5th' in (g['team1'], g['team2'])]

w22_5th = [g for g in draw['games'] if g['grade'] == '5th' and g['week'] == 22]
w23_5th = [g for g in draw['games'] if g['grade'] == '5th' and g['week'] == 23]

print('\nSwap Maitland game <-> W22 game:')
for mg in mait_games:
    mopp = mg['team2'] if mg['team1'] == 'Maitland 5th' else mg['team1']
    mw = mg['week']
    if mw in (22, 23):
        continue
    if mw in SWAP_ONLY:
        continue
    if is_blocked('Maitland 5th', 22) or is_blocked(mopp, 22):
        continue

    for w22g in w22_5th:
        t1, t2 = w22g['team1'], w22g['team2']
        if mw in tw[t1] or mw in tw[t2]:
            continue
        if is_blocked(t1, mw) or is_blocked(t2, mw):
            continue
        if 22 in tw[mopp]:
            continue

        nb, new_tw = try_swap(
            mg['game_id'], 'Maitland 5th', mopp, mw,
            w22g['game_id'], t1, t2, 22
        )
        delta = len(nb) - len(current_b2b)
        fixed = [x for x in current_b2b if x not in nb]
        created = [x for x in nb if x not in current_b2b]
        if ('Maitland 5th', 22, 23) in fixed:
            print(f'  {mg["game_id"]} (Mait vs {mopp} W{mw}) <-> {w22g["game_id"]} ({t1} vs {t2} W22)')
            print(f'    delta={delta}, fixes={fixed}, creates={created}')

print('\nSwap Maitland game <-> W23 game:')
for mg in mait_games:
    mopp = mg['team2'] if mg['team1'] == 'Maitland 5th' else mg['team1']
    mw = mg['week']
    if mw in (22, 23):
        continue
    if mw in SWAP_ONLY:
        continue
    if is_blocked('Maitland 5th', 23) or is_blocked(mopp, 23):
        continue

    for w23g in w23_5th:
        t1, t2 = w23g['team1'], w23g['team2']
        if mw in tw[t1] or mw in tw[t2]:
            continue
        if is_blocked(t1, mw) or is_blocked(t2, mw):
            continue
        if 23 in tw[mopp]:
            continue

        nb, new_tw = try_swap(
            mg['game_id'], 'Maitland 5th', mopp, mw,
            w23g['game_id'], t1, t2, 23
        )
        delta = len(nb) - len(current_b2b)
        fixed = [x for x in current_b2b if x not in nb]
        created = [x for x in nb if x not in current_b2b]
        if ('Maitland 5th', 22, 23) in fixed:
            print(f'  {mg["game_id"]} (Mait vs {mopp} W{mw}) <-> {w23g["game_id"]} ({t1} vs {t2} W23)')
            print(f'    delta={delta}, fixes={fixed}, creates={created}')
