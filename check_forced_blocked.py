"""Check forced games and blocked games satisfaction on a checkpoint."""
import sys
import os
import pickle

sys.path.insert(0, os.path.dirname(__file__))

from config import load_season_data
from config.season_2026 import FORCED_GAMES, BLOCKED_GAMES
from analytics.storage import DrawStorage


def game_matches_team_filter(game, entry):
    """Check if a game matches the team/club filter of an entry.

    Returns:
        True if the game involves the specified teams/club.
        None if no team/club filter exists (matches all).
    """
    if 'teams' in entry:
        teams = entry['teams']
        if len(teams) == 2:
            # Both must match (order-independent)
            t1_matches = any(t in game.team1 or t in game.team2 for t in [teams[0]])
            t2_matches = any(t in game.team1 or t in game.team2 for t in [teams[1]])
            # More precise: one team matches teams[0], the other matches teams[1]
            match_a = (teams[0] in game.team1 and teams[1] in game.team2)
            match_b = (teams[1] in game.team1 and teams[0] in game.team2)
            return match_a or match_b
        elif len(teams) == 1:
            return teams[0] in game.team1 or teams[0] in game.team2
    elif 'club' in entry:
        club = entry['club']
        return club in game.team1 or club in game.team2
    return None  # No team filter


def game_matches_scope(game, entry):
    """Check if a game matches the scope fields of an entry."""
    if 'grade' in entry:
        if game.grade != entry['grade']:
            return False
    if 'grades' in entry:
        if game.grade not in entry['grades']:
            return False
    if 'date' in entry:
        if game.date != entry['date']:
            return False
    if 'day' in entry:
        if game.day != entry['day']:
            return False
    if 'field_location' in entry:
        if game.field_location != entry['field_location']:
            return False
    if 'field_name' in entry:
        if game.field_name != entry['field_name']:
            return False
    if 'round_no' in entry:
        if game.round_no != entry['round_no']:
            return False
    if 'time' in entry:
        times = entry['time'] if isinstance(entry['time'], list) else [entry['time']]
        # game.time could be "HH:MM:SS" or "HH:MM", normalize
        game_time = game.time
        if len(game_time) > 5:
            game_time = game_time[:5]
        if game_time not in times:
            return False
    return True


def check_forced_games(games, forced_games):
    """Check each forced game entry.

    FORCED_GAMES logic: variables matching scope but NOT matching teams get eliminated.
    The remaining variables (matching scope AND matching teams) must sum to 1
    (or custom count/constraint).

    So we check: count of games matching BOTH scope AND teams.
    """
    passes = []
    fails = []

    for i, entry in enumerate(forced_games):
        desc = entry.get('description', f'Entry #{i}')
        constraint_type = entry.get('constraint', 'equal')  # default: exactly equal
        required_count = entry.get('count', 1)

        matching = []
        for g in games:
            if not game_matches_scope(g, entry):
                continue
            team_match = game_matches_team_filter(g, entry)
            if team_match is None or team_match:
                matching.append(g)

        count = len(matching)

        if constraint_type == 'lesse':
            satisfied = count <= required_count
            requirement = f"<= {required_count}"
        elif constraint_type == 'greatere':
            satisfied = count >= required_count
            requirement = f">= {required_count}"
        else:
            satisfied = count == required_count
            requirement = f"== {required_count}"

        result = {
            'description': desc,
            'requirement': requirement,
            'actual': count,
            'satisfied': satisfied,
            'matching_games': matching,
        }

        if satisfied:
            passes.append(result)
        else:
            fails.append(result)

    return passes, fails


def check_blocked_games(games, blocked_games):
    """Check each blocked game entry.

    BLOCKED_GAMES logic: variables matching scope AND matching teams get eliminated.
    If no teams/club specified, ALL variables matching scope are eliminated.

    So we check: count of games matching scope AND teams should be 0.
    """
    passes = []
    fails = []

    for i, entry in enumerate(blocked_games):
        desc = entry.get('description', f'Entry #{i}')

        matching = []
        for g in games:
            if not game_matches_scope(g, entry):
                continue
            team_match = game_matches_team_filter(g, entry)
            if team_match is None or team_match:
                # No team filter = block ALL in scope; team filter = block matching
                matching.append(g)

        count = len(matching)
        satisfied = count == 0

        result = {
            'description': desc,
            'requirement': '== 0 (blocked)',
            'actual': count,
            'satisfied': satisfied,
            'matching_games': matching,
        }

        if satisfied:
            passes.append(result)
        else:
            fails.append(result)

    return passes, fails


def main():
    checkpoint_path = 'checkpoints/run_217/simple_solve_intermediate_110/solution.pkl'

    print(f"Loading checkpoint: {checkpoint_path}")
    with open(checkpoint_path, 'rb') as f:
        solution = pickle.load(f)

    print(f"Loaded {len(solution)} variables, {sum(1 for v in solution.values() if v)} scheduled")

    print("Converting to DrawStorage...")
    draw = DrawStorage.from_X_solution(solution, description='Checkpoint analysis')
    print(f"Draw has {draw.num_games} games across {draw.num_weeks} weeks")

    games = draw.games

    print("\n" + "=" * 80)
    print("FORCED GAMES CHECK")
    print("=" * 80)

    f_passes, f_fails = check_forced_games(games, FORCED_GAMES)

    print(f"\n--- FAILURES ({len(f_fails)}) ---")
    for r in f_fails:
        print(f"\n  FAIL: {r['description']}")
        print(f"        Requirement: {r['requirement']}, Actual: {r['actual']}")
        if r['matching_games']:
            for g in r['matching_games']:
                print(f"        Game: {g.team1} vs {g.team2} | {g.grade} | {g.date} {g.day} {g.time} | {g.field_name} @ {g.field_location}")

    print(f"\n--- PASSES ({len(f_passes)}) ---")
    for r in f_passes:
        detail = ""
        if r['matching_games']:
            g = r['matching_games'][0]
            detail = f" -> {g.team1} vs {g.team2} | {g.date} {g.day} {g.time} | {g.field_name} @ {g.field_location}"
            if len(r['matching_games']) > 1:
                detail += f" (+ {len(r['matching_games'])-1} more)"
        print(f"  PASS: {r['description']} [req {r['requirement']}, got {r['actual']}]{detail}")

    print("\n" + "=" * 80)
    print("BLOCKED GAMES CHECK")
    print("=" * 80)

    b_passes, b_fails = check_blocked_games(games, BLOCKED_GAMES)

    print(f"\n--- FAILURES ({len(b_fails)}) ---")
    for r in b_fails:
        print(f"\n  FAIL: {r['description']}")
        print(f"        Found {r['actual']} game(s) that should be blocked:")
        for g in r['matching_games']:
            print(f"        Game: {g.team1} vs {g.team2} | {g.grade} | {g.date} {g.day} {g.time} | {g.field_name} @ {g.field_location}")

    print(f"\n--- PASSES ({len(b_passes)}) ---")
    for r in b_passes:
        print(f"  PASS: {r['description']}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Forced games:  {len(f_passes)} PASS, {len(f_fails)} FAIL (of {len(FORCED_GAMES)} total)")
    print(f"Blocked games: {len(b_passes)} PASS, {len(b_fails)} FAIL (of {len(BLOCKED_GAMES)} total)")


if __name__ == '__main__':
    main()
