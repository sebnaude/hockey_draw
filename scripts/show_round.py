#!/usr/bin/env python
"""Show games for a specific round."""
import json
import sys
from pathlib import Path
from collections import defaultdict

# Grade display order and colors (ANSI)
GRADE_ORDER = ['PHL', '2nd', '3rd', '4th', '5th', '6th']
GRADE_COLORS = {
    'PHL': '\033[91m',  # Red
    '2nd': '\033[93m',  # Yellow
    '3rd': '\033[92m',  # Green
    '4th': '\033[94m',  # Blue
    '5th': '\033[95m',  # Magenta
    '6th': '\033[96m',  # Cyan
}
RESET = '\033[0m'

# Club extraction from team name
def get_club(team_name):
    """Extract club name from team name."""
    # Handle special cases
    if team_name.startswith('Colts Gold'):
        return 'Colts'
    if team_name.startswith('Colts Green'):
        return 'Colts'
    if team_name.startswith('Uni Seapigs'):
        return 'Uni'
    if team_name.startswith('Wests Red'):
        return 'Wests'
    if team_name.startswith('Tigers Black'):
        return 'Tigers'
    if team_name.startswith('Port Stephens'):
        return 'Port Stephens'
    # Default: first word
    return team_name.split()[0]

def get_all_teams(draw):
    """Extract all unique teams from the entire draw."""
    teams = set()
    for g in draw['games']:
        teams.add((g['team1'], g['grade']))
        teams.add((g['team2'], g['grade']))
    return teams

def show_round(draw_path: str, round_no: int):
    with open(draw_path) as f:
        draw = json.load(f)
    
    games = [g for g in draw['games'] if g['round_no'] == round_no]
    
    if not games:
        print(f"No games found for round {round_no}")
        return
    
    # Get all teams in the draw
    all_teams = get_all_teams(draw)
    
    # Teams playing this round
    playing_teams = set()
    for g in games:
        playing_teams.add((g['team1'], g['grade']))
        playing_teams.add((g['team2'], g['grade']))
    
    # Teams with byes
    bye_teams = all_teams - playing_teams
    
    print(f"{'='*70}")
    print(f"ROUND {round_no} - {len(games)} games - {games[0]['date']}")
    print(f"{'='*70}")
    
    # ========== VIEW 1: BY FIELD ==========
    print(f"\n{'─'*70}")
    print("VIEW 1: BY FIELD (ordered by time)")
    print(f"{'─'*70}")
    
    # Group by field, then venue
    by_field = defaultdict(list)
    for g in games:
        key = (g['field_location'], g['field_name'])
        by_field[key].append(g)
    
    # Sort each field's games by time
    for key in by_field:
        by_field[key].sort(key=lambda g: g['time'])
    
    # Display grouped by field
    for (venue, field), field_games in sorted(by_field.items(), key=lambda x: (x[0][0], x[0][1])):
        venue_short = venue[:30] if len(venue) > 30 else venue
        print(f"\n=== {field} @ {venue_short} ===")
        for g in field_games:
            color = GRADE_COLORS.get(g['grade'], '')
            grade_str = f"{color}{g['grade']:4}{RESET}"
            print(f"  {g['time']}  {grade_str}  {g['team1']:20} vs {g['team2']:20}")
    
    # ========== VIEW 2: BY CLUB ==========
    print(f"\n{'─'*70}")
    print("VIEW 2: BY CLUB (see spread of game times)")
    print(f"{'─'*70}")
    
    # Group games by club involvement
    by_club = defaultdict(list)
    for g in games:
        club1 = get_club(g['team1'])
        club2 = get_club(g['team2'])
        by_club[club1].append(g)
        if club1 != club2:
            by_club[club2].append(g)
    
    # Sort clubs alphabetically
    for club in sorted(by_club.keys()):
        club_games = sorted(by_club[club], key=lambda g: g['time'])
        times = [g['time'] for g in club_games]
        
        # Calculate spread
        if len(times) >= 2:
            first = times[0]
            last = times[-1]
            spread = f"({first} - {last})"
        else:
            spread = ""
        
        print(f"\n=== {club} === {len(club_games)} games {spread}")
        for g in club_games:
            color = GRADE_COLORS.get(g['grade'], '')
            grade_str = f"{color}{g['grade']:4}{RESET}"
            # Show which team from this club
            if get_club(g['team1']) == club:
                team_marker = g['team1']
            else:
                team_marker = g['team2']
            opponent = g['team2'] if get_club(g['team1']) == club else g['team1']
            print(f"  {g['time']}  {grade_str}  {team_marker:20} vs {opponent:20}  {g['field_name']}")
    
    # ========== BYES ==========
    if bye_teams:
        print(f"\n{'─'*70}")
        print(f"BYES ({len(bye_teams)} teams)")
        print(f"{'─'*70}")
        bye_by_grade = defaultdict(list)
        for team, grade in bye_teams:
            bye_by_grade[grade].append(team)
        
        for grade in GRADE_ORDER:
            if grade in bye_by_grade:
                color = GRADE_COLORS.get(grade, '')
                teams = sorted(bye_by_grade[grade])
                print(f"  {color}{grade}{RESET}: {', '.join(teams)}")

if __name__ == '__main__':
    draw_path = sys.argv[1] if len(sys.argv) > 1 else 'draws/2026/draw_v1.0.json'
    round_no = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    show_round(draw_path, round_no)
