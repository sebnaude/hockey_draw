# main.py
"""
Main entry point for the scheduling system.

This module provides the original single-solve approach to schedule generation.
For staged solving with checkpointing, see main_staged.py.

Usage:
    python main.py                    # Run single solve
    python main_staged.py             # Run staged solve
    python main_staged.py --resume    # Resume from checkpoint

See DRAW_RULES.md for documentation of all scheduling rules.
See claude.md for AI assistant instructions.
"""
import os
from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as tm
from models import *
from utils import *
from constraints import *
from generate_x import generate_X
from utils import convert_X_to_roster, export_roster_to_excel
import json

def create_schedule(game_vars, data, constraints, model):
    X, Y = game_vars
    X.update(Y)
    print(f"Generated {len(X)} decision variables.")  

    prior_val = 0
    for constraint in constraints:
        constraint.apply(model, X, data)
        print(constraint.__class__.__name__, f" {len(model.Proto().constraints) - prior_val} constraints added.")
        prior_val = len(model.Proto().constraints) 
        
    penalties_dict = data.get('penalties', {})

    print(f'Number of penalties per constraint: {[(name, len(info.get("penalties", []))) for name, info in penalties_dict.items()]}')

    total_penalty = sum(
        info['weight'] * sum(info['penalties'])  # Multiply weight by sum of violations
        for info in penalties_dict.values() if 'penalties' in info
    )

    model.Maximize(
        sum(X.values()) - sum(Y.values()) - total_penalty  
    )
    print(f"Total constraints in model: {len(model.Proto().constraints)}")
    
    solver = cp_model.CpSolver()

    solver.parameters.log_search_progress = True
    solver.parameters.cp_model_probing_level = 0  # Avoid aggressive pruning to analyze infeasibility

    solver.parameters.max_time_in_seconds = 28800 # this last one was close, maybe double it and be ok

    all_vars = list(X.values())
    # callback = SaveStateCallback(all_vars, interval=1)  # If SaveStateCallback is needed, import and use

    # checkpoint_path = data.get('checkpoint_path', None)
    # if checkpoint_path is not None:
    #     if not callback.load_checkpoint(path = checkpoint_path):
    #         print("No checkpoint found, starting fresh.")

    # Solve—note callback is positional
    status = solver.Solve(model)
    # status = solver.Solve(model, solution_callback=callback)
    # Print the best solution found
    # if callback.best_solution:
    #     print("Best solution found:", callback.best_solution)
    #     print("Objective value:", callback.best_obj_value)
        
    print(f"Solver status: {solver.StatusName()}") 
    X_outcome = X

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        X_solution = {key: solver.Value(var) for key, var in X.items()}
        for constraint_name, info in data.get('penalties', {}).items():
            if 'penalties' in info:
                unresolved_penalties = [p for p in info['penalties'] if solver.Value(p) > 0]
                data['penalties'][constraint_name]['unresolved'] = unresolved_penalties
        return X_outcome, X_solution, data
    else:
        print("No feasible schedule found.")
        return X_outcome, {}, data

def main():
    # --- Data loading and setup (full data, not just minimal test data) ---
    # Load all required data as in main_notebook_translation.py
    import os
    import pandas as pd
    from datetime import datetime, timedelta, time as tm
    from collections import defaultdict

    # Define FIELDS
    FIELDS = [
        PlayingField(location='Newcastle International Hockey Centre', name='SF'),
        PlayingField(location='Newcastle International Hockey Centre', name='EF'),
        PlayingField(location='Newcastle International Hockey Centre', name='WF'),
        PlayingField(location='Maitland Park', name='Maitland Main Field'),
        PlayingField(location='Central Coast Hockey Park', name='Wyong Main Field'),
    ]

    TEAMS_DATA = os.path.join('data', '2025', 'teams')
    CLUBS = []
    TEAMS = []
    for file in os.listdir(TEAMS_DATA):
        df = pd.read_csv(os.path.join(TEAMS_DATA, file))
        club_name = df['Club'].iloc[0].strip()
        home_field = (
            'Maitland Park' if club_name == 'Maitland' else
            'Central Coast Hockey Park' if club_name == 'Gosford' else
            'Newcastle International Hockey Centre'
        )
        club = Club(name=club_name, home_field=home_field)
        CLUBS.append(club)
        teams = [Team(name=f"{row['Team Name'].strip()} {row['Grade'].strip()}", club=club, grade=row['Grade'].strip()) for _, row in df.iterrows()]
        TEAMS.extend(teams)

    teams_by_grade = defaultdict(list)
    for team in TEAMS:
        teams_by_grade[team.grade].append(team.name)
    GRADES = [Grade(name=grade, teams=teams) for grade, teams in sorted(teams_by_grade.items())]

    teams_by_club = defaultdict(list)
    for team in TEAMS:
        teams_by_club[team.club.name].append(team.name)
    for club, teams in teams_by_club.items():
        club_obj = next((c for c in CLUBS if c.name == club), None)
        if club_obj:
            club_obj.num_teams = len(teams)
    for grade in GRADES:
        grade.num_teams = len(grade.teams)

    # Timeslot and field/time setup (simplified, can be expanded as needed)
    day_time_map = {'Newcastle International Hockey Centre': {'Sunday':[tm(8, 30), tm(10, 0), tm(11,30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30), tm(19, 0)]},
                    'Maitland Park': {'Sunday':[tm(9, 0), tm(10, 30), tm(12, 0), tm(13, 30), tm(15, 0), tm(16,30)]}}
    phl_game_times = {'Newcastle International Hockey Centre': {'Friday':[tm(19, 0)], 'Sunday':[tm(11,30), tm(13, 0), tm(14, 30), tm(16, 0)]},
                      'Central Coast Hockey Park': {'Friday':[tm(20, 0)], 'Sunday': [tm(15, 0)]},
                      'Maitland Park': {'Sunday':[ tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]}}
    field_unavailabilities = {'Maitland Park': 
                                {'weekends':[datetime(2025, 4, 19), datetime(2025, 4, 12), datetime(2025, 5, 10), datetime(2025, 5, 24), datetime(2025, 6, 28), datetime(2025,5,3) , datetime(2025,6,7) ], # # Enter the saturday of the weekend to be safe and definitely catch Fridays as being out too #
                                'whole_days':[datetime(2025, 4, 25),],
                                'part_days':[],
                                },
                            'Newcastle International Hockey Centre': 
                                {'weekends':[datetime(2025, 4, 19), datetime(2025,5,3), datetime(2025,6,7)],
                                'whole_days':[datetime(2025, 4, 25), datetime(2025, 5, 31)],
                                'part_days':[datetime(2025, 6,1, 8, 30), datetime(2025, 6, 1, 10,0), datetime(2025,6,1,11,30), ],
                                },
                            'Central Coast Hockey Park': 
                                {'weekends':[ datetime(2025, 4, 19), datetime(2025, 4, 5),datetime(2025,5,3), datetime(2025,6,7),],#
                                'whole_days':[datetime(2025, 4, 25), ],
                                'part_days':[],
                                },
                            }
    
    start = datetime(2025, 3, 21)
    end = datetime(2025, 9, 2)
    # Use merged_dict logic from notebook for timeslot generation
    merged_dict = defaultdict(lambda: defaultdict(list))
    for d in (phl_game_times, day_time_map):
        for field, days in d.items():
            for key, times in days.items():
                merged_dict[field][key].extend(times)
    for field in merged_dict:
        for key in merged_dict[field]:
            merged_dict[field][key] = list(dict.fromkeys(merged_dict[field][key]))
            merged_dict[field][key].sort()
    # Generate timeslots
    from main_notebook_translation import generate_timeslots, max_games_per_grade, get_club_from_clubname
    timeslots = generate_timeslots(start, end, merged_dict, FIELDS, field_unavailabilities)
    TIMESLOTS = [Timeslot(date=t['date'], day=t['day'], time=t['time'], week=t['week'], day_slot=t['day_slot'], field=t['field'], round_no=t['round_no']) for t in timeslots]
    max_rounds = 21
    num_rounds = max_games_per_grade(GRADES, max_rounds)
    num_rounds['max'] = max_rounds
    for grade, rounds in num_rounds.items():
        grade_obj = next((t for t in GRADES if t.name == grade), None)
        if grade_obj is not None:
            grade_obj.set_games(rounds)
    max_day_slot_per_field = {field.location: max(t.day_slot for t in TIMESLOTS if t.field.location == field.location) for field in FIELDS}
    club_days = {'Crusaders': datetime(2025, 6, 22), 'Wests': datetime(2025, 7, 13), 'University': datetime(2025, 7, 27), 'Tigers': datetime(2025, 7, 6), 'Port Stephens': datetime(2025, 7, 20)}
    preference_no_play = {'Maitland':[{'date': '2025-07-20', 'field_location':'Newcastle International Hockey Centre'}, {'date': '2025-08-24', 'field_location':'Newcastle International Hockey Centre'}],
                         'Norths':[{'team_name': 'Norths PHL', 'date': '2025-03-23', 'time':'11:30'}, {'team_name': 'Norths PHL',  'date': '2025-03-23', 'time':'13:00'}, {'team_name': 'Norths PHL',  'date': '2025-03-23', 'time':'14:30'}, {'team_name': 'Norths PHL',  'date': '2025-03-23', 'time':'16:00'}]}
    phl_preferences = {'preferred_dates' :[]}
    UNAVAILABILITY_PATH = os.path.join('data', '2025', 'noplay')

    data = {
        'teams': TEAMS,
        'grades': GRADES,
        'fields': FIELDS,
        'clubs': CLUBS,
        'timeslots': TIMESLOTS,
        'num_rounds': num_rounds,
        'current_week': 0,
        'penalties': {},
        'day_time_map': day_time_map,
        'phl_game_times': phl_game_times,
        'phl_preferences': phl_preferences,
        'max_day_slot_per_field': max_day_slot_per_field,
        'field_unavailabilities': field_unavailabilities,
        'club_days': club_days,
        'preference_no_play': preference_no_play,
    }

    # --- Timeslot and variable generation ---
    from generate_x import generate_X
    model = cp_model.CpModel()
    X, Y, conflicts, unavailable_games = generate_X(UNAVAILABILITY_PATH, model, data)
    data['unavailable_games'] = unavailable_games
    data['team_conflicts'] = conflicts

    # --- Constraints ---
    constraints = [
        EnsureEqualGamesAndBalanceMatchUps(),
        NoDoubleBookingTeamsConstraint(),
        NoDoubleBookingFieldsConstraint(),
        FiftyFiftyHomeandAway(),
        MaxMaitlandHomeWeekends(),
        ClubDayConstraint(),
        PHLAndSecondGradeTimes(),
        PHLAndSecondGradeAdjacency(),
        TeamConflictConstraint(),
        AwayAtMaitlandGrouping(),
        MinimiseClubsOnAFieldBroadmeadow(),
        EnsureBestTimeslotChoices(),
        MaximiseClubsPerTimeslotBroadmeadow(),
        ClubVsClubAlignment(),
        ClubGradeAdjacencyConstraint(),
        MaitlandHomeGrouping(),
        EqualMatchUpSpacingConstraint(),
    ]

    # --- Create and solve the schedule ---
    X_outcome, X_solution, data = create_schedule(game_vars=(X, Y), data=data, constraints=constraints, model=model)

    # --- Convert solution to roster and export ---
    roster = convert_X_to_roster(X_solution, data)
    output_path = os.path.join('draws', 'schedule.xlsx')
    export_roster_to_excel(roster, data, filename=output_path)
    print(f"Draw saved to {output_path}")

if __name__ == "__main__":
    main()
