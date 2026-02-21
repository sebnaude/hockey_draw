"""
Generate decision variables X and Y for the scheduling model.
Ported and adapted from the original notebook.
"""
from typing import Tuple, Dict, Any
from ortools.sat.python import cp_model
from models import Team, Timeslot

def generate_X(model: cp_model.CpModel, data: dict) -> Tuple[Dict[Any, Any], Dict[Any, Any], dict, dict]:
    """
    Generate decision variables for all possible games and timeslots.
    Returns X, Y, conflicts, unavailable_games.
    """
    teams = data['teams']
    timeslots = data['timeslots']
    num_dummy_timeslots = data.get('num_dummy_timeslots', 3)
    games = {
        (t1.name, t2.name, t1.grade): (t1, t2, t1.grade)
        for i, t1 in enumerate(teams)
        for t2 in teams[i + 1:] if t1.grade == t2.grade
    }
    data['games'] = games
    print(f"Generated {len(games)} games.")

    X = {}
    Y = {}
    conflicts = {}
    unavailable_games = {}
    for (t1_name, t2_name, grade_name), (t1, t2, grade) in games.items():
        for t in timeslots:
            if not t.day:
                continue  # skip dummy timeslots
            key = (t1_name, t2_name, grade_name, t.day, t.day_slot, t.time, t.week, t.date, t.round_no, t.field.name, t.field.location)
            X[key] = model.NewBoolVar(f'X_{t1_name}_{t2_name}_{t.day}_{t.time}_{t.week}_{t.field.name}_{t.field.location}')
    print(f"Decision variables {len(X)}")
    # Dummy Y and empty conflicts/unavailable_games for now
    return X, Y, conflicts, unavailable_games
