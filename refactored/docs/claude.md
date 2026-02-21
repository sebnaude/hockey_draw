# Claude AI Instructions for Hockey Draw Scheduling System

This document provides guidance for AI assistants working with this codebase.

## Project Overview

This is a constraint programming system for generating hockey competition draws (schedules). It uses Google OR-Tools' CP-SAT solver to find optimal game schedules that satisfy numerous hard and soft constraints.

### Key Components

```
refactored/
├── main.py                 # Main entry point for schedule generation
├── main_staged.py          # Staged solving with checkpoints
├── constraints.py          # Original constraint implementations
├── constraints_ai.py       # AI-enhanced constraint implementations
├── models.py               # Pydantic data models
├── utils.py                # Utility functions
├── generate_x.py           # Decision variable generation
├── draw_analytics.py       # DrawStorage and DrawAnalytics classes
├── draw_tester.py          # DrawTester for modification testing
├── main_notebook_translation.py  # Full implementation from notebook
├── DRAW_RULES.md           # Documentation of all scheduling rules
├── example_draw_workflow.py # Example usage of analytics/testing tools
├── tests/
│   ├── test_constraints.py      # Unit tests for constraints
│   ├── test_constraints_ai.py   # Tests comparing AI vs original constraints
│   └── test_draw_outcomes.py    # Outcome validation tests
└── data/                   # Team data files
```

## Understanding the System

### How Scheduling Works

1. **Data Loading**: Teams, clubs, grades, and timeslots are loaded from CSV files
2. **Variable Generation**: Boolean decision variables `X[key]` are created for each possible (game, timeslot) combination
3. **Constraint Application**: Constraints are added to the model to restrict valid solutions
4. **Optimization**: The solver maximizes scheduled games minus penalties
5. **Solution Export**: Valid assignments are converted to a roster and exported to Excel

### Decision Variable Structure

Each game variable uses a tuple key:
```python
key = (team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)
#      0      1      2      3    4         5     6     7     8         9           10
```

### Constraint Types

1. **Required (Hard)**: Must be satisfied or no solution exists
   - No double-booking teams
   - No double-booking fields
   - Equal games per team

2. **Strong**: Very important for practical schedule
   - PHL/2nd grade separation
   - Home/away balance
   - Club grade adjacency

3. **Medium**: Important for quality
   - Venue limits
   - Timeslot optimization
   - Club days

4. **Soft**: Preferences with penalties
   - Home grouping
   - Club diversity
   - Preferred times

## Working with Constraints

### Adding a New Constraint

1. Create a class inheriting from `Constraint` (or `ConstraintAI`)
2. Implement the `apply(self, model, X, data)` method
3. Add constraint logic using `model.Add()` for hard constraints
4. For soft constraints, create penalty variables and add to `data['penalties']`
5. Register in `main.py`'s constraint list

### Example Constraint Pattern

```python
class MyNewConstraint(Constraint):
    """Description of what this constraint enforces."""
    
    def apply(self, model, X, data):
        current_week = data.get('current_week', 0)
        
        # Group variables by relevant key
        groups = defaultdict(list)
        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            if key[6] <= current_week:
                continue
            
            group_key = (key[6], key[4])  # e.g., (week, day_slot)
            groups[group_key].append(var)
        
        # Add constraints
        for group_key, vars_list in groups.items():
            model.Add(sum(vars_list) <= 1)  # Example: max 1 per group
```

### Soft Constraint Pattern

```python
class MySoftConstraint(Constraint):
    def apply(self, model, X, data):
        if 'penalties' not in data:
            data['penalties'] = {}
        data['penalties']['MyConstraint'] = {'weight': 1000, 'penalties': []}
        
        # ... logic ...
        
        penalty = model.NewIntVar(0, 100, 'penalty_name')
        model.Add(penalty == some_expression)
        data['penalties']['MyConstraint']['penalties'].append(penalty)
```

## Common Tasks

### Running the Scheduler

```bash
cd refactored
python main.py
```

### Running Tests

```bash
cd refactored
pytest tests/ -v
```

### Validating a Draw

```python
from tests.test_draw_outcomes import validate_draw, print_validation_report

validator = validate_draw(X_solution, data)
print(validator.summary())
```

## Key Design Decisions

### Why Staged Solving?

CP-SAT can save model state and continue with additional constraints. Staged solving:
1. First ensures a feasible schedule exists (required constraints)
2. Then progressively adds quality constraints
3. Allows resumption if a stage hangs
4. Provides intermediate solutions

### Why Separate `constraints.py` and `constraints_ai.py`?

- `constraints.py`: Original implementations, preserved for comparison
- `constraints_ai.py`: Refactored versions with:
  - Better code organization
  - Reduced edge cases
  - Priority classification
  - Helper methods for common patterns

## Data Model

### Key Entities

- **Club**: Organization with name, home field
- **Team**: Belongs to a club, has a grade (PHL, 2nd, 3rd, etc.)
- **Grade**: Competition level with list of teams
- **Timeslot**: Specific date/time/field combination
- **Field**: Physical playing field at a location

### Grade Hierarchy

```
PHL (Premier Hockey League) - Top grade
2nd Grade
3rd Grade
4th Grade
5th Grade
6th Grade - Lowest grade
```

Adjacent grades (e.g., 3rd & 4th) have special constraints.

## Troubleshooting

### Model is Infeasible

1. Check constraint order - required constraints should be minimal
2. Use `solver.parameters.log_search_progress = True`
3. Try removing constraints one by one to find the conflict
4. Check if team/field data is consistent

### Solver Takes Too Long

1. Increase `solver.parameters.max_time_in_seconds`
2. Consider staged solving - solve required first
3. Check if there are too many soft constraints
4. Try adjusting penalty weights

### Memory Issues

1. Reduce number of decision variables by tighter filtering in `generate_X`
2. Use staged solving with checkpoints
3. Increase system memory limits

## Draw Management Tools

### DrawStorage (Pliable Format)

The `DrawStorage` class provides a flexible JSON-based format for storing and manipulating draws:

```python
from draw_analytics import DrawStorage, DrawAnalytics

# Create from solver solution
draw = DrawStorage.from_X_solution(X_solution, description="Season 2025")
draw.save("draw.json")

# Load existing draw
draw = DrawStorage.load("draw.json")

# Query games
games = draw.get_games_by_team("Maitland PHL")
week_5 = draw.get_games_by_week(5)
filtered = draw.filter_games(team="Souths", grade="PHL", week=3)

# Convert back to X dict for solving
X_dict = draw.to_X_dict()
```

### DrawAnalytics (Comprehensive Analytics)

Generate detailed analytics and Excel reports:

```python
from draw_analytics import DrawAnalytics

analytics = DrawAnalytics(draw, data)

# Individual analytics
games_df = analytics.games_played_by_team_grade()       # Games per team per grade
matchups = analytics.team_matchups_crosstab()           # Who plays who
home_away = analytics.home_away_analysis()              # Home/away balance
away_balance = analytics.away_team_balance()            # Maitland/Gosford balance
club_schedule = analytics.club_season_schedule("Maitland")  # Club's full schedule
compliance = analytics.constraint_compliance_summary()  # Quick constraint check

# Export everything to Excel
analytics.export_analytics_to_excel("analytics.xlsx")
```

### DrawTester (Modification Testing)

Test game modifications without affecting the original draw:

```python
from draw_tester import DrawTester

# Load and test
tester = DrawTester.from_file("draw.json", data)

# Find games
games = tester.find_game(team="Souths PHL", week=3)

# Move a game
tester.move_game("G00123", new_week=4, new_day_slot=5, new_time="15:00")

# Swap two games
tester.swap_games("G00045", "G00067")

# Check for violations
report = tester.run_violation_check()
print(report.summary())
print(report.full_report())

# Reset to original
tester.reset()

# Save modified draw if acceptable
if not report.has_violations:
    tester.save_modified_draw("draw_modified.json")
```

### What-If Analysis

Test hypothetical changes:

```python
from draw_tester import what_if_move_game

# Test moving a Friday game to Saturday
report = what_if_move_game(
    "draw.json",
    data,
    game_id="G00123",
    new_day="Saturday",
    new_time="14:00",
    new_day_slot=3
)

if report.has_violations:
    print("This move would cause violations:")
    for v in report.violations:
        print(f"  • {v.constraint}: {v.message}")
else:
    print("This move is valid!")
```

## API Reference

### Important Functions

```python
# Generate decision variables
X, Y, conflicts, unavailable_games = generate_X(folder_path, model, data)

# Create and solve schedule
X_outcome, X_solution, data = create_schedule(game_vars=(X, Y), data=data, constraints=constraints, model=model)

# Convert to roster
roster = convert_X_to_roster(X_solution, data)

# Export to Excel
export_roster_to_excel(roster, data, filename='schedule.xlsx')
```

### Important Data Keys

```python
data = {
    'teams': List[Team],           # All teams
    'grades': List[Grade],         # All grades
    'fields': List[PlayingField],  # All fields
    'clubs': List[Club],           # All clubs
    'timeslots': List[Timeslot],   # All available timeslots
    'games': List[Tuple],          # All possible games (t1, t2, grade)
    'num_rounds': Dict[str, int],  # Rounds per grade
    'current_week': int,           # Week to start scheduling from
    'penalties': Dict,             # Penalty tracking
    'team_conflicts': List,        # Conflicting team pairs
    'club_days': Dict,             # Special club day dates
    # ... more configuration
}
```

## Best Practices

### When Modifying Constraints

1. Always add tests for new constraints
2. Document the rule in DRAW_RULES.md
3. Consider both unit tests and outcome tests
4. Test with both small and full datasets

### When Debugging

1. Use small test cases first
2. Print constraint counts after each constraint
3. Use the solver's logging
4. Validate intermediate solutions

### Code Style

1. Use descriptive variable names
2. Add docstrings to all constraint classes
3. Group related constraints together
4. Keep `apply()` methods focused

## Version Information

- Python 3.8+
- OR-Tools latest stable version
- Pydantic for data validation
- Pandas for data I/O
- Pytest for testing
