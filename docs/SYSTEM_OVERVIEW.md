# Hockey Draw Scheduling System - Technical Overview

## System Architecture

This system generates optimal hockey competition schedules using Google OR-Tools' CP-SAT constraint programming solver.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         HOCKEY DRAW SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │
│  │   INPUT      │    │   SOLVER     │    │        OUTPUT            │  │
│  │              │    │              │    │                          │  │
│  │ • Team CSVs  │───▶│ • Variables  │───▶│ • Excel Schedule         │  │
│  │ • Config     │    │ • Constraints│    │ • JSON Draw (pliable)    │  │
│  │ • Noplay     │    │ • Objectives │    │ • Analytics Report       │  │
│  │ • Field Avail│    │ • Staged     │    │ • Violation Report       │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
refactored/
├── run.py                  # 🚀 MAIN ENTRY POINT (use this)
│
├── core/                   # Core scheduling engine
│   ├── __init__.py
│   ├── models.py           # Pydantic data models
│   ├── constraints.py      # Constraint implementations
│   ├── solver.py           # Staged solver with checkpoints
│   └── variables.py        # Decision variable generation
│
├── analytics/              # Draw analysis and testing
│   ├── __init__.py
│   ├── storage.py          # DrawStorage (pliable JSON format + partial import)
│   ├── reports.py          # ClubReport, GradeReport, ComplianceCertificate
│   └── tester.py           # DrawTester (modification testing)
│
├── config/                 # Season configuration
│   ├── __init__.py
│   ├── season_2025.py      # 2025 season settings
│   └── season_2026.py      # 2026 season settings (template)
│
├── data/                   # Input data files
│   └── {year}/
│       ├── teams/          # Team CSV files
│       ├── noplay/         # Unavailability XLSX files
│       └── field_availability/
│
├── docs/                   # Documentation
│   ├── SYSTEM_OVERVIEW.md  # This file
│   ├── DRAW_RULES.md       # Constraint documentation
│   ├── README.md           # User guide
│   └── claude.md           # AI assistant instructions
│
├── tests/                  # Test suite (216 tests)
│   ├── test_ai_constraints_comprehensive.py  # 70 AI constraint tests
│   ├── test_constraints.py
│   ├── test_constraints_ai.py
│   ├── test_constraints_equivalence.py
│   ├── test_draw_outcomes.py
│   └── test_utils.py
│
├── draws/                  # Output schedules
├── checkpoints/            # Solver checkpoints
│
├── requirements.txt        # Python dependencies
└── pytest.ini             # Test configuration
```

---

## Data Flow

### 1. Input Phase

```
Team CSVs ──────┐
                │
Season Config ──┼──▶ load_data() ──▶ data dict
                │
Unavailabilities┘
```

**Data dict contains:**
- `teams`: List[Team] - All teams with clubs and grades
- `grades`: List[Grade] - Grade definitions
- `clubs`: List[Club] - Club definitions with home fields
- `timeslots`: List[Timeslot] - All available game slots
- `num_rounds`: Dict[str, int] - Rounds per grade
- `field_unavailabilities`: Dict - Blocked dates per venue
- `club_days`: Dict - Special club event dates
- `preference_no_play`: Dict - Soft scheduling restrictions

### 2. Variable Generation Phase

```
data dict ──▶ generate_X() ──▶ X (decision variables)
```

**Decision variable key structure:**
```python
X[(team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)]
```

Each variable is a boolean: 1 = game scheduled at this slot, 0 = not scheduled.

### 3. Constraint Application Phase (Staged)

```
┌────────────────────────────────────────────────────────────────┐
│ Stage 1: Required (MUST satisfy)                               │
│   • NoDoubleBookingTeams                                       │
│   • NoDoubleBookingFields                                      │
│   • EnsureEqualGamesAndBalanceMatchUps                        │
│   • TeamConflictConstraint                                     │
│                         ↓ checkpoint                           │
├────────────────────────────────────────────────────────────────┤
│ Stage 2: Structural (Important for practicality)               │
│   • FiftyFiftyHomeandAway                                      │
│   • MaxMaitlandHomeWeekends                                    │
│   • ClubDayConstraint                                          │
│   • PHLAndSecondGradeTimes                                     │
│   • PHLAndSecondGradeAdjacency                                │
│   • ClubGradeAdjacencyConstraint                              │
│                         ↓ checkpoint                           │
├────────────────────────────────────────────────────────────────┤
│ Stage 3: Optimization (Venue efficiency)                       │
│   • AwayAtMaitlandGrouping                                     │
│   • MinimiseClubsOnAFieldBroadmeadow                          │
│   • EnsureBestTimeslotChoices                                  │
│   • MaximiseClubsPerTimeslotBroadmeadow                       │
│   • ClubVsClubAlignment                                        │
│                         ↓ checkpoint                           │
├────────────────────────────────────────────────────────────────┤
│ Stage 4: Soft (Preferences with penalties)                     │
│   • MaitlandHomeGrouping                                       │
│   • EqualMatchUpSpacing                                        │
│   • PreferredTimesConstraint                                   │
│                         ↓ final solution                       │
└────────────────────────────────────────────────────────────────┘
```

### 4. Solving Phase

```
model + constraints ──▶ CP-SAT Solver ──▶ X_solution
```

The solver maximizes: `Σ(X) - Σ(Y) - penalties`
- Σ(X): Total scheduled games
- Σ(Y): Dummy games (penalized)
- penalties: Soft constraint violations

### 5. Output Phase

```
X_solution ──┬──▶ Excel Schedule (traditional format)
             ├──▶ JSON Draw (pliable format for modifications)
             ├──▶ Analytics Workbook (cross-tabs, compliance)
             └──▶ Violation Report (if any issues)
```

---

## Key Classes

### Data Models (`core/models.py`)

| Class | Description |
|-------|-------------|
| `PlayingField` | Field name + location |
| `Grade` | Grade name + team list |
| `Club` | Club name + home field |
| `Team` | Team name + club + grade |
| `Timeslot` | Date/time/field/slot info |
| `Game` | Two teams + timeslot + grade |
| `WeeklyDraw` | Games for one week |
| `Roster` | Complete season schedule |

### Constraints (`constraints.py` / `constraints_ai.py`)

The system has two parallel constraint sets:
- **`constraints.py`** — Original human-written constraints (**read-only, source of truth**)
- **`constraints_ai.py`** — AI-enhanced equivalents (opt-in via `--ai` flag)

All 18 constraint pairs have been audited for parity. Use `--ai` to select the AI set.

| Constraint | Type | Description |
|------------|------|-------------|
| `NoDoubleBookingTeams` | Hard | Team plays max once per week |
| `NoDoubleBookingFields` | Hard | Field hosts max one game per slot |
| `EnsureEqualGames` | Hard | Teams play equal games |
| `FiftyFiftyHomeAway` | Hard | Away teams get balanced home/away |
| `PHLAndSecondGradeAdjacency` | Hard | PHL/2nd play adjacent slots |
| `ClubGradeAdjacency` | Hard | Adjacent grades don't overlap |
| `MaitlandHomeGrouping` | Soft | Group Maitland home games |
| `PreferredTimes` | Soft | Honor club preferences |

### Analytics (`analytics/`)

| Class | Description |
|-------|-------------|
| `DrawStorage` | Pliable JSON format for draws |
| `DrawAnalytics` | Generates cross-tabs and reports |
| `DrawTester` | Tests game modifications |
| `ViolationReport` | Constraint violation summary |

---

## Running the System

### Quick Start

```bash
cd refactored
python run.py              # Generate new draw
python run.py --resume     # Resume from checkpoint
```

### Testing Modifications

```python
from analytics import DrawTester

tester = DrawTester.from_file("draws/draw_2025.json", data)
tester.move_game("G00123", new_week=5, new_day_slot=3)
report = tester.run_violation_check()
print(report.full_report())
```

### Generating Analytics

```python
from analytics import DrawStorage, DrawAnalytics

draw = DrawStorage.load("draws/draw_2025.json")
analytics = DrawAnalytics(draw, data)
analytics.export_analytics_to_excel("analytics.xlsx")
```

---

## Season Configuration

Each season requires a config file (`config/season_{year}.py`) with:

```python
SEASON_CONFIG = {
    'year': 2026,
    'start_date': datetime(2026, 3, 22),
    'end_date': datetime(2026, 9, 19),
    'max_rounds': 21,  # or 15 for 3-round format
    
    'fields': [...],
    'day_time_map': {...},
    'phl_game_times': {...},
    'field_unavailabilities': {...},
    'club_days': {...},
}
```

See `config/season_2025.py` for a complete example.

---

## Checkpoint System

Checkpoints are saved after each stage:

```
checkpoints/
└── run_1/
    ├── stage1_required/
    │   ├── solution.pkl
    │   └── metadata.json
    ├── stage2_strong/
    │   ├── solution.pkl
    │   └── metadata.json
    └── ...
```

Resume from any stage:
```bash
python run.py --resume run_1 stage2_strong
```

---

## Testing

```bash
# Run all tests (216 tests, ~21s)
pytest tests/ -v

# Run comprehensive AI constraint tests (70 tests)
pytest tests/test_ai_constraints_comprehensive.py -v

# Run specific test file
pytest tests/test_constraints.py -v

# Run with coverage
pytest tests/ --cov=core
```

### Test Suite Overview

| File | Tests | Purpose |
|------|-------|---------|
| `test_ai_constraints_comprehensive.py` | 70 | Full AI constraint coverage (feasibility, rejection, parity, combined, incremental) |
| `test_constraints.py` | 14 | Original constraint unit tests |
| `test_constraints_ai.py` | 8 | AI constraint unit tests |
| `test_constraints_equivalence.py` | 18 | Original vs AI parity tests |
| `test_constraints_comprehensive.py` | 8 | Extended constraint tests |
| `test_analytics_*.py` | 50 | Analytics, storage, tester tests |
| `test_draw_outcomes.py` | 4 | Outcome validation tests |
| `test_utils.py` | 20 | Utility function tests |

---

## Performance Notes

- **Memory**: Large numbers of timeslots × teams can exhaust memory
- **Solving time**: Typically 1-8 hours for full season
- **Staged approach**: Allows partial solutions if later stages fail
- **Hints**: Previous stage solutions seed the next stage

---

## Legacy Code

The following files have been cleaned up from the original notebook-based approach:

| File | Status | Notes |
|------|--------|-------|
| `main_notebook_translation.py` | REMOVED | Original 3787-line monolith |
| `main.py` | REMOVED | Simple solve functionality merged into `main_staged.py --simple` |
| `generate_x.py` | REMOVED | Stub, not actually used |

**Current active files:**
- `main_staged.py` - Staged solver with `--simple` mode for single-solve
- `constraints.py` - Original constraint implementations (READ-ONLY)
- `constraints_ai.py` - AI-enhanced constraint implementations
- `utils.py` - Utilities including `build_season_data()`
- `config/season_{year}.py` - Season-specific configuration

The refactored code in `core/` and `analytics/` should be used going forward.
