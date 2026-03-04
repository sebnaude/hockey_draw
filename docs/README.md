# Hockey Draw Scheduling System

A constraint programming system for generating hockey competition draws (schedules) using Google OR-Tools' CP-SAT solver. The system handles complex scheduling requirements including team availability, field constraints, home/away balance, and club preferences.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Required Data Inputs](#required-data-inputs)
- [File Structure](#file-structure)
- [Running the System](#running-the-system)
- [Output Files](#output-files)
- [Configuration](#configuration)

---

## Overview

This system generates optimal game schedules for a hockey competition by solving a constraint satisfaction problem. It considers:

- **17+ constraints** ranging from physical necessities (no double-booking) to preferences (club days)
- **Staged solving** for improved performance and partial solutions
- **Multiple venues** including home fields for Maitland and Gosford
- **Analytics generation** for schedule validation

---

## Installation

```powershell
cd refactored
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+
- ortools (Google OR-Tools)
- pandas
- pydantic
- openpyxl / xlsxwriter
- pytest (for testing)

---

## Required Data Inputs

### 1. Team Data Files (CSV)

**Location:** `data/{year}/teams/{club_name}.csv`

Each club requires a CSV file with their teams. One file per club.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `Club` | string | Club name | `Maitland` |
| `Grade` | string | Grade level | `PHL`, `2nd`, `3rd`, `4th`, `5th`, `6th` |
| `Team Name` | string | Team identifier | `Maitland`, `Red`, `Seapigs` |

**Example file:** `data/2025/teams/maitland.csv`
```csv
Club,Grade,Team Name
Maitland,PHL,Maitland
Maitland,2nd,Maitland
Maitland,4th,Maitland
Maitland,5th,Maitland
Maitland,6th,Maitland
```

**Team naming convention:** The full team name is constructed as `{Team Name} {Grade}`, e.g., `Maitland PHL`, `Souths 2nd`.

---

### 2. Team Unavailability Files (XLSX) - Optional

**Location:** `data/{year}/noplay/{club_name}_noplay.xlsx`

Excel files specifying when teams/clubs cannot play. Each file has 3 sheets:

#### Sheet: `club_noplay`
Club-wide restrictions (all teams affected)

| Column | Format | Description |
|--------|--------|-------------|
| `whole_weekend` | `DD/MM/YYYY` | Block entire weekend (by week number) |
| `whole_day` | `DD/MM/YYYY` | Block specific day |
| `timeslot` | `DD/MM/YYYY HH:MM` | Block specific timeslot |

#### Sheet: `teams_noplay`
Individual team restrictions

| Column | Format | Description |
|--------|--------|-------------|
| `team` | string | Full team name (e.g., `Maitland PHL`) |
| `whole_weekend` | `DD/MM/YYYY` | Block entire weekend |
| `whole_day` | `DD/MM/YYYY` | Block specific day |
| `timeslot` | `DD/MM/YYYY HH:MM` | Block specific timeslot |

#### Sheet: `team_conflicts`
Teams that cannot play at the same time (e.g., shared players)

| Column | Type | Description |
|--------|------|-------------|
| `team1` | string | First conflicting team |
| `team2` | string | Second conflicting team |

---

### 3. Field Availability (XLSX) - Optional

**Location:** `data/{year}/field_availability/{venue}_availability.xlsx`

Specifies when fields are unavailable.

---

### 4. Configuration Data (Hardcoded in main_staged.py)

The following data points are currently configured in `main_staged.py` and should be updated each season:

#### Playing Fields

```python
FIELDS = [
    PlayingField(location='Newcastle International Hockey Centre', name='SF'),  # South Field
    PlayingField(location='Newcastle International Hockey Centre', name='EF'),  # East Field
    PlayingField(location='Newcastle International Hockey Centre', name='WF'),  # West Field
    PlayingField(location='Maitland Park', name='Maitland Main Field'),
    PlayingField(location='Central Coast Hockey Park', name='Wyong Main Field'),
]
```

#### Game Times by Venue/Day

```python
day_time_map = {
    'Newcastle International Hockey Centre': {
        'Sunday': [tm(8, 30), tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30), tm(19, 0)]
    },
    'Maitland Park': {
        'Sunday': [tm(9, 0), tm(10, 30), tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    }
}
```

#### PHL-Specific Game Times

```python
phl_game_times = {
    'Newcastle International Hockey Centre': {
        'Friday': [tm(19, 0)], 
        'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
    },
    'Central Coast Hockey Park': {
        'Friday': [tm(20, 0)], 
        'Sunday': [tm(15, 0)]
    },
    'Maitland Park': {
        'Sunday': [tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    }
}
```

#### Season Dates

```python
start = datetime(2025, 3, 21)  # Season start date
end = datetime(2025, 9, 2)     # Season end date
max_rounds = 21                # Maximum number of rounds
```

#### Field Unavailabilities (by venue)

```python
field_unavailabilities = {
    'Maitland Park': {
        'weekends': [datetime(2025, 4, 19), datetime(2025, 4, 12), ...],  # Whole weekends blocked
        'whole_days': [datetime(2025, 4, 25)],                           # Specific days blocked
        'part_days': [],                                                  # Partial day blocks
    },
    'Newcastle International Hockey Centre': {
        'weekends': [datetime(2025, 4, 19), datetime(2025, 5, 3), ...],
        'whole_days': [datetime(2025, 4, 25), datetime(2025, 5, 31)],
        'part_days': [datetime(2025, 6, 1, 8, 30), datetime(2025, 6, 1, 10, 0), ...],
    },
    'Central Coast Hockey Park': {
        'weekends': [datetime(2025, 4, 19), datetime(2025, 4, 5), ...],
        'whole_days': [datetime(2025, 4, 25)],
        'part_days': [],
    },
}
```

#### Club Days (Special events)

```python
club_days = {
    'Crusaders': datetime(2025, 6, 22),
    'Wests': datetime(2025, 7, 13),
    'University': datetime(2025, 7, 27),
    'Tigers': datetime(2025, 7, 6),
    'Port Stephens': datetime(2025, 7, 20)
}
```

#### No-Play Preferences (Soft constraints)

```python
preference_no_play = {
    'Maitland': [
        {'date': '2025-07-20', 'field_location': 'Newcastle International Hockey Centre'},
        {'date': '2025-08-24', 'field_location': 'Newcastle International Hockey Centre'}
    ],
    'Norths': [
        {'team_name': 'Norths PHL', 'date': '2025-03-23', 'time': '11:30'},
        {'team_name': 'Norths PHL', 'date': '2025-03-23', 'time': '13:00'},
    ]
}
```

---

## File Structure

```
refactored/
├── main.py                     # Simple single-solve entry point
├── main_staged.py              # Staged solving with checkpoints
├── constraints.py              # Original constraint implementations (READ-ONLY — do not edit)
├── constraints_ai.py           # AI-enhanced constraint implementations (editable)
├── models.py                   # Pydantic data models
├── utils.py                    # Utility functions
├── generate_x.py               # Decision variable generation
├── draw_analytics.py           # Analytics and pliable JSON format
├── draw_tester.py              # Modification testing tool
├── DRAW_RULES.md               # Documentation of all scheduling rules
├── claude.md                   # AI assistant instructions
├── data/
│   └── 2025/
│       ├── teams/              # Team CSV files (required)
│       │   ├── maitland.csv
│       │   ├── souths.csv
│       │   └── ...
│       ├── noplay/             # Unavailability XLSX files (optional)
│       │   ├── maitland_noplay.xlsx
│       │   └── ...
│       └── field_availability/ # Field availability (optional)
├── draws/                      # Output schedules
├── checkpoints/                # Solver checkpoints
└── tests/                      # Test suite (216 tests)
    ├── test_constraints.py              # Original constraint unit tests
    ├── test_constraints_ai.py           # AI constraint unit tests
    ├── test_constraints_comprehensive.py # 70 comprehensive AI tests
    ├── test_constraints_equivalence.py  # AI vs original equivalence tests
    ├── test_analytics_*.py              # Analytics module tests
    ├── test_draw_outcomes.py            # Draw outcome tests
    └── test_utils.py                    # Utility function tests
```

---

## Running the System

### Using the CLI (Recommended)

```powershell
cd refactored

# Generate a new draw (staged solving - recommended)
python run.py generate --year 2025 --staged

# Generate with AI constraints (opt-in alternative constraint set)
python run.py generate --year 2025 --ai

# Generate with AI constraints in simple mode, excluding problematic constraints
python run.py generate --simple --ai --exclude EnsureBestTimeslotChoices MinimiseClubsOnAFieldBroadmeadow MaximiseClubsPerTimeslotBroadmeadow --year 2025 --workers 14

# Resume from a checkpoint
python run.py generate --year 2025 --staged --resume run_1

# Test an existing draw for violations
python run.py test draws/draw_2025.json

# Generate full analytics report
python run.py analyze draws/draw_2025.json

# Generate stakeholder Excel report
python run.py report draws/draw_2025.json --output analytics.xlsx

# Generate club-specific report
python run.py club-report draws/draw_2025.json Maitland --output reports/

# Generate compliance certificate
python run.py cert draws/draw_2025.json --output compliance.xlsx

# Get help
python run.py --help
```

### Legacy Commands

```powershell
# Direct staged solver
python main_staged.py

# Resume from checkpoint (legacy)
python main_staged.py --resume run_1 stage2_strong
```

---

## Output Files

After a successful solve, the following files are generated in `draws/`:

| File | Description |
|------|-------------|
| `schedule_{timestamp}.xlsx` | Traditional Excel schedule (one sheet per week) |
| `draw_{timestamp}.json` | Pliable JSON format for further processing |
| `analytics_{timestamp}.xlsx` | Comprehensive analytics workbook |
| `violations_{timestamp}.txt` | Violation report (if any issues found) |

### Analytics Workbook Sheets

| Sheet | Description |
|-------|-------------|
| `Summary` | Grade-level statistics |
| `Compliance Check` | Quick constraint compliance status |
| `Games Per Team` | Cross-tab of games per team per grade |
| `Home-Away Analysis` | Home/away/neutral breakdown per team |
| `Away Team Balance` | Maitland/Gosford home/away balance |
| `Matchup Matrix` | Team vs team matchup counts per grade |
| `Field Usage` | Field utilization per week |
| `Club-{name}` | Complete season schedule per club |
| `All Games` | Full game list sorted by week |

---

## Configuration

### Grade Order

The system assumes grades are ordered as:
```
PHL > 2nd > 3rd > 4th > 5th > 6th
```

### Home Field Mapping

| Club | Home Field |
|------|------------|
| Maitland | Maitland Park |
| Gosford | Central Coast Hockey Park |
| All others | Newcastle International Hockey Centre |

### Constraint Stages

Constraints are applied in 4 stages (see `DRAW_RULES.md` for details):

1. **Stage 1 - Required**: No double-booking, equal games (must satisfy)
2. **Stage 2 - Structural**: Home/away balance, adjacency rules
3. **Stage 3 - Optimization**: Venue efficiency, timeslot choices
4. **Stage 4 - Soft**: Preferences with penalties

---

## Troubleshooting

### Common Issues

**"Club {name} does not exist in clubs"**
- Ensure the club name in `{club}_noplay.xlsx` matches the `Club` column in team CSVs

**"Team {name} does not exist in games"**
- Check team name format: `{Team Name} {Grade}` (e.g., `Maitland PHL`)

**No solution found in Stage 1**
- Too many hard constraints; check field unavailabilities don't block all options

**Memory issues**
- Use staged solving with checkpoints
- Reduce `max_rounds` for initial testing

---

## Stakeholder Reports

### Club Reports

Generate a comprehensive report for a single club:

```powershell
python run.py club-report draws/draw_2025.json Maitland --output reports/
```

**Output includes:**
- All games for the club across all grades
- Home/away balance per team
- Bye weeks
- Opponent frequency
- Field usage summary

### Grade Reports

```python
from analytics.reports import GradeReport

report = GradeReport(draw, data, 'PHL')
report.generate_excel('phl_analysis.xlsx')
```

### Compliance Certificate

Verify the draw satisfies all constraints:

```powershell
python run.py cert draws/draw_2025.json --output compliance.xlsx
```

**Output includes:**
- Per-constraint pass/fail status
- Violation counts by severity
- Detailed violation list (if any)
- Official timestamp

---

## Testing

```powershell
cd refactored
pytest tests/ -v
```

---

## License

Internal use only.
