# Copilot Instructions for Hockey Draw Scheduler

## IMPORTANT: Always Load Skill File First

**Before performing any action on this codebase, ALWAYS read and understand the skill file:**

📁 `.github/copilot-skills/hockey-draw-scheduler.md`

This skill file contains critical information about:
- How to run the solver correctly
- Expected long running times (hours to days)
- Memory and CPU management options
- Checkpoint/resume system
- Common issues and solutions

## Critical Rules

### 1. Solver Execution
- **NEVER** run solver commands with `isBackground: false` - they can run for HOURS
- **ALWAYS** use `isBackground: true` for generate commands
- **CHECK** terminal output periodically with `get_terminal_output`

### 2. Long-Running Operations
Solver stages have these time limits:
- Stage 1: 2 hours
- Stage 2: 4 hours  
- Stage 3: 8 hours
- Stage 4: 72 hours (3 days!)

### 3. Memory Management
If encountering crashes without Python tracebacks:
1. Check logs in `logs/` directory
2. Recommend `--low-memory` flag
3. Suggest reducing workers: `--workers 4`

### 4. OR-Tools API
- Use `solver.status_name(status)` NOT `solver.StatusName()` (deprecated)
- Always check ortools version compatibility

### 5. File Locations
- Team data: `data/2025/teams/*.csv`
- Output schedules: `draws/`
- Checkpoints: `checkpoints/run_X/`
- Solver logs: `logs/`

## Quick Commands Reference

```powershell
# Standard generate (background)
.\.venv\Scripts\python.exe run.py generate --year 2025

# Low memory mode
.\.venv\Scripts\python.exe run.py generate --year 2025 --low-memory

# Resume from checkpoint
.\.venv\Scripts\python.exe run.py generate --year 2025 --resume run_13 stage1_required

# Test draw
.\.venv\Scripts\python.exe run.py test draws/draw.json

# Analyze draw
.\.venv\Scripts\python.exe run.py analyze draws/draw.json
```

## When User Asks About This Project

1. **First**: Read the skill file
2. **Then**: Understand the specific request
3. **Check**: Recent checkpoints and logs for context
4. **Recommend**: Appropriate memory/worker settings based on available resources

## Data File Formats

### Team CSV Files (`data/{year}/teams/{club}.csv`)
```csv
Club,Grade,Team Name
Maitland,PHL,Maitland
Maitland,2nd,Maitland
Colts,5th,Colts Gold
Colts,5th,Colts Green
```
- **Club**: Club name (matches filename without .csv)
- **Grade**: One of `PHL`, `2nd`, `3rd`, `4th`, `5th`, `6th`
- **Team Name**: Display name (can differ for multiple teams in same grade)

### No-Play Data

#### Option 1: XLSX Files (`data/{year}/noplay/{club}_noplay.xlsx`)
Excel files with 3 sheets:

**Sheet: `club_noplay`** - Club-wide restrictions (all teams affected)
| Column | Format | Description |
|--------|--------|-------------|
| `whole_weekend` | `DD/MM/YYYY` | Block entire weekend |
| `whole_day` | `DD/MM/YYYY` | Block specific day |
| `timeslot` | `DD/MM/YYYY HH:MM` | Block specific timeslot |

**Sheet: `teams_noplay`** - Individual team restrictions
| Column | Format | Description |
|--------|--------|-------------|
| `team` | string | Full team name (e.g., `Maitland PHL`) |
| `whole_weekend` | `DD/MM/YYYY` | Block entire weekend |
| `whole_day` | `DD/MM/YYYY` | Block specific day |
| `timeslot` | `DD/MM/YYYY HH:MM` | Block specific timeslot |

**Sheet: `team_conflicts`** - Teams that cannot play at same time
| Column | Description |
|--------|-------------|
| `team1` | First conflicting team |
| `team2` | Second conflicting team |

#### Option 2: Python Config (`config/season_{year}.py`)
Use `PREFERENCE_NO_PLAY` dictionary for soft constraints:
```python
PREFERENCE_NO_PLAY = {
    'Crusaders_6th_Masters': {
        'club': 'Crusaders',
        'grade': '6th',
        'dates': [datetime(2026, 4, 17), datetime(2026, 4, 18)],
        'reason': 'NSW Masters Championships',
    },
}
```

### Season Config (`config/season_{year}.py`)
Contains all season-specific settings:
- `FIELDS` - Playing field definitions
- `DAY_TIME_MAP` - Standard game times by venue/day
- `PHL_GAME_TIMES` - PHL-specific game times
- `FIELD_UNAVAILABILITIES` - Venue closures
- `CLUB_DAYS` - Club day events (back-to-back games)
- `PREFERENCE_NO_PLAY` - Soft no-play constraints
- `SEASON_CONFIG` - Main config dict with dates, paths, etc.

### Team Naming Conventions (`config/team_naming.py`)
When a club has multiple teams in the same grade, use these standard names:

| Club | First Team | Second Team | Notes |
|------|------------|-------------|-------|
| Tigers | Tigers | Tigers Black | Black & Yellow |
| Wests | Wests | Wests Red | Red & Green |
| University | Uni | Uni Seapigs | Seapigs variant |
| Colts | Colts Gold | Colts Green | Gold & Green |

Default for unlisted clubs: `ClubName`, `ClubName 2`, etc.

Use the helper functions:
```python
from config.team_naming import get_team_names, get_team_name

# Get names for 2 teams
names = get_team_names('Tigers', count=2)  # ['Tigers', 'Tigers Black']

# Get specific team name
name = get_team_name('Colts', index=1)  # 'Colts Green'
```

### Documentation File (`data/{year}/documentation.txt`)
Notes on team naming conventions and other clarifications.
