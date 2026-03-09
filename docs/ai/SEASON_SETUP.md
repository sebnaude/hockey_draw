# Season Setup Guide

> **Purpose:** Complete checklist for setting up a new season.
> **When to use:** At the start of each season, before generating the first draw.

---

## Pre-Season Information Gathering

### Required Information (Must Have)

| Information | Source | Config Location |
|-------------|--------|-----------------|
| Season start date | HNH Committee | `SEASON_CONFIG['start_date']` |
| Season end date | HNH Committee | `SEASON_CONFIG['last_round_date']` |
| Grand final date | HNH Committee | `SEASON_CONFIG['end_date']` |
| Team nominations by club | Club submissions | `data/{year}/teams/*.csv` |
| Blocked weekends | HNH Committee | `FIELD_UNAVAILABILITIES` |
| Number of rounds | HNH Committee/AGM | `SEASON_CONFIG['max_rounds']` |

### PHL-Specific Information

| Information | Source | Config Location |
|-------------|--------|-----------------|
| Friday night games at Gosford (count) | AGM decision | `FRIDAY_NIGHT_CONFIG['gosford_friday_count']` |
| Friday night dates at Gosford | Club agreements | `FRIDAY_NIGHT_CONFIG['friday_dates']` |
| Which clubs play Friday at Gosford | Club agreements | `FRIDAY_NIGHT_CONFIG['friday_clubs']` |
| Gosford Friday start time | AGM decision | `FRIDAY_NIGHT_CONFIG['gosford_friday_times']` |
| NIHC Friday start time | HNH Committee | `FRIDAY_NIGHT_CONFIG['nihc_friday_times']` |

### Club Requests (Soft Constraints)

| Information | Source | Config Location |
|-------------|--------|-----------------|
| Club days | Club requests | `CLUB_DAYS` dict |
| No-play requests | Club requests | `PREFERENCE_NO_PLAY` dict |
| Special games (e.g., Taree) | Club requests | `SPECIAL_GAMES` dict |

---

## Step-by-Step Setup Process

### Step 1: Create Season Config File

```bash
# Copy template
cp config/season_template.py config/season_{year}.py
```

Update all `9999` year references to the actual year.

### Step 2: Update Team Data

Create/update `data/{year}/teams/*.csv` files for each club:

```csv
Club,Grade,Team Name
Maitland,PHL,Maitland PHL
Maitland,2nd,Maitland 2nd
Maitland,3rd,Maitland
...
```

**Team naming rules:**
- Single team in grade: Just club name (e.g., "Maitland")
- Multiple teams in grade: Club name + identifier (e.g., "Tigers", "Tigers Black")
- See `config/team_naming.py` for helpers

### Step 3: Update Season Dates

In `SEASON_CONFIG`:

```python
SEASON_CONFIG = {
    'year': 2026,
    'start_date': datetime(2026, 3, 22),      # First playing Sunday
    'last_round_date': datetime(2026, 8, 30),  # Last regular round
    'end_date': datetime(2026, 9, 19),         # Grand Final
    'max_rounds': 22,  # See "Understanding max_rounds" below
    ...
}
```

### Step 4: Update Blocked Weekends

In `FIELD_UNAVAILABILITIES`:

```python
FIELD_UNAVAILABILITIES = {
    'Newcastle International Hockey Centre': {
        'weekends': [
            datetime(2026, 4, 4),   # Easter
            datetime(2026, 5, 16),  # Masters SC
            ...
        ],
        'whole_days': [datetime(2026, 4, 25)],  # ANZAC Day
        'part_days': [],
    },
    ...
}
```

### Step 5: Update Game Time Dictionaries

See `GAME_TIME_DICTIONARIES.md` for full details.

**Quick summary:**
- `PHL_GAME_TIMES` → Controls which slots have PHL variables
- `SECOND_GRADE_TIMES` → Controls which slots have 2nd grade variables
- `DAY_TIME_MAP` → Base timeslots for all other grades

### Step 6: Add Club Requests

**No-play requests (soft):**
```python
PREFERENCE_NO_PLAY = {
    'Crusaders_6th_Masters': {
        'club': 'Crusaders',
        'grade': '6th',
        'dates': [datetime(2026, 4, 17), ...],
        'reason': 'Masters State Championships',
    },
}
```

**Club days:**
```python
CLUB_DAYS = {
    'Crusaders': datetime(2026, 6, 14),
}
```

### Step 7: Run Pre-Season Report

```powershell
.\.venv\Scripts\python.exe run.py preseason --year 2026
```

Verify:
- ✅ Correct number of teams per grade
- ✅ Correct number of available weekends
- ✅ Blocked weekends correctly excluded
- ✅ Club requests listed

### Step 8: Generate Draw

```powershell
# ALWAYS use isBackground: true for this command
.\.venv\Scripts\python.exe run.py generate --year 2026
```

---

## Understanding max_rounds

### How It Works

`max_rounds` is the **default cap** on games per team per season:

```python
# In utils.py
def max_games_per_grade(grades: List, max_rounds: int) -> Dict[str, int]:
    """Calculate max games per team for each grade."""
    for grade in grades:
        T = len(teams_in_grade)
        # Each team can play at most max_rounds games
        g0 = min(calculated_games, max_rounds)
```

### Relationship to Available Weekends

- **Available weekends** = Total Sundays - Blocked weekends
- **max_rounds** should be ≤ available weekends
- Setting `max_rounds` higher than available weekends has no effect

### Grade-Specific Overrides

Currently there is **no per-grade override** for max_rounds. The system uses:

```
num_rounds[grade] = min(calculated_from_team_count, max_rounds)
```

**Future enhancement:** Add `GRADE_ROUND_OVERRIDES` dict to allow:
```python
GRADE_ROUND_OVERRIDES = {
    'PHL': 22,   # Force exactly 22 rounds
    '6th': 18,   # Cap at 18 rounds
}
```

### 2026 Configuration

For 2026 with 22 available Sundays:
- Set `max_rounds: 22` to allow up to 22 games
- PHL with 5 teams: Each team plays 20 games (5-1) * 5 rounds = 20
- Lower grades calculate their own based on team counts

---

## Verification Checklist

Before generating:

- [ ] All team CSV files created in `data/{year}/teams/`
- [ ] `SEASON_CONFIG` dates set correctly
- [ ] `FIELD_UNAVAILABILITIES` updated for blocked weekends
- [ ] `PHL_GAME_TIMES` and `SECOND_GRADE_TIMES` reviewed
- [ ] `FRIDAY_NIGHT_CONFIG` set for Gosford/NIHC Friday nights
- [ ] `PREFERENCE_NO_PLAY` includes all soft no-play requests
- [ ] `CLUB_DAYS` includes all club day events
- [ ] Pre-season report shows correct team counts
- [ ] Pre-season report shows correct weekend count
