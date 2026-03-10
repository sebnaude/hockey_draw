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

## Understanding Rounds: Available vs Played

> ⚠️ **CRITICAL DISTINCTION:** There are TWO different round concepts that MUST NOT be conflated.

### Concept 1: Maximum Available Weekends (Hard Cap)

This is the **maximum number of weekends a grade CAN play** - a hard ceiling.

```python
MAX_WEEKENDS_PER_GRADE = {
    'PHL': 22,   # 20 Sundays + 2 extra via Friday nights (rescued weekends)
    '2nd': 20,   # 20 Sundays only
    '3rd': 20,   # etc.
}
```

**How it's calculated:**
- Start with total calendar Sundays in season
- Subtract blocked weekends (Easter, State Championships, etc.)
- For PHL: Add weekends "rescued" by Friday night option (see below)
- Can be manually overridden to be LOWER than calculated

**⚠️ FRIDAY NIGHTS ARE NOT EXTRA WEEKENDS:**
- A Friday night game at Gosford is **part of that weekend**, NOT an additional weekend
- If PHL plays Friday at Gosford, they don't also play Sunday that week
- Friday games "rescue" weekends that would otherwise be blocked (e.g., State Championships on Saturday/Sunday)
- Example: May 15-17 Masters SC blocks Sunday play, but PHL can play Friday at Gosford instead → weekend is still playable

### Concept 2: Actual Played Rounds (Calculated)

This is **how many games each team actually plays**, calculated from:
- Number of teams in the grade
- Maximum available weekends
- The scheduling formula chosen

**Two Formula Options:**

#### Formula 1: Strict Equal Matchups (Default)
Every matchup occurs exactly the same number of times. May result in "no-play" weekends.

```
T = number of teams
W = max available weekends

Even teams: games = floor(W / (T-1)) × (T-1)
Odd teams:  games = floor(W / T) × (T-1)
```

Example: 18 weekends, 6 teams → 15 games (each opponent 3×), 3 no-play weekends

### Games Formula (Automatic Calculation)

The system automatically calculates maximum feasible games using:

```python
# In utils.py max_games_per_grade():
max_matches = W * floor(T/2)     # Total games possible across all weekends  
g0 = floor(2 * max_matches / T)  # Max games per team
g0 = min(g0, W)                  # Can't exceed available weekends
if T is odd and g0 is odd:
    g0 -= 1                      # Force even for odd team counts
```

**Key insight for odd teams:** One team must sit out each weekend, so max games < weekends.

The `EnsureEqualGamesAndBalanceMatchUps` constraint enforces:
- Each team plays exactly `g0` games
- Each pair meets `base` or `base+1` times (allowing slight variation to maximize games)

### Configuration

```python
# Set max available weekends per grade
MAX_WEEKENDS_PER_GRADE = {
    'PHL': 22,   # 20 Sundays + 2 rescued via Friday
    '2nd': 20,
    '3rd': 20,
    # ...
}

# Optional: Force exact round count (overrides formula entirely)
GRADE_ROUNDS_OVERRIDE = {
    # '2nd': 18,  # Uncomment to force exactly 18 rounds
}
```

### Quick Reference

| Concept | Config Key | What it Controls |
|---------|-----------|------------------|
| Max available weekends | `MAX_WEEKENDS_PER_GRADE` | Hard ceiling on playable games |
| Exact override | `GRADE_ROUNDS_OVERRIDE` | Force specific round count |

### 2026 Configuration

- 24 total Sundays (Mar 22 - Aug 30)
- 4 blocked weekends (Easter, Masters SC, U16 Girls SC)
- = 20 available Sundays for most grades
- PHL: 22 available (20 + 2 rescued via Friday at Gosford)
- PHL with 6 teams: Each plays 22 games (4-5× vs each opponent)

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
- [ ] **`seasons/{year}/{year}_club_requests.md` created** with all implementations tracked
