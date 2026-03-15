# Configuration Reference

> **Purpose:** Complete reference for all configurable parameters in the system.
> **When to use:** When you need to change any system behavior.

---

## Configuration File Location

All season-specific configuration lives in: `config/season_{year}.py`

---

## SEASON_CONFIG (Core Settings)

```python
SEASON_CONFIG = {
    'year': 2026,
    'start_date': datetime(2026, 3, 22),      # First playing Sunday
    'last_round_date': datetime(2026, 8, 30),  # Last regular season round
    'end_date': datetime(2026, 9, 19),         # Grand Final date
    
    'max_rounds': 22,           # Maximum games per team (see notes below)
    'num_dummy_timeslots': 3,   # Overflow slots for scheduling flexibility
    
    'play_anzac_sunday': True,  # Whether to play on ANZAC Sunday
    
    # Data paths (relative to project root)
    'teams_data_path': 'data/2026/teams',
    'noplay_data_path': 'data/2026/noplay',
    'field_availability_path': 'data/2026/field_availability',
    
    # References to other config sections
    'fields': FIELDS,
    'day_time_map': DAY_TIME_MAP,
    'phl_game_times': PHL_GAME_TIMES,
    'second_grade_times': SECOND_GRADE_TIMES,
    'field_unavailabilities': FIELD_UNAVAILABILITIES,
    'club_days': CLUB_DAYS,
    'phl_preferences': PHL_PREFERENCES,
    'preference_no_play': PREFERENCE_NO_PLAY,
    'friday_night_config': FRIDAY_NIGHT_CONFIG,
    'special_games': SPECIAL_GAMES,
}
```

### max_rounds Parameter

**What it does:**
- Sets the maximum number of games any team can play in the season
- Used by `max_games_per_grade()` to calculate actual games per grade
- Acts as a cap: `actual_games = min(calculated, max_rounds)`

**How to set it:**
- Count available playing weekends (total Sundays minus blocked)
- Set max_rounds to this number
- Example: 24 Sundays - 2 blocked = 22 available → `max_rounds: 22`

**Grade interaction:**
- Lower grades with fewer teams may play fewer games naturally
- PHL with 5 teams: 4 matchups × 5 rounds = 20 games (capped by math, not max_rounds)
- 4th grade with 11 teams: Each pair meets 2× = 20 games per team

---

## FIELDS (Playing Surfaces)

```python
FIELDS = [
    {'location': 'Newcastle International Hockey Centre', 'name': 'SF'},  # South Field
    {'location': 'Newcastle International Hockey Centre', 'name': 'EF'},  # East Field
    {'location': 'Newcastle International Hockey Centre', 'name': 'WF'},  # West Field
    {'location': 'Maitland Park', 'name': 'Maitland Main Field'},
    {'location': 'Central Coast Hockey Park', 'name': 'Wyong Main Field'},
]
```

**Field codes:**
- SF = South Field (NIHC) - Lower grades only, PHL excluded
- EF = East Field (NIHC) - All grades
- WF = West Field (NIHC) - All grades

---

## DAY_TIME_MAP (Base Timeslots)

```python
DAY_TIME_MAP = {
    'Newcastle International Hockey Centre': {
        'Sunday': [tm(8, 30), tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30), tm(19, 0)]
    },
    'Maitland Park': {
        'Sunday': [tm(9, 0), tm(10, 30), tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    },
    'Central Coast Hockey Park': {
        'Sunday': [tm(12, 0), tm(13, 30)],  # 12pm or 1:30pm only
    }
}
```

**Purpose:**
- Defines ALL possible timeslots for lower grades (3rd-6th)
- PHL and 2nd grade have their own filtering dicts (see below)

---

## PHL_GAME_TIMES (PHL Variable Filtering)

**Purpose:** Controls which decision variables are created for PHL games.

```python
PHL_GAME_TIMES = {
    'Newcastle International Hockey Centre': {
        'EF': {  # East Field only (SF excluded for PHL)
            'Friday': [tm(19, 0)],  # 7pm
            'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
        },
        'WF': {  # West Field only
            'Friday': [tm(19, 0)],
            'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
        },
        # NOTE: SF deliberately excluded
    },
    'Central Coast Hockey Park': {
        'Wyong Main Field': {
            'Friday': [tm(20, 0)],  # 8pm (AGM confirmed)
            'Sunday': [tm(12, 0), tm(13, 30)]
        },
    },
    'Maitland Park': {
        'Maitland Main Field': {
            'Sunday': [tm(12, 0), tm(13, 0), tm(15, 0), tm(16, 30)]
        },
    },
}
```

**Key rules:**
- PHL cannot play on South Field (SF) - excluded from dict
- Gosford is PHL-only venue (lower grades excluded)
- Friday nights only at NIHC (7pm) and Gosford (8pm)

**Effect:** Only ~2,000 PHL variables created vs ~20,000+ if unrestricted.

---

## SECOND_GRADE_TIMES (2nd Grade Variable Filtering)

**Purpose:** Controls which decision variables are created for 2nd grade.

```python
SECOND_GRADE_TIMES = {
    'Newcastle International Hockey Centre': {
        'EF': {  # No SF
            'Sunday': [tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30)]
        },
        'WF': {
            'Sunday': [tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30)]
        },
    },
    # Gosford NOT listed - PHL-only venue
    'Maitland Park': {
        'Maitland Main Field': {
            'Sunday': [tm(10, 30), tm(12, 0), tm(13, 0), tm(15, 0), tm(16, 30)]
        },
    },
}
```

**Key rules:**
- 2nd grade cannot play at Gosford (not listed)
- Times are PHL times ± 1 slot (for back-to-back scheduling)
- Cannot create NEW timeslots - must exist in DAY_TIME_MAP

---

## FIELD_UNAVAILABILITIES (Blocked Dates)

```python
FIELD_UNAVAILABILITIES = {
    'Newcastle International Hockey Centre': {
        'weekends': [
            datetime(2026, 4, 4),   # Easter
            datetime(2026, 5, 16),  # Masters SC
        ],
        'whole_days': [datetime(2026, 4, 25)],  # ANZAC Day
        'part_days': [],  # For future: specific timeslots blocked
    },
    ...
}
```

**Effect:** Completely removes all game variables for these dates - HARD constraint.

---

## FRIDAY_NIGHT_CONFIG

```python
FRIDAY_NIGHT_CONFIG = {
    'gosford_friday_count': 8,       # Total Friday games at Gosford (AGM decision)
    'friday_clubs': {                # Which clubs play at Gosford Fridays
        'Wests': 2,
        'Souths': 2,
        'Norths': 1,
        'Tigers': 2,
        'Maitland': 1,
    },
    'friday_dates': [...],           # Confirmed Friday dates
    'gosford_friday_times': [tm(20, 0)],  # 8pm
    'nihc_friday_times': [tm(19, 0)],     # 7pm
}
```

---

## PREFERENCE_NO_PLAY (Soft No-Play Requests)

```python
PREFERENCE_NO_PLAY = {
    'Crusaders_6th_Masters': {
        'club': 'Crusaders',
        'grade': '6th',           # Single grade
        'dates': [datetime(2026, 4, 17), ...],
        'reason': 'Masters State Championships',
    },
    'Souths_U18_SC': {
        'club': 'Souths',
        'grades': ['PHL', '2nd'],  # Multiple grades
        'dates': [datetime(2026, 5, 24)],
        'reason': 'U18 State Championships',
    },
}
```

**Effect:** Adds penalty to PreferredTimesConstraint (severity 4) - SOFT constraint.

---

## CLUB_DAYS (Special Events)

```python
CLUB_DAYS = {
    'Crusaders': datetime(2026, 6, 14),  # All teams play back-to-back, same field
}
```

**Effect:** ClubDayConstraint ensures all club teams play on that date, on same field, in contiguous slots.

---

## PHL_PREFERENCES

```python
PHL_PREFERENCES = {
    'preferred_dates': [],  # Specific date preferences for PHL games
}
```

**Note:** Only `preferred_dates` key is supported. Other PHL rules are enforced via constraints, not this dict.

---

## SPECIAL_GAMES

```python
SPECIAL_GAMES = {
    'taree_game': {
        'teams': ['Tigers', 'Souths'],
        'grades': ['PHL', '2nd'],
        'month': 5,  # May
        'date': None,  # TBC
    },
}
```

**Note:** Currently informational only. Actual scheduling requires manual intervention or constraint addition.

---

## FORCED_GAMES (Partial Key Variable Elimination)

Forces specific matchups on specific dates/venues by eliminating all non-matching variables from the solver.

```python
FORCED_GAMES = [
    # Each entry is a partial key dict. Fields split into:
    #   Scope fields: grade, day, day_slot, time, week, date, round_no, field_name, field_location
    #   Game fields:  teams (list of 1-2 club names)
    #
    # Variables matching ALL scope fields but NOT matching game fields → eliminated.
    # Multiple entries with the same scope are OR'd (any match keeps the var).
    
    # Example: Force exact matchup on a specific Friday
    {
        'teams': ['Maitland', 'Souths'],      # Club names (auto-resolved via grade)
        'grade': 'PHL',
        'date': '2026-05-08',
        'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'description': 'NIHC Friday Night - Souths vs Maitland',
    },
    
    # Example: Any opponent for a specific team
    {
        'teams': ['Norths'],                   # Single team = Norths must be involved
        'grade': 'PHL',
        'date': '2026-07-24',
        'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'description': 'NIHC Friday Night - Norths home (opponent TBC)',
    },
]
```

### Field Reference

| Field | Type | Maps to Key Index | Description |
|-------|------|-------------------|-------------|
| `teams` | list | 0, 1 (team1, team2) | 1-2 club names. Auto-resolved to full team names using grade. |
| `grade` | str/list | 2 | Grade name(s). Also used to resolve club names to team names. |
| `day` | str | 3 | Day of week (e.g., 'Friday', 'Sunday') |
| `day_slot` | int | 4 | Time slot index |
| `time` | str | 5 | Game time (e.g., '19:00') |
| `week` | int | 6 | Week number |
| `date` | str | 7 | Date string 'YYYY-MM-DD' |
| `round_no` | int | 8 | Round number |
| `field_name` | str | 9 | Field name (e.g., 'EF', 'WF') |
| `field_location` | str | 10 | Venue name |
| `description` | str | — | Logging only, not used for matching |

### How It Works

1. **Scope** = all specified fields EXCEPT teams (grade, date, day, venue, etc.)
2. **Game** = teams (resolved to full team names using grade + teams list)
3. For each variable in `generate_X()`:
   - If it matches ALL scope fields: check if teams also match
   - If teams DON'T match → variable eliminated (not created)
   - If scope doesn't match → variable unaffected

### Team Name Resolution

Club names in `teams` are auto-resolved to full team names:
- `'Maitland'` + `grade='PHL'` → `'Maitland PHL'`
- `'Colts'` + `grade='5th'` → `['Colts Gold 5th', 'Colts Green 5th']` (both teams match)

### Implementation

- Config: `FORCED_GAMES` list in `config/season_{year}.py`
- Passed through: `SEASON_CONFIG['forced_games']` → `build_season_data()` → `data['forced_games']`
- Filtering: `_build_forced_game_rules()` and `_is_blocked_by_forced_games()` in `utils.py`
- Called from: `generate_X()` — checked for each variable before creation
