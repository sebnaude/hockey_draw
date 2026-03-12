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
    'max_weekends_per_grade': MAX_WEEKENDS_PER_GRADE,
    'grade_rounds_override': GRADE_ROUNDS_OVERRIDE,
}
```

---

## Rounds Configuration (CRITICAL)

> ⚠️ **See `SEASON_SETUP.md` for full explanation of Available vs Played rounds.**

### max_rounds (Default Fallback)

```python
'max_rounds': 20,  # Default max weekends if grade not in MAX_WEEKENDS_PER_GRADE
```

### MAX_WEEKENDS_PER_GRADE (Per-Grade Maximum)

Sets the **maximum available weekends** (hard cap) for each grade:

```python
MAX_WEEKENDS_PER_GRADE = {
    'PHL': 22,   # 20 Sundays + 2 rescued via Friday (NOT 20+8!)
    '2nd': 20,   # 20 Sundays only
    '3rd': 20,
    '4th': 20,
    '5th': 20,
    '6th': 20,
}
```

**⚠️ FRIDAY NIGHTS:**
- Friday games are **part of the weekend**, NOT additional weekends
- Friday at Gosford "rescues" weekends blocked for Sunday (e.g., State Championships)
- PHL gets 22 weekends = 20 normal + 2 rescued (NOT 20 + 8 Friday nights!)

### Games Per Team Calculation

The system automatically calculates the maximum feasible games per team using:

```python
# Formula (in utils.py max_games_per_grade):
max_matches = W * floor(T/2)     # Total possible games across all weekends
g0 = floor(2 * max_matches / T)  # Games per team
g0 = min(g0, W)                  # Can't exceed weekends
if T is odd and g0 is odd:
    g0 -= 1                      # Force even for odd team counts
```

**Key insight:** For odd team counts, one team must sit out each weekend, so maximum games < available weekends.

**The `EnsureEqualGamesAndBalanceMatchUps` constraint then enforces:**
- Each team plays exactly `g0` games
- Each pair meets `base` or `base+1` times (where `base = g0 // T` for odd teams)

### GRADE_ROUNDS_OVERRIDE (Hard Override)

Force exact round count, bypassing all formulas:

```python
GRADE_ROUNDS_OVERRIDE = {
    # '2nd': 18,  # Force exactly 18 rounds
}
```

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
    'friday_dates': [...],           # Confirmed Friday dates at Gosford
    'gosford_friday_times': [tm(20, 0)],  # 8pm
    'nihc_friday_times': [tm(19, 0)],     # 7pm
    
    # NIHC (Broadmeadow) Friday night matchup restrictions (2026+)
    'nihc_friday_games': {
        '2026-05-08': [('Maitland', 'Souths')],  # Only Souths vs Maitland
        '2026-06-19': [('Tigers', 'Wests')],     # Only Tigers vs Wests
        '2026-07-24': 'norths_only',              # Any matchup with Norths
    },
}
```

### nihc_friday_games (NIHC Matchup Filtering)

This dict controls which PHL matchups are allowed at NIHC/Broadmeadow on Friday nights:

| Date Key | Value | Effect |
|----------|-------|--------|
| `'YYYY-MM-DD'` | `[('Club1', 'Club2')]` | Only listed matchups get variables |
| `'YYYY-MM-DD'` | `'norths_only'` | Any matchup including Norths allowed |
| Date NOT in dict | - | NO games allowed at NIHC that Friday |

**Rules:**
- Club names must be alphabetically sorted in tuples: `('Maitland', 'Souths')` not `('Souths', 'Maitland')`
- Filtering happens in `generate_X()` - only listed matchups create decision variables
- Only dates in the dict can have Friday games at NIHC; all other Fridays are blocked

---

## PREFERENCE_NO_PLAY (Soft No-Play Requests)

Two formats are supported:

### 2026 Format (Recommended - Structured)

```python
PREFERENCE_NO_PLAY = {
    'Crusaders_6th_Masters': {       # Descriptive key (any unique identifier)
        'club': 'Crusaders',          # REQUIRED: Club name (must match clubs list)
        'grade': '6th',               # OPTIONAL: Single grade filter
        'grades': ['PHL', '2nd'],     # OPTIONAL: Multiple grades filter (use one or other)
        'dates': [datetime(2026, 4, 17), datetime(2026, 4, 18)],  # REQUIRED: List of dates
        'reason': 'Masters SC',       # OPTIONAL: Documentation only
    },
    'Souths_U18_SC': {
        'club': 'Souths',
        'grades': ['PHL', '2nd'],
        'dates': [datetime(2026, 5, 24)],
        'reason': 'U18 State Championships',
    },
    'Gosford_Recovery': {
        'club': 'Gosford',            # No grade filter = all club teams affected
        'dates': [datetime(2026, 6, 21)],
        'reason': 'Recovery after Men\'s SC',
    },
}
```

### 2025 Format (Legacy - Still Supported)

```python
PREFERENCE_NO_PLAY = {
    'Maitland': [                     # Key IS the club name
        {'date': '2025-07-20', 'field_location': 'Newcastle International Hockey Centre'},
        {'date': '2025-08-24', 'field_location': 'Newcastle International Hockey Centre'},
    ],
    'Norths': [
        {'team_name': 'Norths PHL', 'date': '2025-03-23', 'time': '11:30'},
    ],
}
```

**Effect:** Adds penalty (weight 10,000,000) to PreferredTimesConstraint (severity 4) - SOFT constraint.

**Key Differences:**
| Field | 2026 Format | 2025 Format |
|-------|-------------|-------------|
| Key | Descriptive identifier | Club name |
| Club name | `'club': 'X'` inside value | The key itself |
| Dates | `'dates': [datetime(...)]` | `'date': 'YYYY-MM-DD'` per entry |
| Grade filter | `'grade'` or `'grades'` | `'team_name'` per entry |

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

## CLI Constraint Slack (--slack N)

Runtime relaxation of constraint limits via the `--slack N` CLI argument.

### How It Works

1. User runs: `python run.py generate --year 2026 --slack 1`
2. `run.py` builds `constraint_slack` dict with N for each supported constraint
3. Dict passed through `build_season_data()` → `data['constraint_slack']`
4. Constraints read: `data.get('constraint_slack', {}).get('ConstraintName', 0)`

### Supported Constraints

| Constraint | Config Key | Base Limit | With --slack N |
|------------|------------|------------|----------------|
| `EqualMatchUpSpacingConstraint` | `EqualMatchUpSpacing` | ±1 round | ±(1+N) rounds |
| `AwayAtMaitlandGrouping` | `AwayAtMaitlandGrouping` | Max 3 away clubs | Max 3+N |
| `MaitlandHomeGrouping` | `MaitlandHomeGrouping` | 0 back-to-back | N allowed |

### constraint_slack Dict Format

```python
# Built automatically by run.py when --slack N is used
data['constraint_slack'] = {
    'EqualMatchUpSpacing': N,
    'AwayAtMaitlandGrouping': N,
    'MaitlandHomeGrouping': N,
}
```

### Implementation Files

| Component | Location |
|-----------|----------|
| CLI argument | `run.py` lines 99-101 |
| Dict builder | `run.py` lines 256-267 |
| Data passing | `utils.py` → `build_season_data()` |
| Original constraints | `constraints/original.py` |
| AI constraints | `constraints/ai.py` |

### Adding New Slack-Enabled Constraints

To make a constraint configurable via `--slack`:

1. Add the constraint name to the dict builder in `run.py`:
   ```python
   constraint_slack = {
       'ExistingConstraint': slack_value,
       'YourNewConstraint': slack_value,  # Add here
   }
   ```

2. In the constraint's `apply()` method, read from slack:
   ```python
   def apply(self, model, data, X):
       slack = data.get('constraint_slack', {}).get('YourNewConstraint', 0)
       actual_limit = self.BASE_LIMIT + slack
       # Use actual_limit instead of hardcoded value
   ```

3. Update both `original.py` and `ai.py` if both versions exist.
