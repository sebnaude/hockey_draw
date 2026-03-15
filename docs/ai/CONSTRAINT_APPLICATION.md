# Constraint Application Guide

> **Purpose:** How to apply different types of constraints and restrictions.
> **When to use:** When someone requests a scheduling restriction.

---

## Constraint Types Overview

| Type | Enforcement | Breaks Solver? | When to Use |
|------|-------------|----------------|-------------|
| **Hard (Variable Removal)** | Variables not created | No (reduces search space) | Absolute restrictions |
| **Hard (Constraint)** | Solver enforces | Yes if impossible | Must-have rules |
| **Soft (Penalty)** | Penalty if violated | No | Preferences |

---

## No-Play Requests

When someone says "Team X can't play on Date Y", you have **two options**:

### Option 1: Soft Constraint (Recommended for most cases)

**Use when:** Request is a preference, player availability issue, or could be overridden if necessary.

**How to apply:**

1. Add to `PREFERENCE_NO_PLAY` in `config/season_{year}.py`:

```python
PREFERENCE_NO_PLAY = {
    'Crusaders_6th_Masters': {       # Descriptive key (unique identifier)
        'club': 'Crusaders',          # REQUIRED: Club name
        'grade': '6th',               # OPTIONAL: Filter to specific grade
        'dates': [datetime(2026, 4, 17), datetime(2026, 4, 18), datetime(2026, 4, 19)],
        'reason': 'Masters State Championships',  # OPTIONAL: Documentation
    },
}
```

**Format Options:**
- `'grade': '6th'` - affects only that grade
- `'grades': ['PHL', '2nd']` - affects multiple grades
- No grade field - affects ALL club teams

2. This adds a penalty (10,000,000 weight) via `PreferredTimesConstraint` (severity 4).

**Behavior:** Solver will try to avoid these dates, but may use them if necessary.

### Option 2: Hard Constraint (XLSX Variable Removal)

**Use when:** Absolutely cannot play - venue closed, team doesn't exist that day, etc.

**How to apply:**

1. Create/update `data/{year}/noplay/{club}_noplay.xlsx` with THREE sheets:

**Sheet: `club_noplay`** (blocks ALL club teams)
| whole_weekend | whole_day | timeslot |
|---------------|-----------|----------|
| 15/06/2026 | | |
| | 14/06/2026 | |

**Sheet: `teams_noplay`** (blocks specific teams)
| team | whole_weekend | whole_day | timeslot |
|------|---------------|-----------|----------|
| Crusaders 6th | 14/06/2026 | | |
| Crusaders PHL | | 15/06/2026 | |

**Sheet: `team_conflicts`** (teams that can't play at same time)
| team1 | team2 |
|-------|-------|
| Crusaders 5th | Crusaders 6th |

**Column meanings:**
- `whole_weekend`: Blocks entire weekend (by week number) - format `DD/MM/YYYY`
- `whole_day`: Blocks specific calendar day - format `DD/MM/YYYY`
- `timeslot`: Blocks specific slot - format `DD/MM/YYYY HH:MM`

2. The system reads this at variable creation time and **excludes those games entirely**.

**Behavior:** No game variables are created for that team on those dates.

### Decision Guide

**Ask the user:**
> "Should this be a hard restriction (team absolutely cannot play) or a soft preference (try to avoid but can override if needed)?"

| User Says | Apply |
|-----------|-------|
| "They can't play, they won't be there" | Hard (noplay XLSX) |
| "They'd prefer not to play" | Soft (PREFERENCE_NO_PLAY) |
| "It's a state championship, most players away" | Soft (PREFERENCE_NO_PLAY) |
| "The venue is closed" | Hard (FIELD_UNAVAILABILITIES) |

---

## Venue/Field Restrictions

### Block Entire Venue for a Date

Add to `FIELD_UNAVAILABILITIES`:

```python
'Newcastle International Hockey Centre': {
    'weekends': [datetime(2026, 5, 16)],  # Blocks Fri-Sun
    'whole_days': [datetime(2026, 4, 25)],  # Blocks specific day
}
```

### Block Specific Field

Currently not directly supported. Workaround:
1. Remove field from `FIELDS` list for that season, OR
2. Add a custom constraint

### Block Specific Timeslot

Use `part_days` (if implemented) or remove from `DAY_TIME_MAP`.

---

## PHL/2nd Grade Time Restrictions

### Restrict PHL to Specific Times

Edit `PHL_GAME_TIMES` in config:

```python
PHL_GAME_TIMES = {
    'Newcastle International Hockey Centre': {
        'EF': {
            'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30)]  # Only these times
        },
    },
}
```

**Effect:** PHL can ONLY play at listed times. Other timeslots have no PHL variables.

### Restrict 2nd Grade to Specific Times

Edit `SECOND_GRADE_TIMES` similarly.

**Important:** Times must exist in `DAY_TIME_MAP`. You cannot create new timeslots.

---

## Club Days

When a club wants all their teams to play back-to-back on the same field:

```python
CLUB_DAYS = {
    'Crusaders': datetime(2026, 6, 14),
}
```

**Enforces:**
- All club teams play on that date
- All games on same field
- Games in contiguous timeslots
- Intra-club matchups where possible

---

## Friday Night Games

### Gosford Friday Nights

1. Set count in `FRIDAY_NIGHT_CONFIG['gosford_friday_count']`
2. Add times to `PHL_GAME_TIMES['Central Coast Hockey Park']`
3. Constraint `PHLAndSecondGradeTimes` enforces exact count

### NIHC Friday Nights

1. Set times in `PHL_GAME_TIMES['Newcastle International Hockey Centre']['EF']['Friday']`
2. Configure limit in `FRIDAY_NIGHT_CONFIG`
3. Constraint limits to 3 Friday games at NIHC

---

## Team Conflicts

When two teams cannot play at the same time (shared players):

```python
# In config or directly in data
TEAM_CONFLICTS = [
    ('Tigers PHL', 'Tigers 2nd'),  # Can't play simultaneously
    ('Souths PHL', 'Souths 2nd'),
]
```

**Effect:** `TeamConflictConstraint` prevents these teams from having games at the same timeslot.

---

## Home/Away Balance

### Maitland/Gosford 50/50 Split

Automatic via `FiftyFiftyHomeandAway` constraint. No configuration needed.

### Maitland Home Grouping

Automatic via `MaitlandHomeGrouping`:
- All Maitland teams home OR all away each week
- No back-to-back home weekends

---

## Grade Adjacency

### Prevent Adjacent Grades Playing Simultaneously

Automatic via `ClubGradeAdjacencyConstraint`:
- PHL and 2nd from same club cannot play at same timeslot
- 3rd and 4th from same club cannot play at same timeslot
- etc.

---

## Club Game Spread

### Minimize Gaps Between a Club's Games

Automatic via `ClubGameSpreadAI`:

**How gaps are calculated:**
1. For each (club, week, day), find all day_slots where the club has a game
2. Two teams at the **same** day_slot counts as **one** timeslot used
3. `gaps = (max_slot - min_slot + 1) - timeslots_used`

**Hard limit:** gaps ≤ 2 (+ slack if configured)
**Soft penalty:** weight 5000 per gap slot

| Scenario | Slots Used | Range | Gaps |
|----------|-----------|-------|------|
| Slots 1,2,3 | 3 | 3 | 0 |
| Slots 1,3 | 2 | 3 | 1 |
| Slots 1,4 | 2 | 4 | 2 (at limit) |
| Slots 1,6 | 2 | 6 | 4 (infeasible) |

**Slack:** `--slack N` raises hard limit to 2+N via `constraint_slack['ClubGameSpread']`

**Severity:** Level 4 (LOW) — first to be relaxed by `--relax`

---

## Constraint Severity Levels

| Level | Name | Example Constraints | Relaxation |
|-------|------|---------------------|------------|
| 1 | CRITICAL | NoDoubleBooking, EqualGames, PHLAdjacency | Never |
| 2 | HIGH | ClubDay, TeamConflict, Maitland grouping | Last resort |
| 3 | MEDIUM | MatchupSpacing, GradeAdjacency | If needed |
| 4 | LOW | Timeslot choices, PreferredTimes, ClubGameSpread | First to relax |

Use `--relax` flag to automatically relax constraints if solver returns INFEASIBLE.

---

## Runtime Constraint Relaxation (--slack)

For specific constraints where hardcoded limits may be too restrictive, use the `--slack N` CLI flag:

```powershell
.\.venv\Scripts\python.exe run.py generate --year 2026 --slack 1
```

### Supported Constraints

| Constraint | Base Limit | Effect of --slack N |
|------------|------------|---------------------|
| `EqualMatchUpSpacingConstraint` | ±1 round from ideal | Allow ±(1+N) rounds |
| `AwayAtMaitlandGrouping` | Max 3 away clubs per Maitland weekend | Max 3+N away clubs |
| `MaitlandHomeGrouping` | No back-to-back home weekends | Allow N back-to-back pairs |
| `ClubGameSpread` | Max 2 gap slots per club/day | Max 2+N gap slots |

### How It Works

1. CLI parses `--slack N` and builds a `constraint_slack` dict
2. The dict is passed through `build_season_data()` → `data['constraint_slack']`
3. Each constraint reads from `data.get('constraint_slack', {})`
4. Constraints apply: `actual_limit = base_limit + constraint_slack.get('ConstraintName', 0)`

### Implementation Location

- **CLI parsing:** `run.py` lines 99-101, 256-267
- **Data passing:** `utils.py` → `build_season_data()` adds `'constraint_slack'` to data dict
- **Constraint code:** Both `constraints/original.py` and `constraints/ai.py` read from `data.get('constraint_slack', {})`

### When to Use

| Situation | Recommended Flag |
|-----------|------------------|
| Solver INFEASIBLE, need quick fix | `--slack 1` |
| Still INFEASIBLE after --slack 1 | `--slack 2` then investigate config |
| One-off relaxation for specific season | `--slack 1` acceptable |
| Permanent relaxation | Update base limit in constraint code |

---

## Adding Custom Constraints

For constraints not covered by config:

1. Check if existing constraint can be configured differently
2. If not, add to `constraints/original.py` (or `constraints/ai.py` for AI version)
3. Add to constraint list in `main_staged.py`
4. Add tests in `tests/test_constraints.py`
5. Document in `docs/ai/CONSTRAINT_APPLICATION.md` (this file)
