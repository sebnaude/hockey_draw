# Copilot Instructions for Hockey Draw Scheduler

## ⚠️ IMPERATIVE: Read Before Acting

**Before making ANY changes to configuration or constraints:**
1. Read the relevant AI documentation in `docs/ai/`
2. When you learn something new, **UPDATE** the relevant AI doc so you know next time
3. When you implement a change, **DOCUMENT HOW** in `seasons/{year}/2026_club_requests.md`

---

## AI Documentation Index

> **UPDATE MANDATE:** When you make changes to any of these areas, you MUST update the corresponding document. This ensures future AI sessions have accurate information.

| Document | Purpose | Contains | Update When |
|----------|---------|----------|-------------|
| `docs/ai/README.md` | Documentation index & protocols | Links to all docs, update procedures | Adding new docs |
| `docs/ai/SEASON_SETUP.md` | Pre-season configuration checklist | Step-by-step setup, data gathering requirements | Changing setup process |
| `docs/ai/CONFIGURATION_REFERENCE.md` | All config parameters | Every config key, valid values, effects | Adding/changing config options |
| `docs/ai/CONSTRAINT_APPLICATION.md` | How to apply restrictions | Soft vs hard constraints, examples | Adding new constraint types |
| `docs/ai/GAME_TIME_DICTIONARIES.md` | PHL/2nd grade variable filtering | How PHL_GAME_TIMES/SECOND_GRADE_TIMES work | Modifying game time logic |
| `docs/ai/SYSTEM_OPERATION.md` | Running the solver | Commands, flags, interpreting output | Changing CLI or solver behavior |
| `seasons/{year}/*.md` | Season-specific reports | Club requests, implementation tracking | Any season-specific change |

### Season Reports (Track Implementation Details)

| Document | Purpose | Update When |
|----------|---------|-------------|
| `seasons/{year}/2026_club_requests.md` | **Master tracking of all requests + HOW they were implemented** | Any config/constraint change |
| `seasons/{year}/2026_club_requests_summary.md` | Club-facing summary (no implementation details) | Sharing with clubs |

---

## Quick Reference: File Locations

| Topic | Location |
|-------|----------|
| **Season config** | `config/season_{year}.py` |
| **AI Documentation** | `docs/ai/` |
| **System Documentation** | `docs/system/` |
| **Season Reports** | `seasons/{year}/` |
| **Draw Rules** | `seasons/RULES.md` |
| **Constraint Code** | `constraints/` |
| **Team Data** | `data/{year}/teams/*.csv` |

---

## Critical Rules

### 1. Solver Execution
- **NEVER** use `isBackground: false` - solver runs for HOURS
- **ALWAYS** use `isBackground: true` for generate commands
- **ALWAYS** specify `--year` flag

### 2. File Locations
```
config/season_{year}.py    # Season configuration
constraints/               # All constraint modules
  ├── original.py          # Original human-written constraints
  ├── ai.py                # AI-enhanced constraints
  ├── soft.py              # Soft/relaxed constraint versions
  ├── severity.py          # Severity-based relaxation system
  └── resolver.py          # Infeasibility resolver
docs/ai/                   # AI assistant documentation
docs/system/               # Human operator documentation
seasons/{year}/            # Season-specific reports
data/{year}/teams/*.csv    # Team data
draws/{year}/              # Output schedules (versioned)
checkpoints/run_X/         # Solver checkpoints
logs/                      # Solver logs
scripts/                   # Utility scripts
```

### 3. OR-Tools API
- Use `solver.status_name(status)` NOT `solver.StatusName()` (deprecated)

---

## ⚠️ CRITICAL: Rounds Concepts (Do NOT Confuse!)

### Max Available Weekends vs Played Rounds
These are **TWO DIFFERENT CONCEPTS**:

| Concept | Config Key | Description |
|---------|-----------|-------------|
| **Max Available Weekends** | `MAX_WEEKENDS_PER_GRADE` | Hard ceiling - max possible games |
| **Played Rounds** | Calculated | Actual games per team (from formula) |

### Friday Nights Are NOT Extra Weekends!
- A Friday night game is **PART OF that weekend**, not additional
- If PHL plays Friday at Gosford, they don't also play Sunday
- Friday "rescues" blocked weekends (e.g., State Championships)
- PHL: 22 available = 20 Sundays + 2 rescued (NOT 20 + 8!)

**Full details:** See `docs/ai/SEASON_SETUP.md` → "Understanding Rounds"

### Games Calculation Formula
The system uses a unified formula for all grades:
```
max_matches = W × floor(T/2)   # Total matches possible
g0 = floor(2 × max_matches / T) # Games per team
g0 = min(g0, W)                 # Cap at available weekends
if T is odd: force g0 even      # Ensure valid scheduling
```
- Matchups distributed as `base` and `base+1` per pair
- Works correctly for any team count (even or odd)

---

## Variable Filtering (Performance Critical)

Two dicts control which **decision variables** are created:

| Dict | Controls | Key Rules |
|------|----------|-----------|
| `PHL_GAME_TIMES` | PHL variables | EF/WF only (no SF), restricted times |
| `SECOND_GRADE_TIMES` | 2nd grade variables | EF/WF only, Gosford not listed, PHL times ± 1 slot |

- Filtering in `utils.py` → `generate_X()`
- **Cannot create NEW timeslots** - only existing `DAY_TIME_MAP` slots
- Format: `{ venue: { field: { day: [times] } } }` (2026+)

**Full details:** See `docs/ai/GAME_TIME_DICTIONARIES.md`

### PHL-Only Restrictions (Lower Grades Excluded)

**Lower grades (3rd-6th) are automatically excluded from:**
- **Gosford (Central Coast Hockey Park)** - PHL-only venue, only Gosford PHL plays there
- **Friday nights** - PHL-only timeslot, all other grades play Sunday only

This exclusion is hardcoded in `generate_X()` and saves ~20,000+ variables.

### Friday Night Game Limits

| Venue | Limit | Type | Source |
|-------|-------|------|--------|
| Broadmeadow (NIHC) | ≤ 3 | Maximum | Operational constraint |
| Gosford (CCHP) | = 8 | Exact | AGM decision 2026 |

- **Constraint:** `PHLAndSecondGradeTimes` (both original and AI version)
- **Config:** `FRIDAY_NIGHT_CONFIG` in `config/season_{year}.py`
- **Testing:** `PHLTimingValidator` verifies both limits

### NIHC Friday Matchup Filtering (2026+)

Specific matchups are locked to specific Friday nights at NIHC via `nihc_friday_games`:

```python
FRIDAY_NIGHT_CONFIG = {
    'nihc_friday_games': {
        '2026-05-08': [('Maitland', 'Souths')],  # Only this matchup allowed
        '2026-06-19': [('Tigers', 'Wests')],     # Only this matchup allowed
        '2026-07-24': 'norths_only',              # Any matchup with Norths
    },
}
```

- Date NOT in dict = NO games allowed at NIHC that Friday
- `'norths_only'` = any matchup where one team is Norths
- Club names in tuples must be alphabetically sorted
- Filtering happens in `generate_X()` - no variables created for non-matching games

---

## Key Config Rules

### PHL_PREFERENCES
**Only** supports `preferred_dates` key. Other keys cause constraint errors.

### Constraint Groups
- **Required constraints**: Must be satisfied (feasibility)
- **Soft constraints**: Preferences with penalties (optimization)
- **Severity groups**: Used by `--relax` flag for infeasibility resolution (see below)

---

## Quick Commands

```powershell
# Generate (ALWAYS background, ALWAYS --year)
.\.venv\Scripts\python.exe run.py generate --year 2026

# Generate with automatic constraint relaxation (if infeasible)
.\.venv\Scripts\python.exe run.py generate --year 2026 --relax

# Generate with relaxed constraint limits (+N to specific limits)
.\.venv\Scripts\python.exe run.py generate --year 2026 --slack 1

# Pre-season report
.\.venv\Scripts\python.exe run.py preseason --year 2026

# Test a draw
.\.venv\Scripts\python.exe run.py test draws/draw.json --year 2026

# Diagnose infeasibility
.\.venv\Scripts\python.exe run.py diagnose --year 2026 --timeout 60
```

---

## --slack Flag (Constraint Limit Relaxation)

When solver returns INFEASIBLE and `--relax` doesn't help, try `--slack N` to loosen specific constraints:

```powershell
.\.venv\Scripts\python.exe run.py generate --year 2026 --slack 1
```

**Affected Constraints:**

| Constraint | Base | +1 Slack | +2 Slack |
|------------|------|----------|----------|
| `EqualMatchUpSpacing` | ±1 round | ±2 rounds | ±3 rounds |
| `AwayAtMaitlandGrouping` | Max 3 | Max 4 | Max 5 |
| `MaitlandHomeGrouping` | No back-to-back | 1 allowed | 2 allowed |

---

## --relax Flag (Infeasibility Resolution)

When solver returns INFEASIBLE, use `--relax` to automatically find and relax the blocking constraint group:

```powershell
.\.venv\Scripts\python.exe run.py generate --year 2026 --relax
```

**How it works:**
1. Tests with all constraints
2. If INFEASIBLE, drops severity level 4 constraints and retests
3. If still INFEASIBLE, drops level 3, then level 2
4. Identifies the blocking severity group
5. Relaxes ALL constraints in that group (slack +1)
6. Solves with ALL constraints together (never locks partial solutions)

**Severity Levels:**
- Level 1: CRITICAL (never relaxed) - double-booking, equal games, PHL adjacency
- Level 2: HIGH (structural) - club days, Maitland grouping, team conflicts
- Level 3: MEDIUM (spacing) - matchup spacing, grade adjacency, club vs club
- Level 4: LOW (optimization) - timeslot choices, club density

See `severity_relaxation.py` for implementation details.

---

## Adding a New Season

1. Copy `config/season_template.py` → `config/season_{year}.py`
2. Update year references (search `9999`)
3. Create `data/{year}/teams/` with club CSV files
4. Update dates, unavailabilities, club days
5. System auto-detects new config

---

## Data Formats

### Team CSV (`data/{year}/teams/{club}.csv`)
```csv
Club,Grade,Team Name
Maitland,PHL,Maitland PHL
Colts,5th,Colts Gold
```

### Team Naming (multiple teams same grade)
| Club | Team 1 | Team 2 |
|------|--------|--------|
| Tigers | Tigers | Tigers Black |
| Wests | Wests | Wests Red |
| Colts | Colts Gold | Colts Green |

See `config/team_naming.py` for helper functions.
