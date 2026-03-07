# Copilot Instructions for Hockey Draw Scheduler

## Quick Reference: Where to Find Information

| Topic | Document |
|-------|----------|
| **How to run solver** | `.github/copilot-skills/hockey-draw-scheduler.md` |
| **Pre-season setup protocol** | `AI_DRAW_INIT.md` |
| **Constraint documentation** | `AI_CONSTRAINTS_AUDIT.md` |
| **System architecture** | `docs/SYSTEM_OVERVIEW.md` |
| **Draw rules & business logic** | `docs/DRAW_RULES.md` |
| **Season configuration** | `config/season_{year}.py` |

---

## Critical Rules

### 1. Solver Execution
- **NEVER** use `isBackground: false` - solver runs for HOURS
- **ALWAYS** use `isBackground: true` for generate commands
- **ALWAYS** specify `--year` flag

### 2. File Locations
```
config/season_{year}.py    # Season configuration
data/{year}/teams/*.csv    # Team data
draws/                     # Output schedules
checkpoints/run_X/         # Solver checkpoints
logs/                      # Solver logs
reports/                   # Pre-season reports
```

### 3. OR-Tools API
- Use `solver.status_name(status)` NOT `solver.StatusName()` (deprecated)

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

# Pre-season report
.\.venv\Scripts\python.exe run.py preseason --year 2026

# Test a draw
.\.venv\Scripts\python.exe run.py test draws/draw.json --year 2026

# Diagnose infeasibility
.\.venv\Scripts\python.exe run.py diagnose --year 2026 --timeout 60
```

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
