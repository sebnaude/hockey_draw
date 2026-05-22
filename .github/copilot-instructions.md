# Copilot Instructions for Hockey Draw Scheduler

## 🚨 MANDATORY: Read AI Docs On EVERY Conversation Start

**At the start of EVERY conversation — before answering ANY question or making ANY change — you MUST read:**

1. `docs/ai/AI_OPERATIONS_MANUAL.md` — The consolidated operations manual covering architecture, data flow, configuration, constraints, workflows, and limitations
2. `config/season_2026.py` (or the relevant season) — The ACTUAL season config including `FIELD_UNAVAILABILITIES`, `FORCED_GAMES`, `BLOCKED_GAMES`, `PHL_GAME_TIMES`, `SECOND_GRADE_TIMES`, and all other config dicts

**This is NON-NEGOTIABLE.** The system has multiple layers of variable filtering and game restriction (time dicts, field unavailabilities, forced games, blocked games, no-play XLSX). You CANNOT answer questions about what is or isn't configured without reading the actual config file. Do NOT guess or assume — READ THE FILES.

**Additional reference docs (read when relevant):**

| Document | Purpose | Read When |
|----------|---------|-----------|
| `docs/ai/README.md` | Documentation index | First time |
| `docs/ai/SEASON_SETUP.md` | Pre-season configuration checklist | Starting new season |
| `docs/ai/CONFIGURATION_REFERENCE.md` | All config parameters | Changing any config |
| `docs/ai/CONSTRAINT_APPLICATION.md` | How to apply restrictions | Adding constraints |
| `docs/ai/GAME_TIME_DICTIONARIES.md` | PHL/2nd grade variable filtering | Modifying game times |
| `docs/ai/SYSTEM_OPERATION.md` | Running the solver | Generating draws |

**When you learn something new, UPDATE the relevant AI doc so you know next time.**

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

## Variable Filtering (Performance Critical)

Two dicts control which **decision variables** are created:

| Dict | Controls | Key Rules |
|------|----------|-----------|
| `PHL_GAME_TIMES` | PHL variables | EF/WF only (no SF), restricted times |
| `SECOND_GRADE_TIMES` | 2nd grade variables | EF/WF only, Gosford not listed, PHL times ± 1 slot |
Two lists control **variable removal** by team/date:

| Config | Purpose | Logic |
|--------|---------|-------|
| `FORCED_GAMES` | Force specific matchups at specific slots | Scope match + team NOT match → variable removed |
| `BLOCKED_GAMES` | Prevent teams from playing on specific dates | Scope match + team match → variable removed |
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
| Maitland Park | = 2 | Exact | Gosford vs Maitland only |

- **Constraint:** `PHLAndSecondGradeTimes` (both original and AI version)
- **Config:** `CONSTRAINT_DEFAULTS` in `config/season_{year}.py` (`gosford_friday_games`, `maitland_friday_games`, `max_friday_broadmeadow`) + `FORCED_GAMES` for date/matchup locking
- **Testing:** `PHLTimingValidator` verifies both limits

---

## Key Config Rules

### PREFERRED_GAMES (spec-020)
Soft, weighted analogue of `FORCED_GAMES` (same scope/team/club grammar + optional `weight`), penalty-on-deviation instead of a hard rule. Replaces the deleted `PHL_PREFERENCES` / `PreferredDates`. Marquee PHL dates are now a soft preferred entry, e.g. `{'grade':'PHL','date':X,'constraint':'equal','count':1,'weight':10000}`. See `docs/system/FORCED_GAMES_AS_COUNT_RULES.md`.

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
# Lock specific weeks from a prior draw and re-solve the rest
.\.\.venv\\Scripts\\python.exe run.py generate --year 2026 --locked draws/2026/draw_v1.0.json --lock-weeks 1
.\.\.venv\\Scripts\\python.exe run.py generate --year 2026 --locked draws/2026/draw_v1.0.json --lock-weeks 1,2,3
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
2. If INFEASIBLE, drops severity level 5 constraints and retests
3. If still INFEASIBLE, drops level 4, then level 3, then level 2
4. Identifies the blocking severity group
5. Relaxes ALL constraints in that group (slack +1)
6. Solves with ALL constraints together (never locks partial solutions)

**Severity Levels:**
- Level 1: CRITICAL (never relaxed) - double-booking, equal games, PHL adjacency
- Level 2: HIGH (structural) - club days, Maitland grouping, team conflicts, matchup spacing
- Level 3: MEDIUM (spacing) - grade adjacency, club vs club
- Level 4: LOW (optimization) - club density at Broadmeadow
- Level 5: VERY LOW (timeslot preferences) - timeslot choices, preferred times

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
