# AI Operations Manual for Hockey Draw Scheduler

> **DOCUMENT CLASS:** AI Operator Documentation  
> **Audience:** AI assistants (Copilot, Claude, etc.)  
> **Purpose:** Complete technical reference for AI to operate this system  
> **Last Updated:** 2026-03-14

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Data Flow](#data-flow)
3. [Core Workflows](#core-workflows)
4. [Configuration Reference](#configuration-reference)
5. [Constraint System](#constraint-system)
6. [Key Limitations & Gaps](#key-limitations--gaps)
7. [Quick Reference](#quick-reference)

---

## System Architecture

### High-Level Overview

```
┌─────────────┐     ┌──────────────┐     ┌────────────────┐     ┌─────────────┐
│ Season      │────▶│ Data Loader  │────▶│ Variable Gen   │────▶│ OR-Tools    │
│ Config      │     │ (config/)    │     │ (utils.py)     │     │ CP-SAT      │
└─────────────┘     └──────────────┘     └────────────────┘     │ Solver      │
                                                                 └──────┬──────┘
                                                                        │
                    ┌──────────────┐     ┌────────────────┐            │
                    │ Excel/JSON   │◀────│ Roster Convert │◀───────────┘
                    │ Output       │     │ (utils.py)     │
                    └──────────────┘     └────────────────┘
```

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Entry Point** | `run.py` | CLI interface for all commands |
| **Staged Solver** | `main_staged.py` | Orchestrates staged/simple solving |
| **Data Loading** | `config/__init__.py` | Loads season configuration |
| **Variable Generation** | `utils.py → generate_X()` | Creates decision variables |
| **Constraints** | `constraints/` | Hard and soft constraint implementations |
| **Analytics/Testing** | `analytics/` | Draw storage, testing, violation checks |
| **Season Configs** | `config/season_{year}.py` | Year-specific configuration |

### File Structure

```
project/
├── run.py                  # CLI entry point
├── main_staged.py          # Solver orchestration
├── utils.py                # Core utilities, variable generation
├── models.py               # Data models (Team, Club, Grade, etc.)
├── config/
│   ├── __init__.py         # Config loading
│   ├── season_2026.py      # Season-specific config
│   └── team_naming.py      # Team name helpers
├── constraints/
│   ├── original.py         # Original human-written constraints
│   ├── ai.py               # AI-enhanced constraints
│   ├── soft.py             # Soft constraint versions
│   ├── severity.py         # Severity-based grouping
│   └── symmetry.py         # Symmetry breaking
├── analytics/
│   ├── storage.py          # DrawStorage, DrawAnalytics, load/save draws
│   ├── tester.py           # DrawTester, violation checks
│   ├── versioning.py       # DrawVersionManager, auto-versioning
│   └── reports.py          # Club/grade reports
├── data/{year}/
│   ├── teams/*.csv         # Team nominations
│   └── noplay/*.xlsx       # Hard no-play rules
├── draws/{year}/           # Output draws (versioned)
│   ├── current.json        # ← ALWAYS the latest draw
│   ├── current.xlsx        # ← Latest schedule Excel
│   ├── current_analytics.xlsx  # ← Latest analytics
│   ├── CHANGELOG.md        # Version history
│   └── versions/           # All versioned draws
│       ├── draw_v1.0.json
│       └── draw_v1.0.xlsx
├── checkpoints/
│   ├── latest/             # ← Latest successful checkpoint
│   └── run_N/              # Per-run checkpoints
└── logs/                   # Solver logs
```

---

## Data Flow

### 1. Configuration Loading

```python
# config/season_2026.py defines SEASON_CONFIG dict
SEASON_CONFIG = {
    'year': 2026,
    'start_date': datetime(2026, 3, 22),
    'fields': FIELDS,
    'phl_game_times': PHL_GAME_TIMES,
    ...
}

# config/__init__.py loads and builds complete data
data = load_season_data(2026)
# Returns: teams, grades, clubs, fields, timeslots, num_rounds, etc.
```

### 2. Variable Generation (Critical!)

`utils.py → generate_X()` creates decision variables. This is where **filtering happens**:

| Grade | Filtering Applied |
|-------|-------------------|
| PHL | `PHL_GAME_TIMES` dict - only listed slots get variables |
| 2nd | `SECOND_GRADE_TIMES` dict - only listed slots get variables |
| 3rd-6th | Excluded from Gosford, excluded from Fridays |

**Variable Format:**
```python
key = (
    team1, team2, grade,    # WHO plays
    day, day_slot, time,    # WHEN
    week, date, round_no,   # WHICH round
    field_name, field_loc   # WHERE
)
X[key] = BoolVar           # 1 = game scheduled, 0 = not
```

### 3. Constraint Application

Constraints are added in `main_staged.py`:

```python
# Each constraint class implements apply(model, X, data)
for constraint_class in stage_config['constraints']:
    constraint = constraint_class()
    constraint.apply(model, X, data)
```

### 4. Solving

```python
solver = cp_model.CpSolver()
status = solver.Solve(model)
# status: OPTIMAL, FEASIBLE, INFEASIBLE, etc.
```

### 5. Output (Unified Versioning)

All solver modes (staged, simple) automatically save through the unified versioning system:

```python
# Both staged and simple modes call this internally:
from analytics.versioning import DrawVersionManager
manager = DrawVersionManager('draws', year=year)
version = manager.save_solver_output(solution, data, description, mode)
```

**Output produced:**
- `draws/{year}/current.json` — Latest draw (always up-to-date)
- `draws/{year}/current.xlsx` — Latest schedule Excel 
- `draws/{year}/current_analytics.xlsx` — Latest analytics
- `draws/{year}/versions/draw_v{X}.{Y}.json` — Versioned draw
- `draws/{year}/versions/draw_v{X}.{Y}.xlsx` — Versioned schedule
- `draws/{year}/CHANGELOG.md` — Auto-updated version history
- `checkpoints/latest/` — Latest solver checkpoint

**For AI: Always read `draws/{year}/current.json` to find the latest draw.**

---

## Core Workflows

### Workflow 1: Generate a New Draw

```powershell
# ALWAYS use isBackground: true - solver runs for hours!
.\.venv\Scripts\python.exe run.py generate --year 2026
```

**Process:**
1. Load data from `config/season_2026.py`
2. Generate ~30,000+ decision variables
3. Apply constraints in stages
4. Solve (may take 2-72+ hours)
5. Export to `draws/` folder

### Workflow 2: Lock Partial Draw and Re-Solve

```powershell
# Lock weeks 1-5, re-solve weeks 6+
.\.venv\Scripts\python.exe run.py generate --year 2026 \
    --locked draws/2026/draw_v1.json --lock-weeks 5
```

**What This Does:**
1. Loads draw from JSON
2. Extracts game keys for weeks 1-5
3. Adds `model.Add(X[key] == 1)` for each locked game
4. Solves with locked games as hard constraints

### Workflow 3: Test Draw for Violations (Post-Hoc)

```powershell
.\.venv\Scripts\python.exe run.py test draws/draw.json --year 2026
```

**What This Does:**
1. Loads draw from JSON
2. Runs `DrawTester.run_violation_check()`
3. Reports violations by severity level

**⚠️ LIMITATION:** This is a POST-HOC statistical check. It does NOT verify solver feasibility.

### Workflow 4: Swap Games and Test

```powershell
.\.venv\Scripts\python.exe run.py swap draws/draw.json G00001 G00002 --year 2026 --save draws/modified.json
```

**What This Does:**
1. Swaps the timeslots of two games
2. Runs violation check
3. Reports which violations were introduced/fixed
4. Optionally saves modified draw

### Workflow 5: Verify Draw Feasibility (Solver-Based)

```powershell
# Verify entire draw is solver-feasible
.\.venv\Scripts\python.exe run.py verify draws/draw.json --year 2026

# Verify only specific weeks (useful for partial draws)
.\.venv\Scripts\python.exe run.py verify draws/draw.json --year 2026 --lock-weeks 5

# Increase timeout for complex draws
.\.venv\Scripts\python.exe run.py verify draws/draw.json --year 2026 --timeout 60
```

**What This Does:**
1. Loads draw from JSON
2. Locks ALL games (or up to `--lock-weeks`)
3. Builds a fresh model with ALL constraints
4. Runs solver's constraint propagation
5. Reports FEASIBLE/INFEASIBLE/UNKNOWN

**Key Difference from `test`:**
- `test` = post-hoc statistical check (fast but incomplete)
- `verify` = solver constraint propagation (slower but definitive)

---

## Configuration Reference

### SEASON_CONFIG (Required)

```python
SEASON_CONFIG = {
    'year': 2026,                             # Season year
    'start_date': datetime(2026, 3, 22),      # First playing Sunday
    'last_round_date': datetime(2026, 8, 30), # Last regular round
    'end_date': datetime(2026, 9, 19),        # Grand Final
    'max_rounds': 22,                         # Max weekends (default)
    'teams_data_path': 'data/2026/teams',     # Team CSV location
    'noplay_data_path': 'data/2026/noplay',   # Hard no-play XLSX
    
    # References to other config dicts
    'fields': FIELDS,
    'day_time_map': DAY_TIME_MAP,
    'phl_game_times': PHL_GAME_TIMES,
    'second_grade_times': SECOND_GRADE_TIMES,
    'field_unavailabilities': FIELD_UNAVAILABILITIES,
    'club_days': CLUB_DAYS,
    'preference_no_play': PREFERENCE_NO_PLAY,
    'friday_night_config': FRIDAY_NIGHT_CONFIG,
    'max_weekends_per_grade': MAX_WEEKENDS_PER_GRADE,
}
```

### MAX_WEEKENDS_PER_GRADE

Sets hard ceiling for available weekends per grade:

```python
MAX_WEEKENDS_PER_GRADE = {
    'PHL': 22,   # 20 Sundays + 2 rescued via Friday
    '2nd': 20,   # 20 Sundays only
    '3rd': 20,
    '4th': 20,
    '5th': 20,
    '6th': 20,
}
```

**⚠️ IMPORTANT:** Friday nights are PART OF a weekend, NOT additional weekends!

### PHL_GAME_TIMES (Variable Filtering)

```python
PHL_GAME_TIMES = {
    'Newcastle International Hockey Centre': {
        'EF': {  # East Field only (SF excluded for PHL)
            'Friday': [tm(19, 0)],           # 7pm
            'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
        },
        'WF': { ... },  # Same as EF
        # SF (South Field) NOT LISTED = excluded
    },
    'Central Coast Hockey Park': {
        'Wyong Main Field': {
            'Friday': [tm(20, 0)],           # 8pm
            'Sunday': [tm(12, 0), tm(13, 30)]
        },
    },
}
```

### SECOND_GRADE_TIMES (Variable Filtering)

```python
SECOND_GRADE_TIMES = {
    'Newcastle International Hockey Centre': {
        'EF': {
            'Sunday': [tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30)]
        },
        # Gosford NOT LISTED = 2nd grade excluded
    },
}
```

### PREFERENCE_NO_PLAY (Soft Constraints)

```python
PREFERENCE_NO_PLAY = {
    'Crusaders_6th_Masters': {
        'club': 'Crusaders',
        'grade': '6th',                      # Optional
        'dates': [datetime(2026, 4, 17)],
        'reason': 'Masters State Champs',
    },
}
```

### FIELD_UNAVAILABILITIES (Hard Blocks)

```python
FIELD_UNAVAILABILITIES = {
    'Newcastle International Hockey Centre': {
        'weekends': [datetime(2026, 4, 4)],  # Easter
        'whole_days': [datetime(2026, 4, 25)], # ANZAC
    },
}
```

---

## Constraint System

### Severity Levels

| Level | Name | Examples | Can Relax? |
|-------|------|----------|------------|
| 1 | CRITICAL | Double-booking, equal games | ❌ Never |
| 2 | HIGH | Club days, team conflicts | ⚠️ With `--slack` |
| 3 | MEDIUM | Matchup spacing, adjacency | ⚠️ With `--slack` |
| 4 | LOW | Timeslot preferences | ✅ Yes |

### Constraint Types

**Hard (Variable Removal):**
- PHL excluded from SF via `PHL_GAME_TIMES` filtering
- Lower grades excluded from Gosford via `generate_X()` filtering

**Hard (Model Constraints):**
- `NoDoubleBookingTeams` - `model.Add(sum(game_vars) <= 1)`
- `EnsureEqualGamesAndBalanceMatchUps` - exact game counts

**Soft (Penalties):**
- `PreferredTimesConstraint` - penalty for no-play dates
- `EnsureBestTimeslotChoices` - prefer certain slots

### Constraint Stages

**Default Mode (`stage1_required` → `stage2_soft`):**

| Stage | Constraints |
|-------|-------------|
| stage1_required | All core + structural constraints |
| stage2_soft | Optimization preferences |

**Severity Mode (`--staged`):**

| Stage | Level | Constraints |
|-------|-------|-------------|
| severity_1 | 1 | Double-booking, equal games, adjacency |
| severity_2 | 2 | Club days, Maitland grouping |
| severity_3 | 3 | Spacing, club alignment |
| severity_4 | 4 | Timeslot optimization |

---

## Key Limitations & Gaps

### 1. `test` vs `verify` Commands

| Command | Method | Speed | What It Catches |
|---------|--------|-------|-----------------|
| `test` | Statistical checks | Fast (~1s) | Known violation patterns |
| `verify` | Solver propagation | Slow (~30s) | ALL constraint violations |

**Use `test` for:** Quick sanity checks during editing.
**Use `verify` for:** Definitive feasibility confirmation before committing.

### 2. Hints Are Not Locks

**Problem:** When resuming from a checkpoint or using a prior solution, it's loaded as a **HINT**, not a lock.

```python
# HINT: "try to use this value"
model.AddHint(X[key], 1)

# LOCK: "must use this value"
model.Add(X[key] == 1)
```

The solver CAN deviate from hints if constraints require it.

### 3. No Incremental Constraint Addition

**Problem:** OR-Tools CP-SAT doesn't support adding constraints to an already-solved model. Between stages, the model is rebuilt fresh.

### 4. No Partial Game Locking

**Problem:** Locking is week-based only (`--lock-weeks N`). You can't lock arbitrary individual games.

### 5. No "What-If" Scenario Testing

**Problem:** No command to quickly test "what if I move game X to slot Y without running full solver?"

Current workaround:
1. Use `DrawTester.move_game()` in Python
2. Run `tester.run_violation_check()`
3. This only catches known violations, not solver infeasibility

---

## Known Bugs / Data Integrity Issues

### 1. day_slot Indexing Inconsistency (FOUND 2026-03-14)

**Symptom:** Locked games from draw files don't match model keys.

**Details:**
- `generate_timeslots()` uses 1-indexed `day_slot` (slot 1 = first time)
- Some draw export paths may use different indexing
- Result: `(t1, t2, grade, day, 3, '13:00', ...)` in draw doesn't match `(t1, t2, grade, day, 4, '13:00', ...)` in model

**Impact:** `--locked` flag may not lock games if keys don't match.

**Verification:** Run `python scripts/verify_locked_keys.py` to check key matching.

**Root Cause:** Needs investigation - check draw creation code vs timeslot generation.

---

## Quick Reference

### Commands

```powershell
# Generate (ALWAYS background, ALWAYS --year)
.\.venv\Scripts\python.exe run.py generate --year 2026

# With slack relaxation
.\.venv\Scripts\python.exe run.py generate --year 2026 --relax

# Test violations - use "current" to test latest draw
.\.venv\Scripts\python.exe run.py test current --year 2026

# Test a specific version
.\.venv\Scripts\python.exe run.py test v2.0 --year 2026

# Test with direct path
.\.venv\Scripts\python.exe run.py test draws/2026/versions/draw_v1.0.json --year 2026

# Pre-season report
.\.venv\Scripts\python.exe run.py preseason --year 2026

# Swap games (auto-saves as minor version)
.\.venv\Scripts\python.exe run.py swap current G001 G002 --year 2026 --save

# Lock partial draw and regenerate remainder
.\.venv\Scripts\python.exe run.py generate --year 2026 \
    --locked draws/2026/current.json --lock-weeks 5

# Migrate legacy flat draws to versioned structure
.\.venv\Scripts\python.exe run.py migrate --year 2026
```

### File Locations (AI Quick Lookup)

| What | Where | Notes |
|------|-------|-------|
| **Latest draw** | `draws/{year}/current.json` | **Always check here first** |
| **Latest schedule** | `draws/{year}/current.xlsx` | Auto-updated |
| **Latest analytics** | `draws/{year}/current_analytics.xlsx` | Auto-updated |
| **Version history** | `draws/{year}/CHANGELOG.md` | Auto-generated |
| **All versions** | `draws/{year}/versions/` | Versioned draws + Excel |
| **Latest checkpoint** | `checkpoints/latest/` | Latest solver state |
| Season config | `config/season_{year}.py` | Year-specific config |
| Team data | `data/{year}/teams/*.csv` | Club team nominations |
| Hard no-play | `data/{year}/noplay/*.xlsx` | No-play Excel rules |
| Solver logs | `logs/solver_*.log` | Solve progress logs |
| Run checkpoints | `checkpoints/run_N/` | Per-run checkpoints |

### Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `load_season_data()` | `config/__init__.py` | Load complete data dict |
| `generate_X()` | `utils.py` | Create decision variables |
| `convert_X_to_roster()` | `utils.py` | Solution → Roster |
| `DrawStorage.load_and_lock()` | `analytics/storage.py` | Load + get locked keys |
| `DrawTester.run_violation_check()` | `analytics/tester.py` | Post-hoc checking |
| `DrawVersionManager.save_solver_output()` | `analytics/versioning.py` | **Unified output** - saves JSON, Excel, analytics, versions, current |
| `resolve_draw_path()` | `run.py` | Resolve "current", "v2.0", or paths to actual file |
| `DrawVersionManager.migrate_legacy_draws()` | `analytics/versioning.py` | Move flat draws into `versions/` |

---

## Document History

This consolidated manual replaces 6 previously fragmented documents (deleted March 2026):
- README.md, SEASON_SETUP.md, CONFIGURATION_REFERENCE.md
- CONSTRAINT_APPLICATION.md, GAME_TIME_DICTIONARIES.md, SYSTEM_OPERATION.md
