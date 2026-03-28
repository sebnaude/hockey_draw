# Hockey Draw Scheduler - Claude Code Instructions

## Mandatory First Steps

**Before answering ANY question or making ANY change, READ:**
1. `config/season_2026.py` (or relevant season) - the ACTUAL config with all dicts
2. This file in its entirety

The system has multiple layers of variable filtering and game restriction. Do NOT guess - READ THE CONFIG.

**Concurrent AI sessions:** Other AI agents may be editing files in this repo at the same time. If you notice a file has changed since you last read it (e.g. via system reminders about modifications), re-read the file before editing and respect those changes — do not revert them unless the user explicitly asks you to.

---

## Project Overview

Constraint programming system for generating hockey competition schedules using Google OR-Tools CP-SAT solver. Generates season draws for ~47 teams across 6 grades (PHL, 2nd-6th), 10 clubs, multiple venues.

**Entry point:** `run.py` (CLI) -> `main_staged.py` (solver orchestration)

## Critical Rules

### 1. `constraints/original.py` — Only Edit When Directed
The original human-written constraints are the **source of truth**. Do NOT edit `constraints/original.py` unless the user explicitly asks you to. By default, all fixes go to `constraints/ai.py` only.

- Default edit targets: `constraints/ai.py`, `tests/test_ai_constraints_comprehensive.py`
- Only edit when user directs: `constraints/original.py`, `tests/test_constraints.py`, `tests/test_constraints_equivalence.py`

### 2. Solver Execution
- **ALWAYS** run solver commands in background - runs for HOURS/DAYS
- **ALWAYS** use `--year` flag
- Use `solver.status_name(status)` NOT deprecated `solver.StatusName()`
- Stage timing is config-driven: `max_time_per_stage` in `SEASON_CONFIG` (default: 2 days per stage)

### 3. Variable Filtering (Performance Critical)

Two dicts control which decision variables are created:

| Dict | Controls | Key Rules |
|------|----------|-----------|
| `PHL_GAME_TIMES` | PHL variables | EF/WF only (no SF), restricted times |
| `SECOND_GRADE_TIMES` | 2nd grade variables | EF/WF only, no Gosford, PHL times +/- 1 slot |

Lower grades (3rd-6th) are auto-excluded from Gosford and Friday nights in `generate_X()`.

**Home-venue filter** (in `generate_X()`): Games at away venues (Maitland Park, Central Coast) must involve the home club. Controlled by `home_field_map` in config. Only Maitland teams play at Maitland Park; only Gosford teams play at Central Coast. NIHC is the default/neutral venue with no restriction. This eliminates ~21,000 variables.

Two lists control variable filtering:

| Config | Purpose |
|--------|---------|
| `FORCED_GAMES` | Force games matching partial keys (sum == 1 by default, supports `constraint` field for `lesse`/`greatere`/etc.) |
| `BLOCKED_GAMES` | Eliminate variables matching scope + team matchers (or ALL vars in scope if no teams specified) |

### 4. Friday Night Limits
- Broadmeadow (NIHC): max 3 games (`CONSTRAINT_DEFAULTS['max_friday_broadmeadow']`)
- Gosford (CCHP): exactly 8 games (`CONSTRAINT_DEFAULTS['gosford_friday_games']`, AGM decision 2026)
- Maitland Park: exactly 2 games (`CONSTRAINT_DEFAULTS['maitland_friday_games']`, Gosford vs Maitland only)

### 5. Weeks vs Rounds
The season spans 27 calendar weeks but has fewer playable rounds:
- **3 no-play weeks** (Easter, Masters, etc.) — no timeslots generated
- **2 Friday-only weeks** (PHL only, no Sunday games)
- **22 Sunday weeks** = playable rounds for non-PHL grades
- **24 total playable weeks** for PHL (22 Sundays + 2 Friday-only)

When checking "consecutive" anything (back-to-back home weeks, etc.), use the playable round sequence, not raw week numbers. No-play weeks don't count as gaps.

---

## File Structure

```
run.py                    # CLI entry point
main_staged.py            # Staged solver orchestration
models.py                 # Data models (Team, Club, Grade, Timeslot, etc.)
utils.py                  # Utilities + generate_X() variable generation
solver_diagnostics.py     # Logging and resource monitoring

config/
  __init__.py             # Config loader (load_season_data)
  season_2026.py          # Active season config
  season_template.py      # Template for new seasons
  team_naming.py          # Team name helpers

constraints/
  original.py             # Original constraints (READ-ONLY)
  ai.py                   # AI-enhanced constraints (edit this)
  soft.py                 # Soft constraint variants
  severity.py             # Severity-based relaxation + CONSTRAINT_TO_SEVERITY mapping
  resolver.py             # Infeasibility resolver
  symmetry.py             # Symmetry breaking

analytics/
  storage.py              # DrawStorage, DrawAnalytics
  tester.py               # DrawTester, violation checks (slack-aware)
  versioning.py           # DrawVersionManager
  reports.py              # Club/grade reports
  preseason_report.py     # Pre-season reports

data/{year}/teams/*.csv   # Team nominations per club
draws/{year}/             # Output draws (versioned)
  current.json            # Latest draw (always check here first)
  current.xlsx            # Latest schedule
  versions/               # All versioned draws (draw_v*.json + .xlsx)
checkpoints/              # Solver checkpoints
  run_XX/                 # Per-run directories
    stage_name/           # Per-stage: solution.pkl, metadata.json, penalties.json
    run_metadata.json     # Run-level config (constraint_slack, excluded, etc.)
  latest/                 # Copy of most recent successful checkpoint
logs/                     # Solver logs
tests/                    # Test suite
scripts/                  # Utility scripts
```

## Decision Variable Structure

```python
key = (team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)
#      0      1      2      3    4         5     6     7     8         9           10
X[key] = BoolVar  # 1 = game scheduled, 0 = not
```

Note: `team1` is always alphabetically before `team2`. Home/away is determined by venue (field_location), NOT by position in the key.

### Dummy Slots (Y variables)

Dummy timeslots are overflow slots not attached to a real time or venue. They ease solver burden by providing extra scheduling capacity. Controlled by:
- `SEASON_CONFIG['num_dummy_timeslots']` — how many dummy slots to create (default 3)
- `PENALTY_WEIGHTS['dummy_slots']` — penalty per dummy slot used (0 = free, higher = discouraged)

Dummy variables use short 4-tuple keys `(t1, t2, grade, index)` and are merged into X. Constraints exclude them via two mechanisms:
- `len(key) < 11` — skips short dummy keys
- `and t.day` / `not key[3]` — skips when iterating timeslots

Game-count constraints (e.g. `EqualGamesAndBalanceMatchUps`) explicitly include dummy vars so the solver can use them as overflow. The objective penalises their use: `Maximize(sum(X) - dummy_penalty - soft_penalties)`.

## Season Dates

Only two date fields:
- `start_date` — first playing day
- `end_date` — last club game before finals

This system does NOT schedule finals. The `end_date` is always the last regular season round, not the grand final date.

## Constraint System

### Severity Levels (used by --relax flag and severity-staged solving)

| Level | Name | Constraints | Relaxable? |
|-------|------|------------|------------|
| 1 | CRITICAL | NoDoubleBooking (Teams/Fields), EqualGamesAndBalanceMatchUps, EqualMatchUpSpacing, FiftyFiftyHomeandAway, MaitlandHomeGrouping, MaxMaitlandHomeWeekends, PHLAndSecondGradeAdjacency, PHLAndSecondGradeTimes | Never |
| 2 | HIGH | ClubDay, AwayAtMaitlandGrouping, TeamConflict | With --relax |
| 3 | MEDIUM | ClubGradeAdjacency, ClubVsClubAlignment, ClubGameSpread | With --relax |
| 4 | LOW | MaximiseClubsPerTimeslotBroadmeadow, MinimiseClubsOnAFieldBroadmeadow | Yes |
| 5 | VERY LOW | EnsureBestTimeslotChoices, PreferredTimesConstraint | Yes |

### Constraint Slack (`--slack N`)

The `--slack` CLI flag loosens specific constraints. Applied to:
- `EqualMatchUpSpacingConstraint`: reduces min_gap toward floor. Formula: `min_gap = max(min(T//2, T-2), T-2 - spacing_base_slack - slack)`. Config: `spacing_base_slack` in `CONSTRAINT_DEFAULTS` (default 0)
- `AwayAtMaitlandGrouping`: max away clubs = 3 + slack
- `MaitlandHomeGrouping`: max consecutive home weeks = 1 + slack (sliding window)
- `ClubVsClubAlignment`: loosens alignment requirement
- `MaximiseClubsPerTimeslotBroadmeadow`: reduces minimum
- `MinimiseClubsOnAFieldBroadmeadow`: increases maximum
- `ClubGameSpread`: increases spread limit (upper) AND allows more double-ups (lower). Formula: `gap >= -(max_overlap + slack)` and `gap <= max_gap + slack`

Slack is stored in checkpoint metadata (`constraint_slack` key) and used by `DrawTester`.

### Key Constraint Details

**FiftyFiftyHomeandAway**: Enforces per-PAIR balance (each Maitland/Gosford team vs each opponent individually), NOT aggregate balance. With odd meeting counts (e.g., 3 meetings), all pairs may land 1H/2A causing aggregate imbalance. This is by design.

**MaitlandHomeGrouping**: Uses sliding window of size (max_consecutive + 1) to enforce max N consecutive home weeks. With slack=0, no back-to-back; slack=3, max 4 consecutive.

**PHLAndSecondGradeAdjacency**: Uses 180-minute time window with location logic — within 180 min must be same location, outside 180 min must be different location.

### 19 Constraint Pairs (original + AI)

All audited for parity. 5 bugs were fixed in AI versions:
1. `EqualMatchUpSpacingConstraintAI` - was a no-op
2. `EnsureBestTimeslotChoicesAI` - missing slot bounding
3. `ClubVsClubAlignmentAI` - missing Sunday field-alignment
4. `MaximiseClubsPerTimeslotBroadmeadowAI` - missing dynamic hard minimum
5. `PHLAndSecondGradeAdjacencyAI` - missing same-location enforcement

6th bug fixed: `MaitlandHomeGrouping` (both original + AI) - pairwise BoolVar check was a no-op with any slack ≥ 1, replaced with sliding window.

## Analyzing Checkpoints

Checkpoints can be tested directly without converting to draw JSON:

```python
import pickle
from config import load_season_data
from analytics.tester import DrawTester

data = load_season_data(2026)
# IMPORTANT: Set constraint_slack to match the solver run's --slack value
data['constraint_slack'] = {
    'EqualMatchUpSpacingConstraint': 3,
    'AwayAtMaitlandGrouping': 3,
    'MaitlandHomeGrouping': 3,
    'ClubVsClubAlignment': 3,
    'MaximiseClubsPerTimeslotBroadmeadow': 3,
    'MinimiseClubsOnAFieldBroadmeadow': 3,
    'ClubGameSpread': 3,
}

with open('checkpoints/latest/solution.pkl', 'rb') as f:
    solution = pickle.load(f)

tester = DrawTester.from_X_solution(solution, data, description='Checkpoint analysis')
report = tester.run_violation_check()
```

Key classes:
- `DrawTester.from_X_solution(X_solution, data, description)` — test directly from pickle
- `DrawTester.from_file(path, data)` — test from saved draw JSON
- `DrawStorage.from_X_solution(X_solution, description)` — convert pickle to DrawStorage

### DrawTester Constraint Checks

The tester runs these checks (matching solver constraint behavior):
- `NoDoubleBookingTeams` — one game per team per week
- `NoDoubleBookingFields` — one game per field per date+slot (not week+slot)
- `EqualGames` — each team plays expected games (from `num_rounds`)
- `BalancedMatchups` — pair meetings within base/base+1
- `FiftyFiftyHomeAway` — per-pair home/away balance for Maitland/Gosford
- `MaxMaitlandHomeWeekends` — sliding window consecutive home weeks (slack-aware)
- `AwayAtMaitlandGrouping` — max away clubs per week (slack-aware)
- `ClubGradeAdjacency` — adjacent grades same club not same timeslot
- `PHLAndSecondGradeAdjacency` — 180-min window + same-location rule

**Important**: The tester uses `game.date` (not `game.week`) for field/slot comparisons because a week can have both Friday and Sunday games on different dates.

## Game Count Calculation

Three-tier override system in `utils.py::max_games_per_grade()`:

| Priority | Method | Config Source | Example |
|----------|--------|--------------|---------|
| 1 (highest) | Exact override | `GRADE_ROUNDS_OVERRIDE['2nd'] = 18` | "2nd grade plays exactly 18" |
| 2 | Max weekends | `MAX_WEEKENDS_PER_GRADE['PHL'] = 22` | PHL uses 22 weeks → formula gives 22 |
| 3 (default) | Formula | `max_rounds` (default 20) | `g0 = min(2*max_matches/T, grade_max_rounds)` |

2026 results: PHL=20, 2nd=18, 3rd=20, 4th=18, 5th=16, 6th=20

## Quick Commands

```powershell
# Generate (ALWAYS background, ALWAYS --year)
.\.venv\Scripts\python.exe run.py generate --year 2026

# Generate with slack (loosens constraints)
.\.venv\Scripts\python.exe run.py generate --year 2026 --simple --slack 3

# Lock weeks from prior draw, re-solve rest
.\.venv\Scripts\python.exe run.py generate --year 2026 --locked draws/2026/current.json --lock-weeks 1,2,3

# Full command with exclusions, hints, and slack
.\.venv\Scripts\python.exe run.py generate --year 2026 --simple --slack 3 --locked draws/2026/current.json --lock-weeks 1 --workers 20 --hint checkpoints/run_XX/simple_solve_intermediate_N/solution.pkl --exclude ConstraintName1 ConstraintName2

# Test a draw for violations
.\.venv\Scripts\python.exe run.py test current --year 2026

# Pre-season report
.\.venv\Scripts\python.exe run.py preseason --year 2026

# Diagnose infeasibility
.\.venv\Scripts\python.exe run.py diagnose --year 2026 --timeout 60

# Swap games
.\.venv\Scripts\python.exe run.py swap current G001 G002 --year 2026 --save

# Run tests
.\.venv\Scripts\python.exe -m pytest tests/ -v

# Export formatted schedule xlsx from DrawStorage
python -c "from analytics.storage import DrawStorage; d=DrawStorage.load('draws/2026/current.json'); d.export_schedule_xlsx('output.xlsx')"

# Export subset of weeks
python -c "from analytics.storage import DrawStorage; d=DrawStorage.load('draws/2026/current.json'); d.export_schedule_xlsx('first_4_weeks.xlsx', weeks=[1,2,4,5], sheet_title='First 4 Playing Weeks')"

# Export to Revo format (external hockey system)
python -c "from analytics.storage import DrawStorage, export_draw_to_revformat; from config import load_season_data; d=DrawStorage.load('draws/2026/current.json'); export_draw_to_revformat(d, load_season_data(2026))"
```

## Export Functions

### DrawStorage.export_schedule_xlsx()
The primary export for human-readable schedules. Works directly from DrawStorage (no Roster or data dict needed).

```python
draw = DrawStorage.load('draws/2026/current.json')
draw.export_schedule_xlsx('output.xlsx')                          # full draw
draw.export_schedule_xlsx('subset.xlsx', weeks=[1,2,4,5])         # specific weeks
draw.export_schedule_xlsx('subset.xlsx', weeks=[1,2], sheet_title='Rounds 1-2')
```

Features: alternating week background colours, blue field sub-headers, column headers, borders, bye listings per week. Grouped by week → date → field (EF/WF/SF at NIHC first, then away venues).

### export_draw_to_revformat()
Exports CSV for the Revo hockey management system with full club name mapping, grade mapping, and bye entries. Supports `week_limit` parameter.

### export_roster_to_excel()
Plain xlsx with one sheet per week (from solver Roster object). Used internally by `save_solver_output()`. No formatting — prefer `export_schedule_xlsx()` for human consumption.

### Report Generation (analytics/reports.py)
- `ClubReport.generate_for_club(club)` — per-club xlsx (schedule, opponents, home/away, byes)
- `GradeReport.generate(grade)` / `generate_all_grades(dir)` — grade schedules + matchup matrices
- `ComplianceCertificate.generate()` — constraint compliance pass/fail
- `generate_html_report()` — styled HTML summary
- `generate_all_reports(draw_path, data, output_dir)` — all of the above in one call

## Adding a New Constraint

```python
class MyConstraintAI(Constraint):
    """What this constraint enforces."""
    def apply(self, model, X, data):
        current_week = data.get('current_week', 0)
        groups = defaultdict(list)
        for key, var in X.items():
            if len(key) < 11 or not key[3]:
                continue
            if key[6] <= current_week:
                continue
            groups[(key[6], key[4])].append(var)
        for group_key, vars_list in groups.items():
            model.Add(sum(vars_list) <= 1)
```

Register in `main_staged.py` STAGES or STAGES_AI dict, and add to `CONSTRAINT_TO_SEVERITY` in `constraints/severity.py`.

## Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `load_season_data()` | `config/__init__.py` | Load complete data dict |
| `generate_X()` | `utils.py` | Create decision variables (filtering happens here) |
| `convert_X_to_roster()` | `utils.py` | Solution -> Roster |
| `max_games_per_grade()` | `utils.py` | Calculate games per team per grade (3-tier override) |
| `DrawStorage.load()` | `analytics/storage.py` | Load draw from JSON |
| `DrawStorage.from_X_solution()` | `analytics/storage.py` | Create draw from checkpoint pickle |
| `DrawStorage.export_schedule_xlsx()` | `analytics/storage.py` | Export formatted schedule xlsx (week colours, field headers, byes) |
| `DrawStorage.save()` | `analytics/storage.py` | Save draw to JSON |
| `DrawAnalytics.export_analytics_to_excel()` | `analytics/storage.py` | Export 9-sheet analytics workbook |
| `export_draw_to_revformat()` | `analytics/storage.py` | Export CSV for Revo hockey management system |
| `export_roster_to_excel()` | `utils.py` | Export Roster to plain xlsx (one sheet per week, from solver) |
| `DrawTester.from_X_solution()` | `analytics/tester.py` | Test checkpoint directly (slack-aware) |
| `DrawTester.run_violation_check()` | `analytics/tester.py` | Post-hoc violation checking |
| `DrawVersionManager.save_solver_output()` | `analytics/versioning.py` | Unified output (JSON + Excel + analytics) |
| `DrawVersionManager.save_new_draw()` | `analytics/versioning.py` | Version a DrawStorage (major/minor) |
| `DrawVersionManager.save_modified_draw()` | `analytics/versioning.py` | Save modified draw as minor version with diff |
| `generate_club_report()` | `analytics/reports.py` | Per-club report (schedule, opponents, home/away) |
| `generate_compliance_certificate()` | `analytics/reports.py` | Constraint compliance certificate |
| `generate_all_reports()` | `analytics/reports.py` | All reports (clubs, grades, compliance, HTML) |

## Draw JSON Schema

Draw files (`draws/{year}/current.json`, `draws/{year}/versions/draw_v*.json`) use `DrawStorage` from `analytics/storage.py`:

```json
{
  "version": "1.0",
  "created_at": "ISO-8601",
  "description": "Season 2026 draw - severity mode",
  "num_weeks": 24,
  "num_games": 457,
  "teams": [],
  "games": [
    {
      "game_id": "G00000",
      "team1": "Maitland PHL",
      "team2": "Norths PHL",
      "grade": "PHL",
      "week": 1,
      "round_no": 1,
      "date": "2026-03-22",
      "day": "Sunday",
      "time": "11:30",
      "day_slot": 3,
      "field_name": "EF",
      "field_location": "Newcastle International Hockey Centre"
    }
  ],
  "metadata": { ... }
}
```

### Draw Metadata (populated by `save_solver_output()`)

| Key | Type | Description |
|-----|------|-------------|
| `generated_at` | str | Timestamp when draw was generated |
| `mode` | str | Solve mode: "simple", "staged", "severity" |
| `year` | int | Season year |
| `solver_config` | dict | Mode, use_ai, workers, relax, locked_weeks, excluded constraints, constraint_slack |
| `constraints_applied` | list | Each constraint name + stage it was applied in |
| `forced_games` | list | Copy of `FORCED_GAMES` config entries |
| `blocked_games` | list | Copy of `BLOCKED_GAMES` config entries |
| `forced_game_outcomes` | list | Each forced game + `satisfied: true/false` |
| `blocked_game_outcomes` | list | Each blocked game + `respected: true/false` |
| `penalties` | dict | Soft constraint penalty counts |
| `stats` | dict | Games by grade, by venue, friday count, date range, weeks |

### Checkpoint Metadata (`checkpoints/run_XX/stage/metadata.json`)

Each checkpoint includes: stage, status, solve_time, timestamp, num_scheduled_games, total_variables, run_id, year, description, mode, use_ai, locked_weeks, excluded_constraints, constraint_slack, penalty_weights, forced_games, blocked_games, and constraints_applied.

Run-level metadata (`run_metadata.json`) additionally includes: environment info, solver_config, data shape.

## Monitoring a Running Solver

When a solver is running in background, do NOT rely solely on the task output buffer (it shows early output, not current state). Instead:

1. **Check checkpoints first**: `ls checkpoints/run_XX/` — intermediate solutions are saved as `simple_solve_intermediate_N/` directories. Count them and check the latest `metadata.json` for solve time, game count, and status.
2. **Read the log file**: `logs/solver_YYYYMMDD_HHMMSS*.log` has the full CP-SAT output including `#1`, `#2` etc. solution lines with objective values.
3. **Check metadata**: `cat checkpoints/run_XX/simple_solve_intermediate_N/metadata.json` shows status (FEASIBLE/OPTIMAL), solve_time, num_scheduled_games.
4. **The task output buffer only captures from process start** — a non-blocking read shows the first N bytes, which is typically presolve output from the first few minutes, NOT current solver state.

The CP-SAT log format: `best:-inf` means no solution found *at that log timestamp*. `best:NNN` means a solution exists with objective NNN. Don't confuse early log lines with current state.

## Common Pitfalls

- **PHL_PREFERENCES** only supports `preferred_dates` key
- **Cannot create NEW timeslots** - only existing `DAY_TIME_MAP` slots get variables
- **Hints are not locks** - `model.AddHint()` is a suggestion, `model.Add(X[key] == 1)` is a lock
- **day_slot indexing** - `generate_timeslots()` uses 1-indexed day_slot; verify with `scripts/verify_locked_keys.py`
- **`test` vs `verify`** - `test` is fast post-hoc statistical check; `verify` uses solver constraint propagation (definitive)
- **Memory** - use `--low-memory` or `--workers 4` to prevent OOM; never use all cores
- **Home/away** - determined by venue (field_location), NOT by team1/team2 position in variable key
- **Week vs date** - a single week can have both Friday and Sunday games; always use `date` not `week` when comparing game slots
- **Per-pair vs aggregate** - FiftyFiftyHomeAway is per-pair; aggregate home/away can be imbalanced even when all pairs are balanced
- **Slack >= 1 effects** - MaitlandHomeGrouping uses sliding window (correct); check constraint docstrings for slack semantics

## Draw Review Checklist

When reviewing, testing, or publishing a draw, always check:

- **Last game of the day on West Field**: If only one field is being used for the last timeslot of the day at NIHC (Broadmeadow), that game should be on West Field (WF), not East Field (EF). Flag this to the user if it's not the case.
- **7pm (19:00) games**: These are the worst timeslot. Flag any non-PHL-Friday games scheduled at 7pm — they should be minimised.

## Skills

Custom Claude Code skills are available in `.claude/commands/`:
- `generate-draw` - Generate a new draw with proper flags
- `test-draw` - Test a draw for constraint violations
- `diagnose` - Diagnose and resolve infeasibility
- `preseason-report` - Generate pre-season configuration report
- `swap-games` - Swap two games and check violations

## Additional Documentation

| Document | Purpose | Read When |
|----------|---------|-----------|
| `docs/ai/AI_OPERATIONS_MANUAL.md` | Complete technical reference | Deep dives |
| `docs/ai/CONFIGURATION_REFERENCE.md` | All config parameters | Changing config |
| `docs/ai/CONSTRAINT_APPLICATION.md` | How to apply restrictions | Adding constraints |
| `docs/ai/GAME_TIME_DICTIONARIES.md` | PHL/2nd grade filtering | Modifying game times |
| `docs/DRAW_RULES.md` | Constraint documentation | Understanding rules |
| `seasons/RULES.md` | Season-specific rules | Season context |
