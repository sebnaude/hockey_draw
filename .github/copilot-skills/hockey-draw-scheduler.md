# Hockey Draw Scheduler - Copilot Skill File

## Project Overview

This is a **Hockey Draw/Schedule Generation System** using OR-Tools CP-SAT constraint programming solver. It generates season schedules for a hockey competition with multiple clubs, teams, grades, fields, and timeslots.

## Quick Reference

### Running the System

**IMPORTANT: `--year` is REQUIRED for all commands!**

```powershell
# Navigate to project directory
cd c:\Users\c3205\Documents\Code\python\draw

# Activate virtual environment
.\.venv\Scripts\activate

# Generate a new draw (original constraints)
.\.venv\Scripts\python.exe run.py generate --year 2025

# Generate 2026 season
.\.venv\Scripts\python.exe run.py generate --year 2026

# Generate with automatic constraint relaxation (if infeasible)
.\.venv\Scripts\python.exe run.py generate --year 2026 --relax

# Generate with AI constraints (opt-in alternative constraint set)
.\.venv\Scripts\python.exe run.py generate --year 2025 --ai

# Generate with simple (non-staged) mode
.\.venv\Scripts\python.exe run.py generate --year 2025 --simple

# Generate with AI constraints, simple mode, exclude problematic constraints
.\.venv\Scripts\python.exe run.py generate --year 2025 --simple --ai --exclude EnsureBestTimeslotChoices MinimiseClubsOnAFieldBroadmeadow MaximiseClubsPerTimeslotBroadmeadow

# Generate with low memory usage (4 workers)
.\.venv\Scripts\python.exe run.py generate --year 2025 --low-memory

# Generate with high performance (all cores)
.\.venv\Scripts\python.exe run.py generate --year 2025 --high-performance

# Generate with custom worker count (max 14 of 16 to avoid OOM)
.\.venv\Scripts\python.exe run.py generate --year 2025 --workers 14

# Resume from checkpoint
.\.venv\Scripts\python.exe run.py generate --year 2025 --resume run_13 stage1_required

# Test a draw for violations - requires --year
.\.venv\Scripts\python.exe run.py test draws/draw_file.json --year 2025

# Analyze a draw - requires --year
.\.venv\Scripts\python.exe run.py analyze draws/draw_file.json --year 2025
```

### Important: Solver Execution

**CRITICAL**: Solver runs can take a LONG time:
- Stage 1 (stage1_required): Up to 2 hours (usually 5-30 minutes)
- Stage 2 (stage2_soft): Up to 72 hours (3 days)

When starting a solver run:
1. **Always run in background** (`isBackground: true`) for terminal commands
2. **Check terminal output periodically** using `get_terminal_output`
3. **Logs are saved** in the `logs/` directory for debugging
4. **Checkpoints are saved** in `checkpoints/run_X/` for resumption

### Key Files

| File | Purpose |
|------|---------||
| `run.py` | CLI entry point - use this to run the system |
| `main_staged.py` | Staged solving main logic (also has `--simple` mode) |
| `config/season_{year}.py` | Season-specific configuration (dates, fields, times) |
| `config/season_template.py` | Template for adding new seasons |
| `constraints.py` | All constraint implementations (READ-ONLY — never edit) |
| `constraints_ai.py` | AI-enhanced constraint implementations (edit this one) |
| `solver_diagnostics.py` | Logging and resource monitoring |
| `models.py` | Data models (Team, Field, Club, etc.) |
| `utils.py` | Utility functions + `build_season_data()` |

### AI Constraints

The system has two parallel constraint sets:

- **`constraints.py`** — Original human-written constraints. **NEVER edit this file.**
- **`constraints_ai.py`** — AI-enhanced versions with cleaner code and equivalent behaviour.

All 18 constraint pairs have been audited and the AI versions brought to full parity. Use the `--ai` flag to select the AI constraint set.

**⚠️ CRITICAL**: `constraints.py` is read-only. If an AI constraint doesn't match the original, fix the AI version, never the original.

#### Three constraints are excluded from initial AI runs:
- `EnsureBestTimeslotChoices` / `EnsureBestTimeslotChoicesAI`
- `MinimiseClubsOnAFieldBroadmeadow` / `MinimiseClubsOnAFieldBroadmeadowAI`
- `MaximiseClubsPerTimeslotBroadmeadow` / `MaximiseClubsPerTimeslotBroadmeadowAI`

### Data Locations

| Directory | Content |
|-----------|---------|
| `config/season_{year}.py` | Season config (dates, times, field unavailabilities) |
| `data/{year}/teams/` | Team CSV files (one per club) |
| `data/{year}/noplay/` | No-play dates configuration |
| `data/{year}/field_availability/` | Field availability |
| `checkpoints/` | Solver checkpoints for resumption |
| `draws/` | Generated schedule outputs |
| `logs/` | Detailed solver logs |

### Adding a New Season

To add support for a new year (e.g., 2027):

1. **Copy the template**: `config/season_template.py` → `config/season_2027.py`
2. **Update all year references** in the new file (search for `9999`)
3. **Create team data folder**: `data/2027/teams/`
4. **Add team CSV files** for each club
5. **Update dates**: start_date, end_date, field unavailabilities, club days
6. The system will **automatically detect** the new config file

### Solver Stages

The staged solver runs **2 stages**:

1. **stage1_required - Required Constraints** (max 2h): 14 core + structural constraints
   - No double-booking teams/fields
   - Equal games per team, 50/50 home/away
   - PHL and 2nds adjacency + times
   - Team conflicts, Max Maitland home weekends
   - Club day events, Equal matchup spacing
   - Club grade adjacency, Club vs Club alignment
   - Maitland home grouping, Away at Maitland grouping

2. **stage2_soft - Soft Preferences** (max 72h): 4 optimization constraints
   - Ensure best timeslot choices
   - Maximise clubs per timeslot at Broadmeadow
   - Minimise clubs on a field at Broadmeadow
   - Preferred times / no-play constraints

### Running Specific Stages

```powershell
# Run both stages (default)
.\.\.venv\Scripts\python.exe run.py generate --year 2025

# Run ONLY stage 1
.\.\.venv\Scripts\python.exe run.py generate --year 2025 --stages stage1_required

# Run ONLY stage 2 (must have checkpoint from stage 1)
.\.\.venv\Scripts\python.exe run.py generate --year 2025 --stages stage2_soft --resume run_X stage1_required
```

### Diagnose Command (Infeasibility Resolution)

Find which constraint causes infeasibility and optionally auto-relax:

```powershell
# Find blocking constraint
.\.\.venv\Scripts\python.exe run.py diagnose --year 2025

# Auto-relax constraints iteratively until feasible
.\.\.venv\Scripts\python.exe run.py diagnose --year 2025 --resolve

# Options
--stage stage1_required  # Which stage to analyze (default)
--timeout 5.0            # Seconds per feasibility test
--max-iterations 10      # Max relaxation iterations
--ai                     # Use AI constraint implementations
```

### --relax Flag (Severity-Based Relaxation)

The `--relax` flag provides automatic infeasibility resolution during generation:

```powershell
# Generate with automatic constraint relaxation
.\.venv\Scripts\python.exe run.py generate --year 2026 --relax

# Works with other flags
.\.venv\Scripts\python.exe run.py generate --year 2026 --relax --simple
.\.venv\Scripts\python.exe run.py generate --year 2026 --relax --ai
```

**How it works:**
1. Tests with all constraints → if INFEASIBLE, starts severity group testing
2. Drops severity level 4 (LOW) constraints and retests
3. If still INFEASIBLE, drops level 3 (MEDIUM), then level 2 (HIGH)
4. Identifies the blocking severity group
5. Relaxes ALL constraints in that group (slack +1)
6. Solves with ALL constraints together (never locks partial solutions)

**Severity Levels:**
| Level | Name | Constraints | Can Relax? |
|-------|------|-------------|------------|
| 1 | CRITICAL | NoDoubleBooking, EqualGames, PHL adjacency, HomeAway | Never |
| 2 | HIGH | ClubDay, MaitlandGrouping, TeamConflict | Yes |
| 3 | MEDIUM | MatchUpSpacing, GradeAdjacency, ClubVsClub | Yes |
| 4 | LOW | TimeslotChoices, ClubDensity, PreferredTimes | Yes |

**Key principle**: Never lock in partial solutions. Always solve with all constraints together.

See `severity_relaxation.py` for implementation details.

### Checkpoint/Resume System

Checkpoints are saved after each successful stage in `checkpoints/run_X/`.

To resume:
```powershell
# Resume from the last successful stage (stage1_required completed)
.\.venv\Scripts\python.exe run.py generate --year 2025 --resume run_13 stage1_required
```

### Common Issues & Solutions

#### Memory Exhaustion
**Symptoms**: Process exits with code 1, no Python traceback
**Solution**: Use `--low-memory` flag or `--workers 4`

#### StatusName() Deprecated
**Symptoms**: `TypeError: 'CpSolverStatus' object is not callable`
**Solution**: Already fixed - use `solver.status_name(status)` not `solver.StatusName()`

#### Stage 2 Crash
**Symptoms**: Process exits during Stage 2 without error
**Solution**: Check `logs/` directory for detailed diagnostics

### Resource Monitoring

The system now includes resource monitoring:
- Memory usage logged every 30 seconds during solve
- Pre/post solve snapshots
- Peak memory tracking

Requires `psutil`:
```powershell
.\.venv\Scripts\python.exe -m pip install psutil
```

### OR-Tools Solver Parameters

Key parameters (configured via `solver_diagnostics.py`):
- `num_workers`: Parallel workers (4 = low memory, 0 = all cores)
- `linearization_level`: 0=none, 1=basic, 2=full (higher = more memory)
- `cp_model_probing_level`: 1-3 (higher = better bounds, more memory)

### Model Size

Current 2025 season:
- ~118,000 decision variables
- ~47 teams across 6 grades
- ~696 timeslots
- ~454 games to schedule

### Testing

```powershell
# Run all tests (216 tests, ~21s)
.\.venv\Scripts\python.exe -m pytest

# Run specific test file
.\.venv\Scripts\python.exe -m pytest tests/test_constraints.py

# Run comprehensive AI constraint tests (70 tests)
.\.venv\Scripts\python.exe -m pytest tests/test_ai_constraints_comprehensive.py -v

# Run with verbose output
.\.venv\Scripts\python.exe -m pytest -v
```

### Test Suite Structure

| File | Tests | Description |
|------|-------|-------------|
| `test_ai_constraints_comprehensive.py` | 70 | Full AI constraint coverage: feasibility, rejection, parity, combined, incremental |
| `test_constraints.py` | 14 | Original constraint unit tests |
| `test_constraints_ai.py` | 8 | AI constraint unit tests + equivalence |
| `test_constraints_comprehensive.py` | 8 | Extended constraint tests |
| `test_constraints_equivalence.py` | 18 | Original vs AI parity tests |
| `test_analytics_reports.py` | 14 | Analytics report tests |
| `test_analytics_storage.py` | 18 | Draw storage tests |
| `test_analytics_tester.py` | 18 | Draw tester tests |
| `test_draw_outcomes.py` | 4 | Outcome validation tests |
| `test_utils.py` | 20 | Utility function tests |

### OOM Prevention in Tests

The combined constraint tests limit solver resources to prevent OOM:
- `solve()` defaults to `workers=8` (not all 16 cores)
- 4-grade combined tests use max 6 weeks (not 10+)
- Incremental tests use 2 grades with `workers=4`
- Always use `--workers 14` (not all 16) for production solver runs

### Grade Hierarchy

| Grade | Description |
|-------|-------------|
| PHL | Premier Hockey League (top grade) |
| 2nd | Second grade |
| 3rd | Third grade |
| 4th | Fourth grade |
| 5th | Fifth grade |
| 6th | Sixth grade |

### Clubs

- Colts, Crusaders, Gosford, Maitland, Norths
- Port Stephens, Souths, Tigers, University, Wests

### Fields

- Newcastle International Hockey Centre (SF, EF, WF)
- Maitland Park (Maitland Main Field)
- Central Coast Hockey Park (Wyong Main Field)

## Development Guidelines

### Adding New Constraints

1. Create constraint class in `constraints.py`:
```python
class MyConstraint(Constraint):
    def apply(self, model, X, data):
        # Add constraints to model
        pass
```

2. Add to appropriate stage in `main_staged.py`:
```python
STAGES = {
    'stage2_strong': {
        'constraints': [
            ...,
            MyConstraint,
        ],
    },
}
```

### Modifying Solver Behavior

Edit `solver_diagnostics.py`:
- `SolverConfig` class for parameter tuning
- `ResourceMonitor` for memory tracking
- Logging configuration

### Debug Mode

Check logs in `logs/solver_YYYYMMDD_HHMMSS.log` for:
- Detailed constraint application
- Memory snapshots
- Solve progress
- Error tracebacks
