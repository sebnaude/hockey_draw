# Hockey Draw Scheduler - Copilot Skill File

## Project Overview

This is a **Hockey Draw/Schedule Generation System** using OR-Tools CP-SAT constraint programming solver. It generates season schedules for a hockey competition with multiple clubs, teams, grades, fields, and timeslots.

## Quick Reference

### Running the System

```powershell
# Navigate to project directory
cd c:\Users\c3205\Documents\Code\python\draw

# Activate virtual environment
.\.venv\Scripts\activate

# Generate a new draw (auto-detected resource config)
.\.venv\Scripts\python.exe run.py generate --year 2025

# Generate with low memory usage (4 workers)
.\.venv\Scripts\python.exe run.py generate --year 2025 --low-memory

# Generate with high performance (all cores)
.\.venv\Scripts\python.exe run.py generate --year 2025 --high-performance

# Generate with custom worker count
.\.venv\Scripts\python.exe run.py generate --year 2025 --workers 8

# Resume from checkpoint
.\.venv\Scripts\python.exe run.py generate --year 2025 --resume run_13 stage1_required

# Test a draw for violations
.\.venv\Scripts\python.exe run.py test draws/draw_file.json

# Analyze a draw
.\.venv\Scripts\python.exe run.py analyze draws/draw_file.json
```

### Important: Solver Execution

**CRITICAL**: Solver runs can take a LONG time:
- Stage 1: Up to 2 hours (usually 5-10 minutes)
- Stage 2: Up to 4 hours
- Stage 3: Up to 8 hours
- Stage 4: Up to 72 hours (3 days)

When starting a solver run:
1. **Always run in background** (`isBackground: true`) for terminal commands
2. **Check terminal output periodically** using `get_terminal_output`
3. **Logs are saved** in the `logs/` directory for debugging
4. **Checkpoints are saved** in `checkpoints/run_X/` for resumption

### Key Files

| File | Purpose |
|------|---------|
| `run.py` | CLI entry point - use this to run the system |
| `main_staged.py` | Staged solving main logic |
| `main.py` | Simple (non-staged) solving |
| `constraints.py` | All constraint implementations |
| `constraints_ai.py` | AI-assisted constraint implementations |
| `solver_diagnostics.py` | Logging and resource monitoring |
| `models.py` | Data models (Team, Field, Club, etc.) |
| `utils.py` | Utility functions |

### Data Locations

| Directory | Content |
|-----------|---------|
| `data/2025/teams/` | Team CSV files (one per club) |
| `data/2025/noplay/` | No-play dates configuration |
| `data/2025/field_availability/` | Field availability |
| `checkpoints/` | Solver checkpoints for resumption |
| `draws/` | Generated schedule outputs |
| `logs/` | Detailed solver logs |

### Solver Stages

The staged solver runs 4 stages:

1. **Stage 1 - Required Constraints** (max 2h): Core rules
   - No double-booking teams
   - No double-booking fields
   - Equal games per team
   - 50/50 home and away

2. **Stage 2 - Strong Structural** (max 4h): Practical quality
   - PHL and 2nds adjacency
   - Club grade adjacency
   - Maitland home weekends

3. **Stage 3 - Medium Constraints** (max 8h): Venue optimization
   - Club day constraints
   - Equal matchup spacing

4. **Stage 4 - Soft Preferences** (max 72h): Quality optimization
   - Preferred times
   - Club alignment
   - Timeslot optimization

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
# Run all tests
.\.venv\Scripts\python.exe -m pytest

# Run specific test file
.\.venv\Scripts\python.exe -m pytest tests/test_constraints.py

# Run with verbose output
.\.venv\Scripts\python.exe -m pytest -v
```

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
