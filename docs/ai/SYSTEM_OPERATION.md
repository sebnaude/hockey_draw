# System Operation Guide

> **Purpose:** How to run the solver and interpret results.
> **When to use:** When generating or testing draws.

---

## Critical Rules

### ⚠️ ALWAYS Use Background Mode

The solver can run for **hours or days**. NEVER use `isBackground: false`.

```powershell
# CORRECT
.\.venv\Scripts\python.exe run.py generate --year 2026
# Use isBackground: true in tool call

# WRONG - will hang the terminal
# isBackground: false
```

### ⚠️ ALWAYS Specify --year

Every command requires `--year`:

```powershell
.\.venv\Scripts\python.exe run.py generate --year 2026
.\.venv\Scripts\python.exe run.py test draws/draw.json --year 2026
.\.venv\Scripts\python.exe run.py preseason --year 2026
```

---

## Available Commands

### Generate a Draw

```powershell
# Standard staged solving (recommended)
.\.venv\Scripts\python.exe run.py generate --year 2026

# Simple mode (all constraints at once)
.\.venv\Scripts\python.exe run.py generate --year 2026 --simple

# With automatic constraint relaxation
.\.venv\Scripts\python.exe run.py generate --year 2026 --relax

# Using AI constraints (opt-in)
.\.venv\Scripts\python.exe run.py generate --year 2026 --ai

# Low memory mode (4 workers)
.\.venv\Scripts\python.exe run.py generate --year 2026 --low-memory

# High performance (all cores)
.\.venv\Scripts\python.exe run.py generate --year 2026 --high-performance

# Custom worker count
.\.venv\Scripts\python.exe run.py generate --year 2026 --workers 14

# Relax specific constraints (add slack to limits)
.\.venv\Scripts\python.exe run.py generate --year 2026 --slack 1
```

### --slack Flag (Constraint Relaxation)

The `--slack N` flag adds N to constraint limits for three configurable constraints:

| Constraint | Base Limit | With --slack 1 | With --slack 2 |
|------------|------------|----------------|----------------|
| `EqualMatchUpSpacingConstraint` | ±1 round | ±2 rounds | ±3 rounds |
| `AwayAtMaitlandGrouping` | Max 3 away clubs | Max 4 | Max 5 |
| `MaitlandHomeGrouping` | No back-to-back | 1 allowed | 2 allowed |

**When to use:** If solver returns INFEASIBLE and `--relax` doesn't help, try `--slack 1`.

### Pre-Season Report

```powershell
.\.venv\Scripts\python.exe run.py preseason --year 2026
```

Outputs:
- Team counts per grade
- Available weekends
- Blocked dates
- PHL configuration summary
- Club requests

### Test a Draw

```powershell
.\.venv\Scripts\python.exe run.py test draws/draw_2026.json --year 2026
```

Checks all constraint violations in an existing draw.

### Analyze a Draw

```powershell
.\.venv\Scripts\python.exe run.py analyze draws/draw_2026.json --year 2026
```

Generates comprehensive analytics Excel file.

### Diagnose Infeasibility

```powershell
# Find which constraint causes infeasibility
.\.venv\Scripts\python.exe run.py diagnose --year 2026

# With automatic resolution
.\.venv\Scripts\python.exe run.py diagnose --year 2026 --resolve
```

### Resume from Checkpoint

```powershell
.\.venv\Scripts\python.exe run.py generate --year 2026 --resume run_13 stage1_required
```

---

## Staged Solving

The solver uses two stages for better performance and checkpoint support.

### How Staged Solving Works (IMPORTANT)

**Critical concept:** OR-Tools CP-SAT does NOT support "locking in" a partial solution and adding constraints. Instead, we use a **HINT-based approach**:

1. **Stage N completes:** Solution is saved as a checkpoint
2. **Stage N+1 starts:** 
   - A fresh model is created with ALL constraints (Stage 1 + Stage 2 + ... + Stage N+1)
   - The Stage N solution is provided as a **HINT** to guide the solver
   - The solver uses the hint as a starting point but may deviate if the combined constraints require it

**When resuming from a checkpoint:**
- The checkpoint solution becomes the HINT
- ALL constraints from all stages up to and including the current stage are applied
- The solver finds the best solution that satisfies all combined constraints

This is NOT the same as "locking in" the prior solution - it's providing guidance.

### Stage 1: Required Constraints (stage1_required)

**Constraints included:**
- NoDoubleBookingTeams
- NoDoubleBookingFields
- EnsureEqualGamesAndBalanceMatchUps
- PHLAndSecondGradeAdjacency
- PHLAndSecondGradeTimes
- FiftyFiftyHomeandAway
- TeamConflictConstraint
- MaxMaitlandHomeWeekends
- ClubDayConstraint
- EqualMatchUpSpacingConstraint
- ClubGradeAdjacencyConstraint
- ClubVsClubAlignment
- MaitlandHomeGrouping
- AwayAtMaitlandGrouping

**Typical time:** 5-30 minutes to several hours

### Stage 2: Soft Constraints (stage2_soft)

**Constraints added (cumulative with Stage 1):**
- EnsureBestTimeslotChoices
- MaximiseClubsPerTimeslotBroadmeadow
- MinimiseClubsOnAFieldBroadmeadow
- PreferredTimesConstraint

**Typical time:** Up to 72 hours

### Resuming from Checkpoint

```powershell
# Resume after stage1_required completed
.\.venv\Scripts\python.exe run.py generate --year 2026 --resume run_13 stage1_required
```

This will:
1. Load the stage1_required solution as a HINT
2. Apply ALL constraints (stage1 + stage2)
3. Run stage2_soft with the hint guiding the search

---

## Monitoring Progress

### Check Log Files

```powershell
Get-Content "logs\solver_YYYYMMDD_HHMMSS.log" -Tail 50
```

### Check Terminal Output

Use `get_terminal_output` tool to see current status.

### Check Checkpoints

```powershell
Get-ChildItem "checkpoints\run_*" -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

---

## Interpreting Solver Status

| Status | Meaning | Action |
|--------|---------|--------|
| `OPTIMAL` | Perfect solution found | Done! |
| `FEASIBLE` | Solution found, may not be optimal | Can use, or wait longer |
| `INFEASIBLE` | No solution exists | Use `--relax` or check constraints |
| `UNKNOWN` | Still searching | Wait, or increase timeout |
| `MODEL_INVALID` | Bug in constraints | Check constraint code |

---

## Solver Output Files

After successful solve:

| File | Location | Content |
|------|----------|---------|
| Solution JSON | `checkpoints/run_X/stage_Y/solution.json` | Raw solution data |
| Metadata | `checkpoints/run_X/stage_Y/metadata.json` | Solve statistics |
| Penalties | `checkpoints/run_X/stage_Y/penalties.json` | Penalty breakdown |
| Log | `logs/solver_YYYYMMDD_HHMMSS.log` | Full solve log |

---

## Timeout Configuration

Default timeouts:
- Stage 1: 3600 seconds (1 hour)
- Stage 2: 259200 seconds (72 hours)

Override with `--timeout`:

```powershell
.\.venv\Scripts\python.exe run.py generate --year 2026 --timeout 7200  # 2 hours
```

---

## Memory Management

The solver can use significant memory. Options:

| Flag | Workers | Memory | Speed |
|------|---------|--------|-------|
| `--low-memory` | 4 | ~4GB | Slow |
| (default) | 8 | ~8GB | Medium |
| `--high-performance` | All | ~16GB+ | Fast |
| `--workers N` | N | Varies | Custom |

If you get OOM errors:
1. Use `--low-memory`
2. Reduce number of grades being solved
3. Close other applications

---

## Common Issues

### Solver Returns INFEASIBLE

1. Run `diagnose` command:
   ```powershell
   .\.venv\Scripts\python.exe run.py diagnose --year 2026
   ```

2. Try with `--relax` flag:
   ```powershell
   .\.venv\Scripts\python.exe run.py generate --year 2026 --relax
   ```

3. Check recent config changes - did you over-constrain?

### Solver Runs Forever

1. Check if making progress (log file shows improving solutions)
2. Current best solution may be good enough
3. Stop and use checkpoint with `--resume`

### Memory Errors

1. Use `--low-memory` flag
2. Reduce worker count
3. Consider solving fewer weeks/grades

---

## Output Locations

| Type | Location |
|------|----------|
| Checkpoints | `checkpoints/run_X/` |
| Logs | `logs/` |
| Draws | `draws/{year}/` |
| Reports | `seasons/{year}/` |
| Pre-season | `reports/` |

---

## Draw Versioning System

The system uses semantic versioning for draws with automatic CHANGELOG generation.

### Version Scheme

- **Major (X.0):** Complete regeneration or structural changes
- **Minor (X.Y):** Incremental updates (game modifications, swaps, fixes)

### Usage

```python
from analytics.versioning import DrawVersionManager

# Initialize for a year
manager = DrawVersionManager("draws", year=2026)

# Save a new major version (fresh generation)
version = manager.save_new_draw(draw, "Initial 2026 season draw")
# Creates: draws/2026/draw_v1.0.json

# Save a minor update (after modifications)
version = manager.save_modified_draw(new_draw, old_draw, "Fixed Maitland clash")
# Creates: draws/2026/draw_v1.1.json

# Load the latest version
draw = manager.load_latest()

# List all versions
print(manager.list_versions())
```

### Directory Structure

```
draws/2026/
├── CHANGELOG.md         # Auto-generated version history
├── draw_v1.0.json       # Major version 1
├── draw_v1.1.json       # Minor update
├── draw_v2.0.json       # New major version
└── current.json         # Copy of latest version
```

### CHANGELOG Format

The CHANGELOG is automatically updated with:
- Version number and timestamp
- Description
- Game count
- For minor versions: diff showing added/removed/modified games
