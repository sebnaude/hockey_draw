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
```

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

The solver uses two stages:

### Stage 1: Required Constraints (stage1_required)

**Constraints included:**
- NoDoubleBookingTeams
- NoDoubleBookingFields
- EnsureEqualGamesAndBalanceMatchUps
- PHLAndSecondGradeAdjacency
- PHLAndSecondGradeTimes
- FiftyFiftyHomeandAway

**Typical time:** 5-30 minutes

### Stage 2: Soft Constraints (stage2_soft)

**Constraints added:**
- ClubDayConstraint
- EqualMatchUpSpacingConstraint
- ClubGradeAdjacencyConstraint
- ClubVsClubAlignment
- ClubGameSpread
- All penalty-based constraints

**Typical time:** Up to 72 hours

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

After a successful solve, ALL output is automatically saved to the versioned directory structure. Both staged and simple modes use the same unified output system.

### Draw Output (draws/{year}/)

| File | Location | Content |
|------|----------|---------|
| **Latest draw JSON** | `draws/{year}/current.json` | **Always the latest draw — use this for AI operations** |
| **Latest schedule** | `draws/{year}/current.xlsx` | Latest schedule Excel (weekly sheets) |
| **Latest analytics** | `draws/{year}/current_analytics.xlsx` | Latest analytics multi-sheet Excel |
| **Violations** | `draws/{year}/current_violations.txt` | Latest violation report (if any) |
| **Changelog** | `draws/{year}/CHANGELOG.md` | Auto-generated version history |
| **Versioned draws** | `draws/{year}/versions/draw_v{X}.{Y}.json` | All versioned draw files |
| **Versioned Excel** | `draws/{year}/versions/draw_v{X}.{Y}.xlsx` | All versioned schedules |

### Checkpoint Output (checkpoints/)

| File | Location | Content |
|------|----------|---------|
| **Latest checkpoint** | `checkpoints/latest/solution.pkl` | **Latest successful solver state** |
| **Latest metadata** | `checkpoints/latest/metadata.json` | Latest solve statistics |
| **Latest pointer** | `checkpoints/latest/pointer.json` | Which run/stage this came from |
| **Run checkpoints** | `checkpoints/run_X/stage_Y/solution.pkl` | Per-run, per-stage checkpoints |
| **Run metadata** | `checkpoints/run_X/stage_Y/metadata.json` | Per-stage solve statistics |

### Logs

| File | Location | Content |
|------|----------|---------|
| Solver log | `logs/solver_YYYYMMDD_HHMMSS.log` | Full solve log |

### Finding the Latest Output

**For AI assistants:** Always use these paths to find the latest state:
- **Latest draw:** `draws/{year}/current.json`
- **Latest checkpoint:** `checkpoints/latest/`
- **Version history:** `draws/{year}/CHANGELOG.md`

**For version-specific access:**
- Use `draws/{year}/versions/draw_v{X}.{Y}.json`
- Or use the `run.py test v2.0 --year 2026` shorthand

### Path Aliases in Commands

The `run.py` commands support special aliases for draw paths:

```powershell
# "current" or "latest" resolves to draws/{year}/current.json
.\.venv\Scripts\python.exe run.py test current --year 2026

# Version strings resolve to draws/{year}/versions/draw_vX.Y.json  
.\.venv\Scripts\python.exe run.py test v2.0 --year 2026

# Direct paths still work
.\.venv\Scripts\python.exe run.py test draws/2026/versions/draw_v1.0.json --year 2026
```

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

| Type | Location | Notes |
|------|----------|-------|
| **Latest draw** | `draws/{year}/current.json` | **Always check here first** |
| **Latest schedule** | `draws/{year}/current.xlsx` | Auto-updated on every solve |
| **Latest analytics** | `draws/{year}/current_analytics.xlsx` | Auto-updated on every solve |
| **Versioned draws** | `draws/{year}/versions/` | All historical versions |
| **Changelog** | `draws/{year}/CHANGELOG.md` | Auto-generated history |
| **Latest checkpoint** | `checkpoints/latest/` | Latest solver state (pkl) |
| **Run checkpoints** | `checkpoints/run_X/` | Per-run checkpoint history |
| **Logs** | `logs/` | Solver log files |
| **Reports** | `reports/` | Pre-season and stakeholder reports |

---

## Draw Versioning System

The system uses semantic versioning with automatic CHANGELOG generation. **All solver modes (staged, simple, resume) use the same unified output system.**

### How It Works

1. **Every successful solve** creates a new versioned draw in `draws/{year}/versions/`
2. **current.json** is automatically updated to always point to the latest
3. **current.xlsx** and **current_analytics.xlsx** are also auto-updated
4. **CHANGELOG.md** records every version with timestamps and descriptions
5. **Game swaps** via `run.py swap --save` create minor versions with diffs

### Version Scheme

- **Major (X.0):** Complete regeneration (from `generate` command)
- **Minor (X.Y):** Incremental updates (game swaps, manual fixes)

### Directory Structure

```
draws/2026/
├── current.json              # ← ALWAYS the latest draw (AI: look here first!)
├── current.xlsx              # ← Latest schedule Excel
├── current_analytics.xlsx    # ← Latest analytics
├── current_violations.txt    # ← Latest violations (if any, deleted when clean)
├── CHANGELOG.md              # Auto-generated version history
└── versions/                 # All versioned draws
    ├── draw_v1.0.json        # Major version 1
    ├── draw_v1.0.xlsx        # Schedule for v1.0
    ├── draw_v1.1.json        # Minor update (e.g., game swap)
    ├── draw_v2.0.json        # Major version 2 (new generation)
    ├── draw_v2.0.xlsx
    └── violations_v2.0.txt   # Violation report for v2.0
```

### Checkpoint Structure

```
checkpoints/
├── latest/                    # ← ALWAYS the latest solver state
│   ├── solution.pkl           # Solution dictionary
│   ├── metadata.json          # Solve stats (status, time, games)
│   ├── penalties.json         # Penalty breakdown
│   └── pointer.json           # Which run/stage this came from
├── run_1/
│   ├── stage1_required/
│   └── stage2_soft/
└── run_N/
```

### Usage in Python

```python
from analytics.versioning import DrawVersionManager

manager = DrawVersionManager("draws", year=2026)

# Load latest draw (always available after first solve)
draw = manager.load_latest()

# Load specific version
draw = manager.load_version(1, 0)  # v1.0

# Save a modification as minor version
version = manager.save_modified_draw(new_draw, old_draw, "Fixed Maitland clash")

# List all versions
print(manager.list_versions())
```

### Migration from Legacy Structure

If draws exist as `draws/{year}/draw_v*.json` (old flat structure), run:

```powershell
.\.venv\Scripts\python.exe run.py migrate --year 2026
```

This moves all versioned draws into `draws/{year}/versions/` and creates `current.json`.
