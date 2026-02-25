# Copilot Instructions for Hockey Draw Scheduler

## IMPORTANT: Always Load Skill File First

**Before performing any action on this codebase, ALWAYS read and understand the skill file:**

📁 `.github/copilot-skills/hockey-draw-scheduler.md`

This skill file contains critical information about:
- How to run the solver correctly
- Expected long running times (hours to days)
- Memory and CPU management options
- Checkpoint/resume system
- Common issues and solutions

## Critical Rules

### 1. Solver Execution
- **NEVER** run solver commands with `isBackground: false` - they can run for HOURS
- **ALWAYS** use `isBackground: true` for generate commands
- **CHECK** terminal output periodically with `get_terminal_output`

### 2. Long-Running Operations
Solver stages have these time limits:
- Stage 1: 2 hours
- Stage 2: 4 hours  
- Stage 3: 8 hours
- Stage 4: 72 hours (3 days!)

### 3. Memory Management
If encountering crashes without Python tracebacks:
1. Check logs in `logs/` directory
2. Recommend `--low-memory` flag
3. Suggest reducing workers: `--workers 4`

### 4. OR-Tools API
- Use `solver.status_name(status)` NOT `solver.StatusName()` (deprecated)
- Always check ortools version compatibility

### 5. File Locations
- Team data: `data/2025/teams/*.csv`
- Output schedules: `draws/`
- Checkpoints: `checkpoints/run_X/`
- Solver logs: `logs/`

## Quick Commands Reference

```powershell
# Standard generate (background)
.\.venv\Scripts\python.exe run.py generate --year 2025

# Low memory mode
.\.venv\Scripts\python.exe run.py generate --year 2025 --low-memory

# Resume from checkpoint
.\.venv\Scripts\python.exe run.py generate --year 2025 --resume run_13 stage1_required

# Test draw
.\.venv\Scripts\python.exe run.py test draws/draw.json

# Analyze draw
.\.venv\Scripts\python.exe run.py analyze draws/draw.json
```

## When User Asks About This Project

1. **First**: Read the skill file
2. **Then**: Understand the specific request
3. **Check**: Recent checkpoints and logs for context
4. **Recommend**: Appropriate memory/worker settings based on available resources
