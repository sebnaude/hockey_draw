# AI Documentation Index

> **Purpose:** This folder contains all documentation that AI assistants (GitHub Copilot, Claude, etc.) need to effectively operate this system.
>
> **⚠️ IMPERATIVE:** When you learn something new about this system, **update the relevant document** so you know it next time. If a user tells you something important about how the system works, configuration requirements, or business rules, add it to the appropriate file here.

---

## Document Index

| Document | Purpose | When to Read |
|----------|---------|--------------|
| `SEASON_SETUP.md` | **Pre-season configuration checklist** | Start of every season |
| `CONFIGURATION_REFERENCE.md` | All configurable parameters and their effects | When updating config |
| `CONSTRAINT_APPLICATION.md` | How to apply different types of constraints | When adding restrictions |
| `GAME_TIME_DICTIONARIES.md` | PHL/2nd grade variable filtering system | When modifying game times |
| `SYSTEM_OPERATION.md` | How to run the solver and interpret results | When generating draws |

---

## Quick Reference: Common Tasks

### Starting a New Season
1. Read `SEASON_SETUP.md` completely
2. Gather required information from clubs
3. Update `config/season_{year}.py`
4. Run pre-season report to verify configuration

### Adding a No-Play Request
1. Read `CONSTRAINT_APPLICATION.md` - "No-Play Requests" section
2. Decide: Soft constraint (PREFERENCE_NO_PLAY) vs Hard constraint (noplay XLSX)
3. Apply the appropriate method
4. Document in season reports

### Modifying PHL/2nd Grade Times
1. Read `GAME_TIME_DICTIONARIES.md`
2. Understand which dict controls which variables
3. Make changes to `PHL_GAME_TIMES` or `SECOND_GRADE_TIMES`
4. Verify with pre-season report

### Running the Solver
1. Read `SYSTEM_OPERATION.md`
2. Use `--year` flag (required)
3. Use `isBackground: true` for terminal commands
4. Monitor via logs or `get_terminal_output`

---

## Document Update Protocol

When updating these documents:
1. Add a dated entry if making significant changes
2. Keep examples current with the latest season
3. Cross-reference related documents
4. Test any code snippets before adding them
