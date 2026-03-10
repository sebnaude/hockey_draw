# AI Documentation Index

> **Purpose:** This folder contains all documentation that AI assistants (GitHub Copilot, Claude, etc.) need to effectively operate this system.
>
> **⚠️ IMPERATIVE:** When you learn something new about this system, **update the relevant document** so you know it next time. If a user tells you something important about how the system works, configuration requirements, or business rules, add it to the appropriate file here.

---

## Document Index

| Document | Purpose | When to Read | Update When |
|----------|---------|--------------|-------------|
| `SEASON_SETUP.md` | Pre-season configuration checklist | Start of every season | Changing setup process |
| `CONFIGURATION_REFERENCE.md` | All configurable parameters and their effects | When updating config | Adding/changing config keys |
| `CONSTRAINT_APPLICATION.md` | How to apply different types of constraints | When adding restrictions | Adding new constraint methods |
| `GAME_TIME_DICTIONARIES.md` | PHL/2nd grade variable filtering system | When modifying game times | Changing variable filtering logic |
| `SYSTEM_OPERATION.md` | How to run the solver and interpret results | When generating draws | Changing CLI or solver |

---

## Season-Specific Documents

Located in `seasons/{year}/`:

| Document | Purpose | Update When |
|----------|---------|-------------|
| `{year}_club_requests.md` | **Master implementation tracking** - all requests + HOW implemented | EVERY config/constraint change |
| `{year}_club_requests_summary.md` | Club-facing summary (no implementation details) | Sharing updates with clubs |
| `constraint_comparison_{year}.md` | AI vs original constraint comparison | Testing constraint equivalence |

---

## ⚠️ CRITICAL: Update Protocol

### When Implementing ANY Change:

1. **Identify the change type:**
   - Config value? → Update `CONFIGURATION_REFERENCE.md`
   - New constraint? → Update `CONSTRAINT_APPLICATION.md`
   - Variable filtering? → Update `GAME_TIME_DICTIONARIES.md`
   - Command/flag change? → Update `SYSTEM_OPERATION.md`

2. **Document HOW in season report:**
   - Add entry to `seasons/{year}/{year}_club_requests.md`
   - Include: Request, Implementation Method, Implementation Location, Notes

3. **Update this index if adding new documents**

### Implementation Methods (Use These Terms)

| Method | When to Use |
|--------|-------------|
| **Config Value** | Setting a value in `config/season_{year}.py` |
| **Variable Filtering** | Modifying `PHL_GAME_TIMES`, `SECOND_GRADE_TIMES`, or `DAY_TIME_MAP` |
| **Hard Constraint** | Adding `model.Add()` in constraint code |
| **Soft Constraint** | Adding to `PREFERENCE_NO_PLAY` with penalty |
| **Field Unavailability** | Adding to `FIELD_UNAVAILABILITIES` |
| **No-Play File** | Adding team/dates to `data/{year}/noplay/` XLSX |

---

## Quick Reference: Common Tasks

### Starting a New Season
1. Read `SEASON_SETUP.md` completely
2. Gather required information from clubs
3. Update `config/season_{year}.py`
4. Run pre-season report to verify configuration
5. Create `seasons/{year}/{year}_club_requests.md` to track implementations

### Adding a No-Play Request
1. Read `CONSTRAINT_APPLICATION.md` - "No-Play Requests" section
2. Decide: Soft constraint (PREFERENCE_NO_PLAY) vs Hard constraint (noplay XLSX)
3. Apply the appropriate method
4. **Document in `seasons/{year}/{year}_club_requests.md`** with implementation method

### Modifying PHL/2nd Grade Times
1. Read `GAME_TIME_DICTIONARIES.md`
2. Understand which dict controls which variables
3. Make changes to `PHL_GAME_TIMES` or `SECOND_GRADE_TIMES`
4. **Document in `seasons/{year}/{year}_club_requests.md`** what was changed and why
5. Verify with pre-season report

### Running the Solver
1. Read `SYSTEM_OPERATION.md`
2. Use `--year` flag (required)
3. Use `isBackground: true` for terminal commands
4. Monitor via logs or `get_terminal_output`

---

## Document Update Log

| Date | Document | Change |
|------|----------|--------|
| 2026-03-10 | All docs | Added implementation tracking requirement |
| 2026-03-10 | README.md | Added update protocol and implementation methods |
