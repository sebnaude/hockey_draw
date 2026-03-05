# AI Draw Initialization Checklist

> **Purpose:** Pre-season configuration verification before running the solver.  
> **Run this checklist BEFORE every new season's first solver run.**

---

## 🔑 Quick Start

```powershell
# Generate the pre-season report to verify all config
.\.venv\Scripts\python.exe run.py preseason --year 2026 --output reports/2026_season_config_loaded.txt
```

This report programmatically outputs ALL loaded configuration. Review it before running the solver.

---

## 📋 Pre-Season Checklist

### 1. Team Data Verification

| Check | Command/Location | What to Verify |
|-------|------------------|----------------|
| All club CSV files exist | `data/{year}/teams/*.csv` | One file per club (e.g., `maitland.csv`, `tigers.csv`) |
| CSV format correct | Each file | Columns: `Club`, `Grade`, `Team Name` |
| No duplicate teams | Pre-season report | Each team name unique within grade |
| Team counts correct per grade | Pre-season report | Match nominations from clubs |
| Odd team counts noted | Pre-season report | Odd counts = byes required (warning shown) |

**Team CSV Format:**
```csv
Club,Grade,Team Name
Maitland,PHL,Maitland PHL
Maitland,2nd,Maitland 2nd
```

**Team Naming Conventions:**
- Use `config/team_naming.py` for multi-team clubs
- Tigers: Tigers, Tigers Black
- Wests: Wests, Wests Red
- University: Uni, Uni Seapigs
- Colts: Colts Gold, Colts Green

---

### 2. Season Dates & Rounds Validation

| Check | Config Key | What to Verify |
|-------|------------|----------------|
| Start date correct | `SEASON_CONFIG['start_date']` | First Sunday of season |
| Last round date correct | `SEASON_CONFIG['last_round_date']` | Last regular round Sunday |
| Grand Final date correct | `SEASON_CONFIG['end_date']` | Grand Final Saturday |
| Max rounds calculation | `SEASON_CONFIG['max_rounds']` | Must equal available weekends |

**Critical Formula:**
```
Available Weekends = Total Sundays between start_date and last_round_date - Blocked Weekends
max_rounds MUST equal Available Weekends
```

The pre-season report validates this automatically:
```
✓ VALID: Sufficient weekends available for configured rounds
```

---

### 3. Field Unavailabilities (Hard Constraints)

| Check | Config Key | What to Verify |
|-------|------------|----------------|
| Blocked weekends listed | `FIELD_UNAVAILABILITIES[venue]['weekends']` | Easter, State Championships, special events |
| Whole days blocked | `FIELD_UNAVAILABILITIES[venue]['whole_days']` | ANZAC Day (Saturday), etc. |
| Part days if needed | `FIELD_UNAVAILABILITIES[venue]['part_days']` | Specific timeslots blocked |
| All venues have entries | All 3 venues | Maitland, NIHC, Gosford |

**Venues:**
- `Newcastle International Hockey Centre` (NIHC/Broadmeadow) - 3 fields: SF, EF, WF
- `Maitland Park` - 1 field
- `Central Coast Hockey Park` (Gosford) - 1 field

**Common Blocked Weekends:**
- Easter weekend
- State Championship weekends (Masters SC, U16 Girls SC, etc.)
- Other confirmed blocked dates

---

### 4. Game Time Dictionaries

#### 4.1 Standard Game Times (`DAY_TIME_MAP`)

| Venue | Day | Times | Notes |
|-------|-----|-------|-------|
| NIHC | Sunday | 08:30 - 19:00 (8 slots) | 90-min intervals |
| Maitland | Sunday | 09:00 - 16:30 (6 slots) | 90-min intervals |
| Gosford | Sunday | 12:00, 13:30 only | Restricted per club request |

**Why This Matters:**
- These define ALL possible game slots for ALL grades
- More slots = more solver variables = slower solve time
- Only add times that can actually be used

#### 4.2 PHL-Specific Game Times (`PHL_GAME_TIMES`)

**CRITICAL:** PHL can only play at certain times. The solver should NOT create variables for invalid PHL times.

| Venue | Day | Times | Notes |
|-------|-----|-------|-------|
| NIHC | Friday | 19:00 (7pm) | Friday night games |
| NIHC | Sunday | 11:30, 13:00, 14:30, 16:00 | Middle-of-day slots only |
| Gosford | Friday | 18:30, 20:00 | 6:30pm or 8pm (AGM confirmed 8pm) |
| Gosford | Sunday | 12:00, 13:30 | Same as standard (only 2 slots) |
| Maitland | Sunday | 12:00, 13:30, 15:00, 16:30 | Afternoon slots |

**Verification Questions:**
- [ ] Are PHL times a SUBSET of standard times (for same venue/day)?
- [ ] Do PHL Friday times match agreed times with clubs?
- [ ] Do Gosford times match the "12pm or 1:30pm ONLY" request?

#### 4.3 PHL State Championship Weekend Slots (`PHL_SC_WEEKEND_SLOTS`)

**Special Case:** PHL can play on blocked SC weekends (at "back end" = Sunday afternoon).

| SC Weekend | Date | Fields | Times |
|------------|------|--------|-------|
| Masters SC | May 17, 2026 | EF: 14:30, 16:00; WF: 14:30 | 3 slots total |
| U16 Girls SC | Jun 21, 2026 | EF: 14:30, 16:00; WF: 14:30 | 3 slots total |

**Why 3 slots?** Minimum needed to cover PHL games while keeping variables low.

---

### 5. Special Requests & Events

#### 5.1 Club Days (`CLUB_DAYS`)

Club days = all teams from one club play back-to-back on same field.

| Club | Date | Status |
|------|------|--------|
| Crusaders | 2026-06-14 | ✅ Configured |
| Wests | TBD | ❌ Date needed |
| Tigers | TBD | ❌ Date needed |
| University | TBD | ❌ Date needed |
| Port Stephens | TBD | ❌ Date needed |
| Colts | TBD | ❌ Date needed |

#### 5.2 No-Play Preferences (`PREFERENCE_NO_PLAY`)

Soft constraints - teams prefer not to play on certain dates.

| Club | Grade | Dates | Reason |
|------|-------|-------|--------|
| Crusaders | 6th | Apr 17-19 | NSW Masters Moorebank |
| Crusaders | 6th | Jun 26-28 | NSW Masters Tamworth |
| Souths | PHL, 2nd | May 24 | U18 State Championships |
| Gosford | All | Jun 21 | Recovery after Men's SC |

#### 5.3 Friday Night Allocations (`FRIDAY_NIGHT_CONFIG`)

| Setting | Value | Notes |
|---------|-------|-------|
| `gosford_friday_count` | 8 | Total Friday nights at Gosford |
| `friday_clubs` | Dict | Wests x2, Souths x2, Tigers x2, Norths x1, Maitland x1 |
| `friday_dates` | List | Confirmed dates (need 8 total) |
| `gosford_friday_times` | [20:00] | 8pm confirmed at AGM |
| `nihc_friday_times` | [19:00] | 7pm at NIHC |

#### 5.4 Special Games (`SPECIAL_GAMES`)

| Game | Teams | Grades | Date | Status |
|------|-------|--------|------|--------|
| Taree Game | Tigers vs Souths | PHL, 2nd | May (TBC) | Date needed |
| Red & Blue Derby | Souths vs Norths | All | August (TBC) | Date needed |

#### 5.5 Team Conflicts

Teams that CANNOT play at the same timeslot (e.g., shared players).

```python
TEAM_CONFLICTS = [
    # ('Team A', 'Team B'),  # Reason
]
```

---

### 6. PHL-Specific Configuration

| Setting | Value | What It Does |
|---------|-------|--------------|
| `phl_2nd_back_to_back` | True | PHL and 2nd grade from same club play consecutively |
| `gosford_2nd_grade_bye` | True | 2nd grade teams get bye when their PHL plays at Gosford |

---

### 7. Home Field Mappings

| Club | Home Venue |
|------|------------|
| Maitland | Maitland Park |
| Gosford | Central Coast Hockey Park |
| All others | Newcastle International Hockey Centre (default) |

---

### 8. Constraint Validation

#### 8.1 PHL Time Constraints

The `PHLAndSecondGradeTimes` constraint enforces:
- No concurrent PHL games at Broadmeadow (same timeslot)
- No concurrent PHL + 2nd grade from same club at Broadmeadow
- Max 3 Friday night games at Broadmeadow

**Verify:** PHL times in config match what the constraint expects.

#### 8.2 Gosford Constraint

Gosford has special rules:
- Only 1 game per week at Gosford (it's an away venue)
- Friday nights or Sunday 12pm/1:30pm only
- 8 Friday night games total per season

**Verify:** `DAY_TIME_MAP` and `PHL_GAME_TIMES` for Gosford match these rules.

---

## 🔧 Configuration Files Reference

### Primary Config File: `config/season_{year}.py`

| Section | Purpose |
|---------|---------|
| `FIELDS` | Playing field definitions (venue + field name) |
| `DAY_TIME_MAP` | Standard game times per venue per day |
| `PHL_GAME_TIMES` | PHL-specific times (subset of standard) |
| `PHL_SC_WEEKEND_SLOTS` | PHL slots on blocked SC weekends |
| `FIELD_UNAVAILABILITIES` | Blocked weekends/days/timeslots |
| `CLUB_DAYS` | Club day dates |
| `PREFERENCE_NO_PLAY` | Soft no-play preferences |
| `PHL_PREFERENCES` | PHL-specific settings |
| `FRIDAY_NIGHT_CONFIG` | Friday night game setup |
| `SPECIAL_GAMES` | Special matches (Taree, Derby, etc.) |
| `SEASON_CONFIG` | Main config dict (pulls all above together) |

### Data Files: `data/{year}/`

| Directory | Purpose |
|-----------|---------|
| `teams/` | One CSV per club with team nominations |
| `noplay/` | Club-specific no-play XLSX files |
| `field_availability/` | Additional field availability data |
| `documentation.txt` | Notes on naming conventions, etc. |

---

## ⚠️ Common Configuration Errors

### Error: "Insufficient weekends for configured rounds"
**Cause:** `max_rounds` > available weekends  
**Fix:** Update `max_rounds` or check `last_round_date` is correct

### Error: Solver creates too many variables
**Cause:** Game time dictionaries too broad  
**Fix:** Restrict times to only those actually needed (especially PHL times)

### Error: PHL games scheduled at invalid times
**Cause:** `PHL_GAME_TIMES` not properly configured  
**Fix:** Ensure PHL times are correct subset of constraint expectations

### Error: Gosford games at wrong times
**Cause:** `DAY_TIME_MAP['Central Coast Hockey Park']` has wrong times  
**Fix:** Gosford Sunday MUST be 12:00 and 13:30 only (per club request)

### Error: Friday nights not scheduled correctly
**Cause:** `FRIDAY_NIGHT_CONFIG` incomplete  
**Fix:** Ensure all dates, clubs, and times are set

---

## 📊 Pre-Season Report Sections

The `preseason` command generates a report with these sections:

1. **SEASON DATES** - Start, end, grand final dates
2. **ROUNDS CALCULATION** - Validates max_rounds matches available weekends
3. **TEAMS BY GRADE** - All teams organized by grade
4. **AVAILABLE GAME TIMES BY VENUE** - Both standard and PHL-specific
5. **FIELD UNAVAILABILITIES** - All blocked dates
6. **SPECIAL REQUESTS** - Club days, no-play, Friday nights, special games
7. **PHL CONFIGURATION** - Back-to-back, Gosford byes, SC weekend slots
8. **HOME FIELD MAPPINGS** - Which club plays where
9. **VALIDATION** - Errors and warnings

---

## ✅ Final Verification Steps

Before running the solver:

1. [ ] Run `preseason --year {year}` and review output
2. [ ] Verify team counts match club nominations
3. [ ] Verify rounds = available weekends (auto-checked in report)
4. [ ] Verify blocked weekends match emails/AGM decisions
5. [ ] Verify PHL times are correct (NIHC: 11:30-16:00, Gosford: 12:00/13:30)
6. [ ] Verify Friday night dates and club allocations
7. [ ] Check for any TBC items that need dates
8. [ ] Review warnings (odd team counts, etc.)

---

## 📁 Related Documents

| Document | Purpose |
|----------|---------|
| `reports/{year}_club_requests.md` | Tracks all club requests and implementation status |
| `reports/{year}_season_config_loaded.txt` | Generated pre-season report |
| `AI_CONSTRAINTS_AUDIT.md` | Constraint comparison (original vs AI) |
| `AI_CONSTRAINTS_GOALS.md` | AI constraint parity tracking |
| `.github/copilot-skills/hockey-draw-scheduler.md` | Copilot skill file with solver commands |
| `.github/copilot-instructions.md` | General Copilot instructions |

---

## 🚀 Ready to Generate

Once all checks pass:

```powershell
# Standard generate (run in background - can take hours!)
.\.venv\Scripts\python.exe run.py generate --year 2026

# With AI constraints
.\.venv\Scripts\python.exe run.py generate --year 2026 --ai

# Low memory mode (if machine has limited RAM)
.\.venv\Scripts\python.exe run.py generate --year 2026 --low-memory --workers 4
```

**Remember:** Always run with `isBackground: true` in Copilot - solver runs can take hours to days!
