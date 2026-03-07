# AI Draw Initialization Checklist

> **Purpose:** Pre-season configuration verification before running the solver.  
> **Run this checklist BEFORE every new season's first solver run.**

---

## 🔑 Pre-Season Protocol

### Step 1: Gather Club Requests
Collect emails from all clubs with their nominations and special requests.

### Step 2: Enter Configuration
Update `config/season_{year}.py` with all requests (teams, dates, constraints).

### Step 3: Generate Technical Report
```powershell
.\.venv\Scripts\python.exe run.py preseason --year 2026 --output reports/2026_season_config_loaded.txt
```
This outputs what the solver will actually use. Review for accuracy.

### Step 4: Update Club Requests Tracking
Maintain `reports/{year}_club_requests.md` - internal tracking of what's been asked vs implemented.

### Step 5: Generate Club Summary for Confirmation
Create/update `reports/{year}_club_requests_summary.md` - a clean, email-friendly summary.
Send to clubs: *"This is what we have recorded. Please confirm or advise of any changes."*

### Step 6: Run Solver
Once clubs confirm, run the solver.

---

## 📁 Pre-Season Documents

| Document | Purpose | Audience |
|----------|---------|----------|
| `reports/{year}_season_config_loaded.txt` | Technical config dump | Internal (developer) |
| `reports/{year}_club_requests.md` | Detailed tracking with implementation status | Internal (developer) |
| `reports/{year}_club_requests_summary.md` | Clean summary for confirmation | External (clubs) |

---

## 📋 Configuration Checklist

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

### 4. Game Variable Generation Dictionaries

**⚠️ CRITICAL CONCEPT:** These dictionaries control which **decision variables** the solver creates.
Fewer variables = faster solve. Only create variables for slots that can actually be used.

The filtering happens in `utils.py` → `generate_X()` function.

#### 4.1 Standard Game Times (`DAY_TIME_MAP`)

Controls timeslot generation for ALL grades (except PHL which has its own dict).

| Venue | Day | Times | Notes |
|-------|-----|-------|-------|
| NIHC | Sunday | 08:30 - 19:00 (8 slots) | All 3 fields (SF, EF, WF) |
| Maitland | Sunday | 09:00 - 16:30 (6 slots) | 1 field |
| Gosford | Sunday | 12:00, 13:30 only | 1 field, restricted per request |

#### 4.2 PHL Game Variable Generation Dict (`PHL_GAME_TIMES`)

**⚠️ THIS IS NOT A PREFERENCE - IT CONTROLS VARIABLE GENERATION**

This dict defines the ONLY slots where PHL game variables will be created.
Slots not in this dict = no PHL variable = impossible to schedule PHL there.

**Structure:** `{ venue: { field: { day: [times] } } }`

**Key Restrictions Built Into This Dict:**
- PHL **cannot play on South Field (SF)** - only EF and WF at NIHC
- Gosford: 1 slot per week max (away venue) - limits variable explosion
- Times restricted to specific windows (no early morning/late evening)

| Venue | Field | Day | Times |
|-------|-------|-----|-------|
| NIHC | EF | Friday | 19:00 |
| NIHC | EF | Sunday | 11:30, 13:00, 14:30, 16:00 |
| NIHC | WF | Friday | 19:00 |
| NIHC | WF | Sunday | 11:30, 13:00, 14:30, 16:00 |
| NIHC | SF | - | **EXCLUDED** (PHL cannot play here) |
| Gosford | Wyong Main | Friday | 20:00 (8pm only, AGM confirmed) |
| Gosford | Wyong Main | Sunday | 12:00, 13:30 |
| Maitland | Main | Sunday | 12:00, 13:00, 15:00, 16:30 |

**MANDATORY PHL SLOT VERIFICATION:**

Before running the solver, verify `PHL_GAME_TIMES` includes slots that meet the season's requirements:

1. **Friday Night Games** - Verify slots exist at required venues:
   - [ ] NIHC (Broadmeadow): Friday time configured? (typically 19:00)
   - [ ] Gosford: Friday time configured? (check AGM-confirmed time)
   
2. **Sunday Games** - Verify all PHL venues have Sunday slots:
   - [ ] NIHC: Sunday times include required windows?
   - [ ] Gosford: Sunday times match club agreement?
   - [ ] Maitland: Sunday times configured?

3. **Field Restrictions** - Verify field exclusions:
   - [ ] Are any fields excluded for PHL? (e.g., SF at NIHC)
   - [ ] Do excluded fields match the season's requirements?

4. **Slot Sufficiency** - Run this verification:
   ```powershell
   .\.venv\Scripts\python.exe -c "from config import load_season_data; d=load_season_data({year}); phl=[t for t in d['teams'] if t.grade=='PHL']; print(f'PHL teams: {len(phl)}'); print(f'PHL games per team: {len(phl)-1}'); print(f'Total PHL matchups: {len(phl)*(len(phl)-1)//2}')"
   ```
   Then verify enough PHL slots exist across all weekends.

**Note:** These requirements change year-on-year based on club agreements. 
Review AGM minutes and club emails to determine current season's PHL slot requirements.

#### 4.3 2nd Grade Game Variable Generation Dict (`SECOND_GRADE_TIMES`)

**NEW IN 2026:** 2nd grade also has restricted variable generation.

**Key Rules:**
- Only venues/fields listed get 2nd grade variables
- SF not listed at NIHC (only EF and WF)
- Gosford not listed (PHL-only venue)
- Times: PHL slots PLUS one slot before/after (where available in `DAY_TIME_MAP`)

**IMPORTANT:** Cannot create NEW timeslots - only existing `DAY_TIME_MAP` slots.

**Structure:** `{ venue: { field: { day: [times] } } }`

| Venue | Field | Day | Times | Notes |
|-------|-------|-----|-------|-------|
| NIHC | EF | Sunday | 10:00, 11:30, 13:00, 14:30, 16:00, 17:30 | PHL + adjacent |
| NIHC | WF | Sunday | 10:00, 11:30, 13:00, 14:30, 16:00, 17:30 | PHL + adjacent |
| Maitland | Main | Sunday | 10:30, 12:00, 13:00, 15:00, 16:30 | PHL + 10:30 before |

*SF and Gosford not listed = no 2nd grade variables created there.*

#### 4.4 How Variable Filtering Works

In `utils.py` → `generate_X()`:

```python
# For PHL games, only create variables for valid timeslots
if is_phl:
    slot_key = (t.field.location, t.field.name, t.day, t.time)
    if slot_key not in phl_valid_slots:
        continue  # Skip - no variable created

# For 2nd grade games, filter if second_grade_times is defined
elif is_second and second_valid_slots:
    slot_key = (t.field.location, t.field.name, t.day, t.time)
    if slot_key not in second_valid_slots:
        continue  # Skip - no variable created
```

This means:
- **PHL games:** Only slots in `PHL_GAME_TIMES` get variables
- **2nd grade games:** Only slots in `SECOND_GRADE_TIMES` get variables (if defined)
- **Other grades (3rd-6th):** Most slots in `DAY_TIME_MAP` get variables, **except PHL-only venues/days**

**Impact:** Reduces PHL variables by ~60%, 2nd grade by ~50%, lower grades by ~15% compared to all possible combinations.

#### 4.5 Lower Grades Exclusion (3rd-6th)

**HARDCODED in `generate_X()` - these are automatic exclusions:**

| Exclusion | Reason | Impact |
|-----------|--------|--------|
| **Gosford (Central Coast Hockey Park)** | PHL-only venue - only Gosford PHL plays there | ~10,000+ vars saved |
| **Friday nights** | PHL-only timeslot - all lower grades play Sunday only | ~14,000+ vars saved |

Lower grades (3rd-6th) **cannot** and **should not** play at:
- Gosford (Central Coast Hockey Park) - this is an away venue for PHL only
- Any Friday timeslot - Friday nights are PHL-exclusive

These exclusions are enforced in `utils.py` → `generate_X()`:
```python
# Build set of PHL-only venues and days
phl_only_venues = {'Central Coast Hockey Park'}  # Gosford is PHL-only
phl_only_days = {'Friday'}  # Friday nights are PHL-only

# Lower grades (3rd-6th): Exclude PHL-only venues and days
if t.field.location in phl_only_venues:
    continue  # Skip - no variable created
if t.day in phl_only_days:
    continue  # Skip - no variable created
```

**DO NOT configure Gosford or Friday slots for non-PHL grades** - they will be ignored anyway.

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

### 6. PHL & 2nd Grade Configuration

**Variable filtering** (enforced in `generate_X()`, not as constraints):

| Grade | Rule | Implementation |
|-------|------|----------------|
| PHL | Cannot play on SF at NIHC | Excluded from `PHL_GAME_TIMES` |
| PHL | Restricted time windows | Only slots in `PHL_GAME_TIMES` |
| 2nd | Cannot play on SF at NIHC | Excluded from `SECOND_GRADE_TIMES` |
| 2nd | Cannot play at Gosford | Gosford excluded from `SECOND_GRADE_TIMES` |
| 2nd | Adjacent to PHL times | PHL slots + 1 before + 1 after |

**Constraint behaviors** (enforced by `PHLAndSecondGradeTimes` constraint):

| Rule | Description |
|------|-------------|
| No concurrent PHL at NIHC | Only 1 PHL game per timeslot at Broadmeadow |
| No PHL + 2nd overlap (same club) | Club's 2nd grade not at same time as club's PHL |
| Max 3 Friday night PHL | Maximum 3 PHL games on Fridays at Broadmeadow |

**Note:** The old config keys `phl_2nd_back_to_back` and `gosford_2nd_grade_bye` are NOT supported in `PHL_PREFERENCES`. 
These rules are implemented via variable filtering (no 2nd grade at Gosford) and constraints (back-to-back scheduling).

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
- **Max 3 Friday night games at Broadmeadow** (NIHC)
- **Exactly 8 Friday night games at Gosford** (CCHP) - AGM decision 2026

**Verify:** PHL times in config match what the constraint expects.

**Testing:** The `PHLTimingValidator` in `tests/test_draw_outcomes.py` validates both Friday limits.

#### 8.2 Gosford Constraint

Gosford has special rules:
- Only 1 game per week at Gosford (it's an away venue)
- Friday nights (8pm) or Sunday 12pm/1:30pm only
- **Exactly 8 Friday night games total per season** (AGM decision 2026)
- Only PHL plays at Gosford (Central Coast Hockey Park)

**Verify:** `DAY_TIME_MAP` and `PHL_GAME_TIMES` for Gosford match these rules.

**Config:** See `FRIDAY_NIGHT_CONFIG` in `config/season_{year}.py` for:
- `gosford_friday_count`: 8 (exact count enforced by constraint)
- `friday_clubs`: Which clubs play at Gosford on Friday
- `gosford_friday_times`: [20:00] (8pm confirmed at AGM)

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
