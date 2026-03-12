# 2026 Season - Club Requests Summary

**Last Updated:** 2026-03-10

This document tracks all requests received from clubs, their implementation status, and **HOW each was implemented**.

> **⚠️ AI MANDATE:** When implementing ANY request, document the implementation method here so future AI sessions know exactly how it was done.

---

## Implementation Methods Reference

| Method | Description | Example |
|--------|-------------|---------|
| **Config Value** | Set in `config/season_2026.py` | `SEASON_CONFIG['max_rounds'] = 20` |
| **Variable Filtering** | Remove variables at generation time | `PHL_GAME_TIMES` dict excludes SF field |
| **Matchup Filtering** | Restrict specific matchups to dates | `nihc_friday_games` dict in `FRIDAY_NIGHT_CONFIG` |
| **Hard Constraint** | Solver must satisfy | `model.Add(sum(vars) == 8)` in constraint |
| **Soft Constraint** | Penalty if violated | `PREFERENCE_NO_PLAY` with penalty weight |
| **Field Unavailability** | Block venue/date entirely | `FIELD_UNAVAILABILITIES` dict |
| **Part Day Unavailability** | Block specific timeslots on a date | `FIELD_UNAVAILABILITIES['part_days']` list |
| **No-Play File** | Team-specific exclusion | `data/{year}/noplay/{club}_noplay.xlsx` |

---

## ✅ IMPLEMENTED

### PHL / Premier League

| Request | Club | Implementation Method | Implementation Location | Notes |
|---------|------|----------------------|------------------------|-------|
| Season remains 4 rounds (20 matches) | HCPL | **Config Value** | `SEASON_CONFIG['max_rounds'] = 20` | Base value; PHL gets 22 via override |
| PHL plays 22 weekends (20 + 2 rescued) | HCPL | **Config Value** | `MAX_WEEKENDS_PER_GRADE['PHL'] = 22` | Friday nights rescue blocked Sundays |
| Start date Sunday 22nd March | HCPL | **Config Value** | `SEASON_CONFIG['start_date']` | First playing Sunday |
| Grand Final Saturday 19th September | HCPL | **Config Value** | `SEASON_CONFIG['end_date']` | After last round |
| Playing ANZAC Sunday (confirmed at AGM) | HCPL | **Config Value** | `SEASON_CONFIG['play_anzac_sunday'] = True` | ANZAC Saturday blocked, Sunday plays |
| PHL & 2nd grade back-to-back | HCPL | **Variable Filtering + Constraint** | `SECOND_GRADE_TIMES` includes PHL times ± 1 slot; `PHLAndSecondGradeAdjacency` constraint | 2nd grade slots overlap PHL for adjacency |
| Teams vs Gosford have 2nd grade bye | HCPL | **Variable Filtering** | `SECOND_GRADE_TIMES` excludes Gosford entirely | No 2nd grade variables at CCHP |
| PHL excluded from South Field (SF) | HCPL | **Variable Filtering** | `PHL_GAME_TIMES` only lists EF and WF | No PHL variables on SF |
| Gosford exactly 8 home Friday nights | Gosford | **Hard Constraint** | `PHLAndSecondGradeTimes` → `model.Add(sum(friday_gosford_vars) == 8)` | AGM decision enforced in constraint |
| Gosford Friday times: 8:00pm only | Gosford | **Variable Filtering** | `PHL_GAME_TIMES['Central Coast Hockey Park']['Friday'] = [tm(20, 0)]` | Only 8pm slot available |
| Gosford Sunday times: 12pm or 1:30pm only | Gosford | **Variable Filtering** | `PHL_GAME_TIMES['Central Coast Hockey Park']['Sunday'] = [tm(12, 0), tm(13, 30)]` | Restricted to 2 slots |
| NIHC Friday night max 3 games | HCPL | **Hard Constraint** | `PHLAndSecondGradeTimes` → `model.Add(sum(friday_broadmeadow_vars) <= 3)` | Operational limit |
| NIHC Friday time: 7:00pm | HCPL | **Variable Filtering** | `PHL_GAME_TIMES['NIHC']['EF/WF']['Friday'] = [tm(19, 0)]` | Junior Boys program alignment |
| Norths Friday night June 12 (80th anniversary) | Norths | **Config Value** | `FRIDAY_NIGHT_CONFIG['friday_dates']` includes `datetime(2026, 6, 12)` | Confirmed date |
| NIHC Friday games: specific matchups only | HCPL | **Matchup Filtering** | `FRIDAY_NIGHT_CONFIG['nihc_friday_games']` dict + filtering in `generate_X()` | Only allowed matchups get variables |
| NIHC Friday: May 8 = Souths vs Maitland | HCPL | **Matchup Filtering** | `nihc_friday_games['2026-05-08'] = [('Maitland', 'Souths')]` | Locked matchup |
| NIHC Friday: Jun 19 = Tigers vs Wests | HCPL | **Matchup Filtering** | `nihc_friday_games['2026-06-19'] = [('Tigers', 'Wests')]` | State champ weekend, evening available |
| NIHC Friday: Jul 24 = Norths vs TBC | HCPL | **Matchup Filtering** | `nihc_friday_games['2026-07-24'] = 'norths_only'` | Any opponent OK |
| Friday night clubs: Wests x2, Tigers x2, Souths x2, Norths x1, Maitland x1 | Various | **Config Value** | `FRIDAY_NIGHT_CONFIG['friday_clubs']` dict | Club allocations tracked |
| Souths no PHL/2nd on May 24 (U18 SC) | Souths | **Soft Constraint** | `PREFERENCE_NO_PLAY['Souths_U18_SC']` | Penalty-based avoidance |
| Gosford no match weekend after Men's SC (Jun 21) | Gosford | **Soft Constraint** | `PREFERENCE_NO_PLAY['Gosford_Post_SC']` | Penalty-based avoidance |

### Field Unavailabilities

| Request | Club | Implementation Method | Implementation Location | Notes |
|---------|------|----------------------|------------------------|-------|
| Easter weekend blocked (Apr 3-5) | HCPL | **Field Unavailability** | `FIELD_UNAVAILABILITIES['NIHC']['weekends']` | All venues blocked |
| Masters SC Friday/Sat blocked (May 15-16) | HCPL | **Part Day Unavailability** | `FIELD_UNAVAILABILITIES['NIHC']['whole_days']` (Sat) + `part_days` (Fri daytime) | Friday 7pm available for PHL |
| Jun 5-7 weekend blocked | HCPL | **Field Unavailability** | `FIELD_UNAVAILABILITIES['NIHC']['weekends']` | NIHC blocked |
| U16 Girls SC Friday/Sat blocked (Jun 19-20) | HCPL | **Part Day Unavailability** | `FIELD_UNAVAILABILITIES['NIHC']['whole_days']` (Sat) + `part_days` (Fri daytime) | Friday 7pm available for PHL |
| ANZAC Day Saturday blocked | HCPL | **Field Unavailability** | `FIELD_UNAVAILABILITIES['NIHC']['whole_days']` | Saturday only, Sunday plays |
| State Championship Sundays: morning blocked | HCPL | **Part Day Unavailability** | `FIELD_UNAVAILABILITIES['NIHC']['part_days']` | 8:30, 10:00, 11:30 blocked; afternoon open for PHL |

### Club Days

| Request | Club | Implementation Method | Implementation Location | Notes |
|---------|------|----------------------|------------------------|-------|
| Crusaders Club Day - June 14 | Crusaders | **Config + Constraint** | `CLUB_DAYS['Crusaders'] = datetime(2026, 6, 14)` + `ClubDayConstraint` | All teams same field, consecutive |

### No-Play Preferences (Soft Constraints)

| Request | Club | Implementation Method | Implementation Location | Notes |
|---------|------|----------------------|------------------------|-------|
| Crusaders 6th - NSW Masters Moorebank (Apr 17-19) | Crusaders | **Soft Constraint** | `PREFERENCE_NO_PLAY['Crusaders_6th_Masters_Moorebank']` | 2026 format: `{'club': 'Crusaders', 'grade': '6th', 'dates': [...]}` |
| Crusaders 6th - NSW Masters Tamworth (Jun 26-28) | Crusaders | **Soft Constraint** | `PREFERENCE_NO_PLAY['Crusaders_6th_Masters_Tamworth']` | 2026 format: `{'club': 'Crusaders', 'grade': '6th', 'dates': [...]}` |

---

## ⚠️ PARTIALLY IMPLEMENTED / NEEDS CLARIFICATION

| Request | Club | Issue | Action Required |
|---------|------|-------|-----------------|
| PHL at back end of State Championship weekends | HCPL | Currently blocking entire weekend. Email says can schedule PHL Sunday afternoon during SC | Decide: unblock Sunday PM for PHL only? |
| Maitland away games to Gosford on Friday night | Maitland/Gosford | Maitland confirmed 1 Friday night match but not specified if this is their away vs Gosford | Confirm with Maitland |
| NIHC Friday night dates with Junior Boys program | HCPL | PHL times set but specific dates not locked in | Get dates from HCPL |
| NIHC events - Gosford gets home game to avoid late slots | Gosford | No mechanism to implement this preference | Consider soft constraint |

---

## ❌ NOT YET IMPLEMENTED

### Special Games

| Request | Club | Details | Reason Not Implemented |
|---------|------|---------|------------------------|
| Tigers vs Souths in Taree (May) | Tigers/Souths | PHL & 2nd grade match at different venue | Date TBC - need specific date |

### Club Days (Pending Dates)

| Request | Club | Details | Reason Not Implemented |
|---------|------|---------|------------------------|
| Wests Club Day | Wests | Need 2026 date | No email received |
| University Club Day | University | Need 2026 date | No email received |
| Tigers Club Day | Tigers | Need 2026 date | No email received |
| Port Stephens Club Day | Port Stephens | Need 2026 date | No email received |
| Colts Club Day | Colts | Any date - string all games together | Need to allocate date |

### Special Events (Pending)

| Request | Club | Details | Reason Not Implemented |
|---------|------|---------|------------------------|
| Red & Blue Derby | Souths/Norths | August date TBC | Need date from clubs |
| Masters State Training Weekend | HCPL | August (Sat/Sun finishing 12pm Sun) | No date set yet |
| August catch-up weekend | HCPL | Reserve for wet weather deferrals | Need to designate weekend |

### Scheduling Preferences (Soft - Best Effort)

| Request | Club | Details | Priority |
|---------|------|---------|----------|
| Colts: Align 5th/6th grade byes with Masters weekends (Illawarra Apr 17-19, Tamworth Jun 26-28) | Colts | If byes available, give to Colts 5th/6th on these weekends | Low - "if possible" |
| Colts: Avoid doubling up 5th Gold and 6th in August (World Cup players) | Colts | Reduce impact of players away at World Cup | Low - "if possible" |
| Colts: Don't schedule two 5th grades at same time (except vs each other) | Colts | Colts Gold 5th vs Colts Green 5th only time they overlap | Medium |
| Colts: Avoid overlapping 4th and 5th, or 5th and 6th | Colts | Spread Colts games across day | Medium |
| Colts: Overlapping 4th and 6th would be excellent | Colts | Allow if other constraints met | Low - nice to have |

---

## 📋 CLUBS WITHOUT DIRECT EMAIL

The following clubs have not submitted direct requests. Team nominations taken from screenshot only:

- **Wests** - No email received
- **University** - No email received  
- **Port Stephens** - No email received

---

## 📅 CONFIRMED FRIDAY NIGHT DATES

| Date | Notes |
|------|-------|
| March 27, 2026 | Confirmed |
| April 17, 2026 | Confirmed (also Masters weekend at Moorebank) |
| April 24, 2026 | Confirmed (ANZAC long weekend Friday) |
| May 29, 2026 | Confirmed |
| June 12, 2026 | Confirmed - Norths 80th Anniversary |
| TBC x3 | Need 3 more dates for 8 total |

---

## 📝 RAW EMAIL EXCERPTS

### Gosford (via HCPL)
> Is it possible for the Gosford game on the Anzac Day long weekend to be played as a home game on a Friday night?
> Can the Gosford Sunday games in Newcastle and Maitland please be scheduled for either 12:00 noon or 1:30pm?
> When NIHC host championship events can Gosford be allocated a home game to avoid late timeslots?

### HCPL General
> Season to remain as 4 rounds (20 matches)
> We are able to schedule PHL matches at the back end of State Championships
> Can confirm we are playing ANZAC weekend (Sunday)
> We can hold over a weekend in August to play any deferred matches

### Tigers
> At this stage mate we are now looking at putting 1 x team in every grade and 2 x teams into 6th grade this year.

### Colts
> We do have a lot of master players (nearly all of us). This list does include the masters weekends and likely players at tournaments, but we think we can manage those weekends rather than doubling up. But if a bye is required for those grades then aligning those bye with the Illawarra and Tamworth weekends would be appreciated.
> We also have some going to the world cup, so avoiding doubling up during August for 5th Gold and 6th grade will also lessen the impact of those players being away.
> We would be keen for a club day any time you can string all our games together.
> We would prefer not to have our 2 5th grades at the same time except when playing each other, and avoid overlapping 4th and 5th or 5th and 6th, but overlapping 4th and 6th would be excellent.

---

## 🔧 TECHNICAL IMPLEMENTATION CHANGES

### 2026-03-11: Runtime Constraint Slack (--slack N)

**Purpose:** Allow runtime relaxation of specific constraint limits when solver returns INFEASIBLE.

**What Changed:**
- Added `--slack N` CLI argument to `run.py`
- Three constraints now support configurable limits:

| Constraint | Base Limit | With --slack N |
|------------|------------|----------------|
| `EqualMatchUpSpacingConstraint` | ±1 round | ±(1+N) rounds |
| `AwayAtMaitlandGrouping` | Max 3 away clubs per Maitland weekend | Max 3+N |
| `MaitlandHomeGrouping` | No back-to-back home weekends | N back-to-back pairs allowed |

**Files Modified:**
- `run.py` - Added CLI argument and constraint_slack dict builder
- `utils.py` - Pass constraint_slack through build_season_data()
- `main_staged.py` - Accept constraint_slack parameter
- `constraints/original.py` - All 3 constraints read from data['constraint_slack']
- `constraints/ai.py` - AI versions also read from constraint_slack

**Usage:**
```powershell
# Standard run (base limits)
.\\.venv\\Scripts\\python.exe run.py generate --year 2026

# With relaxed constraints (+1 to all limits)
.\\.venv\\Scripts\\python.exe run.py generate --year 2026 --slack 1
```

**When to Use:**
- If solver returns INFEASIBLE and `--relax` doesn't help
- For one-off relaxation when constraints are overconstrained

**Documentation Updated:**
- `docs/ai/CONFIGURATION_REFERENCE.md` - New section on constraint_slack
- `docs/ai/CONSTRAINT_APPLICATION.md` - New section on runtime relaxation
- `docs/ai/SYSTEM_OPERATION.md` - Added --slack to command list
