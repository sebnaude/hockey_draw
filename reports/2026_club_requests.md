# 2026 Season - Club Requests Summary

**Last Updated:** 2026-03-05

This document tracks all requests received from clubs and their implementation status.

---

## ✅ IMPLEMENTED

### PHL / Premier League

| Request | Club | Status | Config Location |
|---------|------|--------|-----------------|
| Season remains 4 rounds (20 matches) | HCPL | ✅ Done | `SEASON_CONFIG['max_rounds']` |
| Start date Sunday 22nd March | HCPL | ✅ Done | `SEASON_CONFIG['start_date']` |
| Grand Final Saturday 19th September | HCPL | ✅ Done | `SEASON_CONFIG['end_date']` |
| Playing ANZAC Sunday (confirmed at AGM) | HCPL | ✅ Done | `SEASON_CONFIG['play_anzac_sunday']` |
| PHL & 2nd grade back-to-back | HCPL | ✅ Done | `PHL_PREFERENCES['phl_2nd_back_to_back']` |
| Teams vs Gosford have 2nd grade bye | HCPL | ✅ Done | `PHL_PREFERENCES['gosford_2nd_grade_bye']` |
| Gosford 8 home Friday nights | Gosford | ✅ Done | `FRIDAY_NIGHT_CONFIG['gosford_friday_count']` |
| Gosford Friday times: 6:30pm or 8:00pm | Gosford | ✅ Done | `PHL_GAME_TIMES` |
| Gosford Sunday times: 12pm or 1:30pm only | Gosford | ✅ Done | `DAY_TIME_MAP` |
| Norths Friday night June 12 (80th anniversary) | Norths | ✅ Done | `FRIDAY_NIGHT_CONFIG['friday_dates']` |
| Friday night clubs: Wests x2, Tigers x2, Souths x2, Norths x1, Maitland x1 | Various | ✅ Done | `FRIDAY_NIGHT_CONFIG['friday_clubs']` |
| Souths no PHL/2nd on May 24 (U18 SC) | Souths | ✅ Done | `PREFERENCE_NO_PLAY['Souths_U18_SC']` |
| Gosford no match weekend after Men's SC (Jun 21) | Gosford | ✅ Done | `PREFERENCE_NO_PLAY['Gosford_Post_SC']` |

### Field Unavailabilities

| Request | Club | Status | Config Location |
|---------|------|--------|-----------------|
| Easter weekend blocked (Apr 3-5) | HCPL | ✅ Done | `FIELD_UNAVAILABILITIES` |
| Masters SC Newcastle blocked (May 15-17) | HCPL | ✅ Done | `FIELD_UNAVAILABILITIES` |
| Jun 5-7 weekend blocked | HCPL | ✅ Done | `FIELD_UNAVAILABILITIES` |
| Girls U16 SC Newcastle blocked (Jun 19-21) | HCPL | ✅ Done | `FIELD_UNAVAILABILITIES` |
| ANZAC Day Saturday blocked | HCPL | ✅ Done | `FIELD_UNAVAILABILITIES` |

### Club Days

| Request | Club | Status | Config Location |
|---------|------|--------|-----------------|
| Crusaders Club Day - June 14 | Crusaders | ✅ Done | `CLUB_DAYS['Crusaders']` |

### No-Play Preferences (Soft Constraints)

| Request | Club | Status | Config Location |
|---------|------|--------|-----------------|
| Crusaders 6th - NSW Masters Moorebank (Apr 17-19) | Crusaders | ✅ Done | `PREFERENCE_NO_PLAY` |
| Crusaders 6th - NSW Masters Tamworth (Jun 26-28) | Crusaders | ✅ Done | `PREFERENCE_NO_PLAY` |

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
