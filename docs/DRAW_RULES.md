# Draw Rules Documentation

This document outlines all the scheduling rules enforced by the constraint system. Rules are categorized as **Hard Constraints** (must be satisfied) or **Soft Constraints** (optimized with penalties).

---

## Table of Contents

1. [Core Scheduling Rules](#core-scheduling-rules)
2. [Hard Constraints](#hard-constraints)
3. [Soft Constraints](#soft-constraints)
4. [Implied Rules](#implied-rules)
5. [Constraint Interactions](#constraint-interactions)

---

## Core Scheduling Rules

These fundamental rules ensure the draw is valid and playable.

### Rule 1: No Double-Booking Teams
**Constraint:** `NoDoubleBookingTeamsConstraint`

**Description:** A team can only play one game per week.

**Enforcement:** For each (week, team) combination, the sum of all game variables must be ≤ 1.

**Rationale:** Teams cannot physically play multiple games simultaneously, and typically play once per week.

---

### Rule 2: No Double-Booking Fields
**Constraint:** `NoDoubleBookingFieldsConstraint`

**Description:** A field can only host one game per timeslot.

**Enforcement:** For each (day, day_slot, week, field_name) combination, the sum of game variables must be ≤ 1.

**Rationale:** Only one game can occur on a physical field at any given time.

---

### Rule 3: Balanced Games and Matchups
**Constraint:** `EnsureEqualGamesAndBalanceMatchUps`

**Description:** 
- Each team plays exactly `num_rounds[grade]` games per season
- Each pair of teams meets each other a balanced number of times (base or base+1)

**Enforcement:**
- Sum of game variables per team equals the required number of games
- For each pair, meetings are bounded: `base ≤ meetings ≤ base + 1`

**Calculation:**
- If even teams: `base = R // (T - 1)`
- If odd teams: `base = R // T`
- Extra matchups are distributed evenly

**Rationale:** Ensures competitive fairness with equal playing opportunities.

---

## Hard Constraints

### Rule 4: PHL and 2nd Grade Adjacency
**Constraint:** `PHLAndSecondGradeAdjacency`

**Description:** PHL and 2nd grade teams from the same club cannot play in adjacent timeslots at different locations.

**Enforcement:** For each club, if PHL plays at location A at time T, 2nd grade cannot play at location B within ±2 hours.

**Rationale:** Allows players/supporters to attend both games without impossible travel between venues.

---

### Rule 5: PHL and 2nd Grade Time Separation
**Constraint:** `PHLAndSecondGradeTimes`

**Description:** 
- PHL games cannot occur simultaneously at Newcastle International Hockey Centre (Broadmeadow)
- 2nd grade and PHL from the same club cannot occur at the same timeslot at Broadmeadow
- Maximum 3 Friday night PHL games at Broadmeadow
- Exactly 8 Friday night PHL games at Gosford (Central Coast Hockey Park) - AGM decision 2026

**Enforcement:** Uses timeslot indicators and sum constraints per location/club combination.

**Rationale:** Prevents spectator/player conflicts and ensures fair distribution of prime-time slots. The Gosford Friday requirement was confirmed at the 2026 AGM.

---

### Rule 6: 50/50 Home and Away Balance
**Constraint:** `FiftyFiftyHomeandAway`

**Description:** Teams from away venues (Maitland, Gosford) should play approximately 50% of their games at home and 50% away.

**Enforcement:** 
```
home_games * 2 >= total_games - 1
home_games * 2 <= total_games + 1
```

**Rationale:** Ensures fairness for teams that must travel significant distances.

---

### Rule 7: Team Conflict Prevention
**Constraint:** `TeamConflictConstraint`

**Description:** User-specified team pairs cannot play at the same timeslot.

**Enforcement:** For each conflicting pair, the sum of their simultaneous games must be ≤ 1.

**Rationale:** Accommodates shared resources (coaches, facilities, transport).

---

### Rule 8: Maximum Maitland Home Weekends
**Constraint:** `MaxMaitlandHomeWeekends`

**Description:** Limits the number of weekends where games are played at Maitland Park.

**Enforcement:** Uses indicator variables to track weekends with games, bounded by `num_games // 2 + 1`.

**Rationale:** Balances venue usage and operational costs for remote venues.

---

### Rule 9: Optimal Timeslot Usage
**Constraint:** `EnsureBestTimeslotChoices`

**Description:** 
- No unused timeslots between two used timeslots on a day
- Games should fill timeslots contiguously from earliest slots
- Location-specific minimum timeslot enforcement

**Enforcement:** Uses indicator variables and ordering constraints to ensure contiguous scheduling.

**Rationale:** Reduces idle time at venues and improves operational efficiency.

---

### Rule 10: Club Day Requirements
**Constraint:** `ClubDayConstraint`

**Description:** On designated club days:
- Every club team must play at least one game
- Teams in the same grade should play each other (intra-club matchups)
- All games must be on the same field
- Games must be in contiguous timeslots

**Enforcement:** Multiple sub-constraints for presence, matchups, field unity, and contiguity.

**Rationale:** Club days are special events requiring all teams together at one venue.

---

### Rule 11: Equal Matchup Spacing
**Constraint:** `EqualMatchUpSpacingConstraint`

**Description:** Matchups between the same teams should be evenly spaced across rounds.

**Enforcement:** Uses round sum calculations with bounds:
- Minimum spacing: `space - SLACK` rounds
- Maximum spacing: `space + SLACK` rounds

Where `space = max_rounds // num_teams`

**Rationale:** Prevents teams from playing each other in consecutive weeks.

---

### Rule 12: Club Grade Adjacency
**Constraint:** `ClubGradeAdjacencyConstraint`

**Description:** Teams from the same club in adjacent grades (e.g., 2nd and 3rd) cannot play at the same timeslot.

**Grade Order:** PHL → 2nd → 3rd → 4th → 5th → 6th

**Enforcement:** For each adjacent grade pair in each club, game sums at same slot must be ≤ 1.

**Rationale:** Allows players/supporters to watch both teams.

---

### Rule 13: Club vs Club Alignment
**Constraint:** `ClubVsClubAlignment`

**Description:** When clubs face each other across multiple grades, games should be aligned to the same round and field.

**Enforcement:** Uses indicator variables to ensure coincidence of matchups across grades, with field alignment constraints.

**Rationale:** Simplifies organization and enhances rivalry weekends.

---

## Soft Constraints

These constraints use penalty variables that are minimized in the objective function.

### Rule 14: Maitland Home Grouping
**Constraint:** `MaitlandHomeGrouping`

**Description:** 
- Encourages all Maitland games in a week to be either all home or all away
- **Hard element:** No back-to-back Maitland home weekends

**Penalty:** `min(home_games, away_games)` per week

**Weight:** 1,000,000

**Rationale:** Reduces travel burden by grouping home/away weeks.

---

### Rule 15: Away at Maitland Grouping
**Constraint:** `AwayAtMaitlandGrouping`

**Description:**
- Encourages minimal variety in away clubs visiting Maitland each weekend
- **Hard limit:** Maximum 3 different away clubs per Maitland weekend

**Penalty:** `num_away_clubs - 1` per week

**Weight:** 100,000

**Rationale:** Reduces complexity and improves travel coordination.

---

### Rule 16: Maximize Clubs Per Timeslot
**Constraint:** `MaximiseClubsPerTimeslotBroadmeadow`

**Description:** Encourages diversity of clubs within each timeslot at Broadmeadow.

**Penalty:** `total_teams_playing - num_clubs` per timeslot

**Weight:** 5,000

**Rationale:** Ensures fair exposure and prevents monopolization of slots.

---

### Rule 17: Minimize Clubs Per Field Per Day
**Constraint:** `MinimiseClubsOnAFieldBroadmeadow`

**Description:**
- Encourages continuity—clubs should play multiple games on the same field
- **Hard limit:** Maximum 5 clubs on any field on any day

**Penalty:** `|num_clubs - 2|` per (week, date, field)

**Weight:** 5,000

**Rationale:** Reduces setup/transition time and improves club experience.

---

### Rule 18: Preferred Times
**Constraint:** `PreferredTimesConstraint`

**Description:** Teams/clubs can specify times they prefer NOT to play.

**Penalty:** 1 per violation

**Weight:** 10,000,000

**Rationale:** Accommodates known commitments (events, holidays, etc.).

---

### Rule 19: PHL Preferred Dates
**Part of:** `PHLAndSecondGradeTimes`

**Description:** PHL games should occur on preferred dates when possible.

**Penalty:** `|games_on_date - 1|` per preferred date

**Weight:** 10,000

**Rationale:** Accommodates broadcast schedules and special events.

---

## Implied Rules

These rules arise from the combination of constraints or data filtering.

### Implied Rule 1: Field-Team Compatibility
**Source:** Decision variable generation (`generate_X`)

**Description:** Games can only be scheduled at venues that are home to at least one of the playing teams.

---

### Implied Rule 2: Grade-Time Restrictions
**Source:** Decision variable generation

**Description:**
- PHL plays only at designated PHL times
- 2nd grade plays between 11:30 and 17:30
- 2nd grade never plays on Friday or South Field

---

### Implied Rule 3: Location-Specific Day Availability
**Source:** Decision variable generation

**Description:** Each venue has defined days and times when games can occur.

---

### Implied Rule 4: Venue Unavailability
**Source:** `field_unavailabilities` data

**Description:** Certain dates are blocked for specific venues (holidays, maintenance, etc.).

---

### Implied Rule 5: Friday Night Restrictions
**Source:** Decision variable generation + `PHLAndSecondGradeTimes` constraint

**Description:**
- Friday games are PHL-only (only PHL grade plays on Fridays)
- No Friday games at Maitland (PHL-only venue)
- Maximum 3 Friday games at Broadmeadow (Newcastle International Hockey Centre)
- Exactly 8 Friday games at Gosford (Central Coast Hockey Park) - AGM decision 2026
- 8pm start time at Gosford confirmed at AGM

**Note:** Lower grades (3rd-6th) are automatically excluded from Friday variables during decision variable generation.

---

## Constraint Interactions

### Interaction 1: PHL + 2nd Grade Constraints
`PHLAndSecondGradeAdjacency` and `PHLAndSecondGradeTimes` work together to ensure:
- No time conflicts between PHL and 2nd grade from same club
- Adequate travel time between venues
- No simultaneous PHL games at main venue

### Interaction 2: Home/Away Balance
`FiftyFiftyHomeandAway`, `MaitlandHomeGrouping`, and `AwayAtMaitlandGrouping` combine to:
- Ensure fair home/away distribution
- Group home games for efficiency
- Limit away team variety per weekend

### Interaction 3: Field Optimization
`EnsureBestTimeslotChoices`, `MinimiseClubsOnAFieldBroadmeadow`, and `MaximiseClubsPerTimeslotBroadmeadow` balance:
- Contiguous timeslot usage
- Club continuity on fields
- Diversity within timeslots

### Interaction 4: Club Day Integration
`ClubDayConstraint` interacts with most other constraints, requiring:
- Temporary relaxation of grade adjacency (intra-club matchups)
- Field concentration instead of distribution
- Priority scheduling for club teams

---

## Constraint Application Stages

The solver applies constraints in four stages, each building on the previous solution. This staged approach improves solver performance and allows partial solutions if soft constraints cannot be fully satisfied.

### Stage 1: Required Constraints (Must Satisfy)

These constraints are physical necessities - the draw is invalid without them.

| Order | Constraint | Why Required |
|-------|-----------|--------------|
| 1 | `NoDoubleBookingTeamsConstraint` | A team cannot play two games at once |
| 2 | `NoDoubleBookingFieldsConstraint` | A field cannot host two games at once |
| 3 | `EnsureEqualGamesAndBalanceMatchUps` | Core fairness - every team gets equal games |
| 4 | `TeamConflictConstraint` | User-specified conflicts (e.g., shared players) |

**Checkpoint:** `checkpoints/stage_1_required.bin`

---

### Stage 2: Structural Constraints (Strong Requirements)

These constraints define the structure of the schedule, primarily for travel feasibility and venue fairness.

| Order | Constraint | Purpose |
|-------|-----------|---------|
| 5 | `FiftyFiftyHomeandAway` | Away teams (Maitland/Gosford) get balanced home/away |
| 6 | `MaxMaitlandHomeWeekends` | No back-to-back Maitland home weekends |
| 7 | `ClubDayConstraint` | Respect special dates (e.g., club days) |
| 8 | `PHLAndSecondGradeTimes` | PHL timing rules at Broadmeadow |
| 9 | `PHLAndSecondGradeAdjacency` | PHL/2nd grade play adjacent slots |
| 10 | `ClubGradeAdjacencyConstraint` | Adjacent grades from same club separated |

**Checkpoint:** `checkpoints/stage_2_structural.bin`

---

### Stage 3: Optimization Constraints (Venue & Distribution)

These constraints optimize for venue efficiency and even distribution.

| Order | Constraint | Purpose |
|-------|-----------|---------|
| 11 | `AwayAtMaitlandGrouping` | Max 3 away clubs at Maitland per weekend |
| 12 | `MinimiseClubsOnAFieldBroadmeadow` | Reduce club switching on Broadmeadow fields |
| 13 | `EnsureBestTimeslotChoices` | Teams get their best available timeslots |
| 14 | `MaximiseClubsPerTimeslotBroadmeadow` | Maximize club diversity per timeslot |
| 15 | `ClubVsClubAlignment` | Align rivalry/paired club matchups |

**Checkpoint:** `checkpoints/stage_3_optimization.bin`

---

### Stage 4: Soft Constraints (Preferences)

These are "nice to have" preferences applied with penalties. The solver will satisfy as many as possible.

| Order | Constraint | Purpose |
|-------|-----------|---------|
| 16 | `MaitlandHomeGrouping` | Group Maitland home games for travel efficiency |
| 17 | `EqualMatchUpSpacingConstraint` | Space out repeat matchups across season |
| 18 | `PreferredTimesConstraint` | Satisfy club/team preferred times |

**Checkpoint:** `checkpoints/stage_4_soft.bin`

---

### Staged Solving Workflow

```
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Required Constraints                              │
│  ├── NoDoubleBookingTeams                                   │
│  ├── NoDoubleBookingFields                                  │
│  ├── EnsureEqualGamesAndBalanceMatchUps                    │
│  └── TeamConflict                                           │
│                    ↓ [Save checkpoint, pass hints]          │
├─────────────────────────────────────────────────────────────┤
│  STAGE 2: Structural Constraints                            │
│  ├── FiftyFiftyHomeAway                                     │
│  ├── MaxMaitlandHomeWeekends                                │
│  ├── ClubDayConstraint                                      │
│  ├── PHLAndSecondGradeTimes                                 │
│  ├── PHLAndSecondGradeAdjacency                            │
│  └── ClubGradeAdjacency                                     │
│                    ↓ [Save checkpoint, pass hints]          │
├─────────────────────────────────────────────────────────────┤
│  STAGE 3: Optimization Constraints                          │
│  ├── AwayAtMaitlandGrouping                                 │
│  ├── MinimiseClubsOnAField                                  │
│  ├── EnsureBestTimeslotChoices                              │
│  ├── MaximiseClubsPerTimeslot                               │
│  └── ClubVsClubAlignment                                    │
│                    ↓ [Save checkpoint, pass hints]          │
├─────────────────────────────────────────────────────────────┤
│  STAGE 4: Soft Preferences                                  │
│  ├── MaitlandHomeGrouping                                   │
│  ├── EqualMatchUpSpacing                                    │
│  └── PreferredTimes                                         │
│                    ↓ [Final solution]                       │
└─────────────────────────────────────────────────────────────┘
```

---

### Constraint Group Summary Table

| Stage | Type | Count | Checkpoint |
|-------|------|-------|------------|
| 1 | Required | 4 | `stage_1_required.bin` |
| 2 | Structural | 6 | `stage_2_structural.bin` |
| 3 | Optimization | 5 | `stage_3_optimization.bin` |
| 4 | Soft/Preferences | 3 | `stage_4_soft.bin` |
| **Total** | | **18** | |

---

## Appendix: Variable Key Structure

Game decision variables use the following key structure:
```python
(team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)
```

Indices:
- 0: team1
- 1: team2
- 2: grade
- 3: day
- 4: day_slot
- 5: time
- 6: week
- 7: date
- 8: round_no
- 9: field_name
- 10: field_location
