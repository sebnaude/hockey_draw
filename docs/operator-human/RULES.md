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
- Maximum 3 Friday night PHL games at Broadmeadow (`CONSTRAINT_DEFAULTS['max_friday_broadmeadow']`)
- Exactly 8 Friday night PHL games at Gosford (Central Coast Hockey Park) - AGM decision 2026 (`CONSTRAINT_DEFAULTS['gosford_friday_games']`)
- Exactly 2 Friday night PHL games at Maitland Park - Gosford vs Maitland only (`CONSTRAINT_DEFAULTS['maitland_friday_games']`)
- NIHC Friday matchups limited to specific dates/teams via `FORCED_GAMES`:
  - May 8: Norths vs Maitland
  - June 19: Tigers vs Wests
  - July 24: Norths vs TBC (any opponent)

**Enforcement:** Uses timeslot indicators and sum constraints per location/club combination. Friday game counts are configured via `CONSTRAINT_DEFAULTS` (`gosford_friday_games`, `max_friday_broadmeadow`). NIHC Friday matchups are filtered at variable generation time using `FORCED_GAMES` and `BLOCKED_GAMES`.

**Rationale:** Prevents spectator/player conflicts and ensures fair distribution of prime-time slots. The Gosford Friday requirement was confirmed at the 2026 AGM. NIHC Friday matchups are pre-scheduled to align with Junior Boys program and special events.

---

### Rule 6: Away-Club Home/Away Expectations (spec-004)

For any club whose home ground is NOT Broadmeadow (currently Maitland and
Gosford; any future expansion clubs automatically inherit the same treatment),
the system enforces three cooperating expectations:

1. **Per-opponent balance.** For every pair of teams, the number of games
   played at the away-club's home ground equals exactly half of their season
   meetings — rounded either way for odd totals. With 3 meetings each pair
   gets 2H/1A or 1H/2A; with 6 meetings each pair gets 3H/3A.

2. **Per-team aggregate balance.** Across all of an away-club team's games,
   home and away totals are within ±1 of each other. Prevents the corner case
   where every odd-meeting pair lands on the same side and the team ends up,
   say, 12H/6A across the season.

3. **Per-club home weekend total — FORCED-Friday aware.** The number of
   weekends the club appears at its home ground equals the maximum games-
   required across its grades (e.g. PHL plays 18 + 3rd plays 20 → 20 weekend
   appearances). Sundays vs Fridays are tracked separately: each PHL
   Friday-night fixture the convenor pins to the home ground via FORCED_GAMES
   reduces the required Sunday count by one (no point adding Sundays on top
   of an already-claimed weekend). When another grade requires more games
   than PHL, the FORCED Fridays are absorbed into the other-grade-driven
   total — they don't add weekends, they just label some of them Friday.

**Constraints involved:**

- `AwayClubHomeWeekendsCount` — enforces (3).
- `AwayClubPerOpponentAndAggregateHomeBalance` — enforces (1) + (2).
- Together these supersede the legacy combined `FiftyFiftyHomeandAway`
  class, which is retained only for backwards-compat parity tests.

**Rationale:** Ensures fairness for teams that must travel significant
distances AND keeps the away-ground's Sundays densely populated rather than
diluted with extra home games beyond what the season actually needs.

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

**Enforcement (spec-008):** Reads the convenor-facing number `S` as
"the number of rounds you want **between** two meetings of the same pair."
So `S=2` means "at least two whole rounds between meetings" — rounds 1 and
4 are fine; rounds 1 and 3 are not (only one round, round 2, between them).

By default `S = ideal_gap(T)` for grade size `T`. The CLI flag `--slack N`
loosens this by `N` rounds (the convenor accepts the gap shrinking by `N`).

**Rationale:** Prevents teams from playing each other in close succession.

---

### Rule 11b: Balanced Byes (spec-008)
**Constraint:** `BalancedByeSpacing`

**Description:** If your team has byes (rounds where it doesn't play),
they'll be spread across the season just like repeat matchups — no team
gets three byes in a row early then has to play every round after.

**Enforcement:** For each team in each grade, the rounds in which the team
doesn't play must sit at least `S` rounds apart, where
`S = ideal_bye_gap(R, byes) = max(0, R // byes - 1)`. E.g. a 20-round
season with 2 byes per team forces those byes at least 9 rounds apart.

**Slack:** Use `--slack N` from the CLI to loosen by `N` rounds, or set a
permanent default via `CONSTRAINT_DEFAULTS['bye_spacing_base_slack']`.
The slack key `BalancedByeSpacing` is independent of matchup spacing —
you can loosen one without touching the other.

**Rationale:** Convenor request — byes are part of the schedule too;
spreading them prevents stretches of "play every weekend" or "bye every
weekend" pile-ups.

---

### Rule 12: Same-Grade Same-Club No Concurrency (spec-007)
**Constraint:** `SameGradeSameClubNoConcurrency` (atom; severity 1, hard).

**Description:** When a club fields multiple teams in the same grade (e.g. Tigers
3rd-A and Tigers 3rd-B), those teams cannot play different opponents at the
same `(week, day_slot)`. One parent literally cannot watch both at once.
Intra-club derbies — the two duplicate teams playing each other — are a single
game with one decision variable and stay feasible.

**Enforcement:** Per `(club, grade, week, day_slot)` bucket, the sum of
cross-club games involving any duplicate-set team from that club must be ≤ 1.

**spec-007 change to convenor expectations:** the older "your adjacent
grades cannot share a timeslot" hard rule has been **removed entirely**.
Convenor experience was that this was over-restrictive — many parents
handle adjacent-grade kids fine with overlapping slots, and forbidding it
hard caused infeasibility on tight weeks. There is no replacement soft
penalty: adjacent-grade concurrency simply happens when the draw needs it
to. If you have a specific real-world conflict (siblings in non-adjacent
grades, a coach who runs two teams), declare it as an entry in
`CONSTRAINT_DEFAULTS['TEAM_PAIR_NO_CONCURRENCY']` and the
`TeamPairNoConcurrency` soft atom will minimise it.

**Rationale:** Keep the rule that's genuinely fundamental; let the convenor
opt in to per-pair preferences rather than imposing a blanket adjacent-grade
ban.

---

### Rule 13: Club vs Club Stacked Alignment

**Constraints:** `ClubVsClubStackedWeekends` + `ClubVsClubStackedCoLocation` (spec-005)

**Description (the convenor's intent):** When our two clubs play, expect
the bigger-grade group to bring along the smaller grades for the day,
back-to-back on one field. If Maitland visits Norths in PHL on a given
Sunday and 2nd grade also plays Maitland-vs-Norths that day, those games
will be on the same field with no slot gap between them. The same holds
for 3rd / 4th / 5th / 6th if they're also playing the pair that day.

**The stacking pattern:**

For each unordered pair of clubs `(A, B)` compute per-grade meeting counts
(the number of Maitland-vs-Norths PHL meetings, the number of
Maitland-vs-Norths 2nd grade meetings, ...). The draw is built so the
higher-count grades' meeting weeks are a **strict superset** of the
lower-count grades' meeting weeks. Example with PHL=4, 2nd=3, 3rd=2,
4th=2, 5th=1, 6th=0:

| Weekend | Active grades for this pair                |
|---------|--------------------------------------------|
| W1      | {PHL, 2nd, 3rd, 4th, 5th}                  |
| W2      | {PHL, 2nd, 3rd, 4th}                       |
| W3      | {PHL, 2nd}                                 |
| W4      | {PHL}                                      |

Totals: PHL=4 ✓, 2nd=3 ✓, 3rd=2 ✓, 4th=2 ✓, 5th=1 ✓. No "stray" pair
weekend where a lower grade plays without every higher grade also playing.

**Co-location on every stacked weekend (≥ 2 grades active):**

- All those games on the **same field** (no splitting across EF/WF).
- **Contiguous day_slots** — no empty slot between two active games.
  Adjacent grades end up on adjacent timeslots, so a parent watching
  multiple grades stays on one field without gaps.

**PHL Friday handling:** PHL Friday-night forced games consume the
per-pair meeting count but cannot satisfy Sunday stacking. The Sunday
budget for PHL stacking = `total PHL meetings(A, B) - FORCED PHL Friday
games(A, B)`. If the convenor pins 2 Maitland-vs-Norths PHL Fridays via
`FORCED_GAMES`, the stacking pretends PHL meetings = total - 2 (the
remaining Sundays).

**Multi-team-per-club-per-grade:** if one club fields 2 teams in 3rd
grade and the other fields 1, that's 2 distinct matchups for the pair in
3rd grade — and the stacking math uses 2 × per-matchup-meetings (not 1).

**Enforcement:** All HARD. The Sunday count per (pair, grade) is pinned
by `sum_w play[g, w] == sunday_budget(g)`; the implication chain
`play[lower, w] <= play[higher, w]` forces the strict-superset structure.
The co-location atom enforces same-field + contiguous-slot only when ≥ 2
grades coincide.

**Rationale:** Clubs travel once per matchup weekend; parents watching
multiple grades of their club versus the same opponent do so contiguously
on one field. Replaces a looser pre-spec-005 implementation that only
asked for "≥ N coincident rounds" without precise stacking or co-location.

---

## Soft Constraints

These constraints use penalty variables that are minimized in the objective function.

### Rule 14: Maitland Home Grouping
**Constraint:** `MaitlandHomeGrouping`

**Description:**
- Encourages all Maitland games in a week to be either all home or all away
- **Hard element:** Maximum consecutive home weekends enforced via sliding window. With `--slack N`, allows up to (1 + N) consecutive home weeks. Default (no slack): no back-to-back.

**Sliding window enforcement:** In any window of (max_consecutive + 1) consecutive Maitland-game weeks, at most max_consecutive can be home weeks. No-play weeks are excluded from the sequence.

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

**Description:** Teams/clubs can specify times they prefer NOT to play. The
`PREFERENCE_NO_PLAY` config dict accepts several filters in combination —
specific dates, time-of-day, day-of-week, venue, field, grade. Date-less
entries (e.g. *"Maitland teams prefer not to play at 08:30"*) apply across
every playable week.

**2026:** Maitland and Port Stephens both prefer not to play at 08:30 on any
date (added by spec-012). The solver receives a soft penalty for every
matching scheduled game and will avoid those slots when feasible.

**Penalty:** 1 per violating scheduled game.

**Weight:** 200,000 in `PENALTY_WEIGHTS['PreferredTimesConstraint']`.

**Rationale:** Accommodates known commitments (events, holidays, club-wide
"we'd rather not start that early" preferences).

---

### Rule 19: PHL Preferred Dates
**Part of:** `PHLAndSecondGradeTimes`

**Description:** PHL games should occur on preferred dates when possible.

**Penalty:** `|games_on_date - 1|` per preferred date

**Weight:** 10,000

**Rationale:** Accommodates broadcast schedules and special events.

---

### Rule 20: Alphabetical Matchup Ordering (Soft Tie-Break)
**Constraint:** `SoftLexMatchupOrdering`

**Description:** In the absence of other influences, matchups between teams are scheduled so that alphabetically-earlier matchups tend to appear in earlier rounds. For example, Norths vs Tigers (N before T) will usually appear before Tigers vs Wests (T before W) across the season.

This is a **pure soft tie-break** — it never prevents any match from being scheduled. Its only effect is to make the published draw feel more predictable and structured when multiple scheduling options are otherwise equivalent.

**Penalty:** Proportional to the alphabetical rank of each pair times the number of games scheduled. Weight is kept very small (default 1) so it cannot override any real scheduling constraint.

**Weight:** 1 (deliberately tiny — tie-break only)

**Rationale:** Clubs reading the published draw see a more orderly sequence of matchups. Removes solver-arbitrary ordering that would otherwise vary between runs with no logical explanation.

---

### Rule 21: Preferred / Avoided Weekends at Away Grounds
**Constraint:** `PreferredWeekendsAwayGround`

**Description:** The convenor can declare specific dates where a particular away venue is *preferred* or *avoided*. Two modes:

- **Avoid:** The solver tries not to schedule any games at that venue on that date. For example, if Newcastle Knights are playing an NRL home game at Maitland Park on a given Sunday, Maitland HC prefers not to host hockey that day (traffic, facilities, parking). The solver will schedule Maitland Park games on other dates if feasibly possible.
- **Prefer:** The solver tries to schedule at least one game at that venue on that date. For example, if extra foot traffic is expected, the convenor may want to make use of the venue.

This is a **soft constraint** — the solver will respect it when feasible, but may schedule (or not schedule) at the venue anyway if no other feasible assignment exists. For a **hard** block, use `BLOCKED_GAMES` instead.

**Configuration:** `PREFERRED_WEEKENDS` list in the season config. Each entry specifies a date (or list of dates), a venue (`field_location`), an optional specific field (`field_name`), and a mode (`'avoid'` or `'prefer'`). An optional `weight` overrides the default penalty.

**2026:** Six NRL-Knights home games at Maitland Park are pre-configured as `avoid` entries (5 April, 26 April, 3 May, 28 June, 5 July, 16 August). Maitland HC does not want to play on these dates.

**Weight:** 1,000 (default `PENALTY_WEIGHTS['preferred_weekends_away_ground']`). Moderate — the solver will look for alternatives but will not sacrifice other important constraints to avoid a clashing date.

---

### Rule 22: Maitland Home/Away Weekend Alternation
**Constraint:** `MaitlandAlternateHomeAway` (spec-012)

**Description:** Across Maitland's playing weekends, the convenor wants
roughly alternating (home, away) pairs rather than long runs of either
type. Two related constraints work together:

- **Hard rule (already in place):** `NonDefaultHomeGrouping` with default
  slack forbids two consecutive home weekends ("HH"). So you'll never see
  Maitland playing at Maitland Park two weekends in a row.
- **Soft rule (spec-012):** `MaitlandAlternateHomeAway` adds a small
  penalty per consecutive playable-week pair where both weekends are the
  same type (HH or AA). Since HH is already hard-forbidden, this
  effectively discourages long runs of away weekends.

Together these push the solver toward an H A H A H A pattern when
feasible. Bye weeks for Maitland (no game scheduled) contribute neither
home nor away to a pair — they're treated as a gap that does not penalise
either neighbour.

**Penalty:** 1 per consecutive same-type pair.

**Weight:** `PENALTY_WEIGHTS['maitland_alternate_home_away']` — 10,000 in
the 2026 season config. Tuned so the atom shapes the alternation without
overwhelming `ClubVsClubAlignment` (50,000) or `PreferredTimesConstraint`
(200,000).

**Scope:** Maitland only. Gosford has its own home/away semantics
(Friday-night PHL forced count + sparse Sunday slots) and is intentionally
out of scope. Adding alternation for other away clubs is a separate spec
if needed.

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
**Source:** Decision variable generation + `PHLAndSecondGradeTimes` constraint + `FORCED_GAMES` / `BLOCKED_GAMES`

**Description:**
- Friday games are PHL-only (only PHL grade plays on Fridays)
- Friday at Maitland Park: Gosford vs Maitland PHL only (7pm, non-Gosford clubs blocked via `BLOCKED_GAMES`)
- Maximum 3 Friday games at Broadmeadow (Newcastle International Hockey Centre)
- Exactly 8 Friday games at Gosford (Central Coast Hockey Park) - AGM decision 2026
- Exactly 2 Friday games at Maitland Park - Gosford vs Maitland only
- 8pm start time at Gosford confirmed at AGM, 7pm at Maitland and NIHC
- **NIHC Friday matchups restricted to specific dates** (2026+):
  - May 8: Norths vs Maitland only
  - June 19: Tigers vs Wests only
  - July 24: Norths vs any opponent
- **Gosford Friday forced dates** (2026): Mar 27, Apr 17, Apr 24, May 15, May 29

**Implementation:** Friday game counts are set in `CONSTRAINT_DEFAULTS` (`gosford_friday_games`, `max_friday_broadmeadow`, `maitland_friday_games`). NIHC Friday matchup filtering is done via `FORCED_GAMES`. Non-confirmed Gosford Friday dates are blocked via `BLOCKED_GAMES`. Non-Gosford clubs are blocked from Maitland Fridays via `BLOCKED_GAMES`. Friday timeslots are controlled by `PHL_GAME_TIMES`.

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
| 10 | `SameGradeSameClubNoConcurrency` (spec-007) | Same-club same-grade duplicate teams cannot share a slot |

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
│  └── SameGradeSameClubNoConcurrency  (spec-007 replacement)  │
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
