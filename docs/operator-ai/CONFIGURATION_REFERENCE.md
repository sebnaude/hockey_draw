# Configuration Reference

> **Purpose:** Complete reference for all configurable parameters in the system.
> **When to use:** When you need to change any system behavior.

---

## Configuration File Location

All season-specific configuration lives in: `config/season_{year}.py`

---

## SEASON_CONFIG (Core Settings)

```python
SEASON_CONFIG = {
    'year': 2026,
    'start_date': datetime(2026, 3, 22),      # First playing Sunday
    'end_date': datetime(2026, 8, 30),         # Last club game before finals

    'max_rounds': 22,           # Maximum games per team (see notes below)
    # Dummy overflow slots: not attached to a real time/venue, eases solver burden.
    # Penalty for using them is set in PENALTY_WEIGHTS['dummy_slots'].
    'num_dummy_timeslots': 3,
    
    'play_anzac_sunday': True,  # Whether to play on ANZAC Sunday
    
    # Data paths (relative to project root)
    'teams_data_path': 'data/2026/teams',
    'noplay_data_path': 'data/2026/noplay',
    'field_availability_path': 'data/2026/field_availability',
    
    # References to other config sections
    'fields': FIELDS,
    'day_time_map': DAY_TIME_MAP,
    'phl_game_times': PHL_GAME_TIMES,
    'second_grade_times': SECOND_GRADE_TIMES,
    'field_unavailabilities': FIELD_UNAVAILABILITIES,
    'club_days': CLUB_DAYS,
    'preferred_games': PREFERRED_GAMES,
    'preference_no_play': PREFERENCE_NO_PLAY,
    'special_games': SPECIAL_GAMES,
    'forced_games': FORCED_GAMES,
    'blocked_games': BLOCKED_GAMES,
    'home_field_map': { 'Maitland': 'Maitland Park', 'Gosford': 'Central Coast Hockey Park' },
    'grade_order': ['PHL', '2nd', '3rd', '4th', '5th', '6th'],
    'penalty_weights': PENALTY_WEIGHTS,
    'max_weekends_per_grade': MAX_WEEKENDS_PER_GRADE,
    'grade_rounds_override': GRADE_ROUNDS_OVERRIDE,
    'grade_scheduling_method': GRADE_SCHEDULING_METHOD,
    'max_time_per_stage': 172800,  # 2 days per solver stage (seconds)
}
```

### num_dummy_timeslots Parameter

**What it does:** Controls how many dummy overflow timeslots are created. These are not attached to a real time or venue — they exist to give the solver extra capacity and ease scheduling pressure.

**Default:** 3

**How it works:**
- Dummy variables use short 4-tuple keys `(t1, t2, grade, index)` and are merged into X
- Constraints exclude them via `len(key) < 11` or `and t.day` checks
- Game-count constraints (e.g. `EqualGamesAndBalanceMatchUps`) explicitly include dummy vars
- The solver penalises using dummy slots via `PENALTY_WEIGHTS['dummy_slots']`
- Set `dummy_slots` penalty weight to 0 for free use, or higher to discourage use

**Config example:**
```python
'num_dummy_timeslots': 3,  # Number of overflow slots

PENALTY_WEIGHTS = {
    ...
    'dummy_slots': 1,  # Penalty per dummy slot used (0 = no penalty)
}
```

### max_time_per_stage Parameter

**What it does:** Sets the default maximum solver time per stage (in seconds). Each stage in severity-based or default solving uses this limit unless the stage dict overrides with its own `max_time_seconds`.

**Default:** 172800 (2 days)

### max_rounds Parameter

**What it does:**
- Sets the maximum number of games any team can play in the season
- Used by `max_games_per_grade()` to calculate actual games per grade
- Acts as a cap: `actual_games = min(calculated, max_rounds)`

**How to set it:**
- Count available playing weekends (total Sundays minus blocked)
- Set max_rounds to this number
- Example: 24 Sundays - 2 blocked = 22 available → `max_rounds: 22`

**Grade interaction:**
- Lower grades with fewer teams may play fewer games naturally
- PHL with 5 teams: 4 matchups × 5 rounds = 20 games (capped by math, not max_rounds)
- 4th grade with 11 teams: Each pair meets 2× = 20 games per team

---

## FIELDS (Playing Surfaces)

```python
FIELDS = [
    {'location': 'Newcastle International Hockey Centre', 'name': 'SF'},  # South Field
    {'location': 'Newcastle International Hockey Centre', 'name': 'EF'},  # East Field
    {'location': 'Newcastle International Hockey Centre', 'name': 'WF'},  # West Field
    {'location': 'Maitland Park', 'name': 'Maitland Main Field'},
    {'location': 'Central Coast Hockey Park', 'name': 'Wyong Main Field'},
]
```

**Field codes:**
- SF = South Field (NIHC) - Lower grades only, PHL excluded
- EF = East Field (NIHC) - All grades
- WF = West Field (NIHC) - All grades

---

## DAY_TIME_MAP (Base Timeslots)

```python
DAY_TIME_MAP = {
    'Newcastle International Hockey Centre': {
        'Sunday': [tm(8, 30), tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30), tm(19, 0)]
    },
    'Maitland Park': {
        'Sunday': [tm(9, 0), tm(10, 30), tm(12, 0), tm(13, 30), tm(15, 0), tm(16, 30)]
    },
    'Central Coast Hockey Park': {
        'Sunday': [tm(12, 0), tm(13, 30)],  # 12pm or 1:30pm only
    }
}
```

**Purpose:**
- Defines ALL possible timeslots for lower grades (3rd-6th)
- PHL and 2nd grade have their own filtering dicts (see below)

**Derived: `no_field_slots` (spec-021).** `config.defaults.compute_no_field_slots(day_time_map)`
computes, per location, the count of distinct game-times (max over its days) — e.g. NIHC = 8,
Maitland Park = 6, Central Coast = 2 above. It is surfaced as `data['no_field_slots']` by
`build_season_data`. The `ClubNoConcurrentSlot` atom reads it to size its per-(club, timeslot,
venue) cap: `max(1, ceil(club_team_count / no_field_slots[location]))`. This is what lets a club
with more teams than a small venue has timeslots (e.g. Central Coast's 2) take *forced* double-ups
instead of going infeasible. It is **derived, not configured** — add a venue or a time to
`DAY_TIME_MAP` and the cap updates automatically.

---

## PHL_GAME_TIMES (PHL Variable Filtering)

**Purpose:** Controls which decision variables are created for PHL games.

```python
PHL_GAME_TIMES = {
    'Newcastle International Hockey Centre': {
        'EF': {  # East Field only (SF excluded for PHL)
            'Friday': [tm(19, 0)],  # 7pm
            'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
        },
        'WF': {  # West Field only
            'Friday': [tm(19, 0)],
            'Sunday': [tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0)]
        },
        # NOTE: SF deliberately excluded
    },
    'Central Coast Hockey Park': {
        'Wyong Main Field': {
            'Friday': [tm(20, 0)],  # 8pm (AGM confirmed)
            'Sunday': [tm(12, 0), tm(13, 30)]
        },
    },
    'Maitland Park': {
        'Maitland Main Field': {
            'Sunday': [tm(12, 0), tm(13, 0), tm(15, 0), tm(16, 30)]
        },
    },
}
```

**Key rules:**
- PHL cannot play on South Field (SF) - excluded from dict
- Gosford is PHL-only venue (lower grades excluded)
- Friday nights only at NIHC (7pm) and Gosford (8pm)

**Effect:** Only ~2,000 PHL variables created vs ~20,000+ if unrestricted.

---

## SECOND_GRADE_TIMES (2nd Grade Variable Filtering)

**Purpose:** Controls which decision variables are created for 2nd grade.

```python
SECOND_GRADE_TIMES = {
    'Newcastle International Hockey Centre': {
        'EF': {  # No SF
            'Sunday': [tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30)]
        },
        'WF': {
            'Sunday': [tm(10, 0), tm(11, 30), tm(13, 0), tm(14, 30), tm(16, 0), tm(17, 30)]
        },
    },
    # Gosford NOT listed - PHL-only venue
    'Maitland Park': {
        'Maitland Main Field': {
            'Sunday': [tm(10, 30), tm(12, 0), tm(13, 0), tm(15, 0), tm(16, 30)]
        },
    },
}
```

**Key rules:**
- 2nd grade cannot play at Gosford (not listed)
- Times are PHL times ± 1 slot (for back-to-back scheduling)
- Cannot create NEW timeslots - must exist in DAY_TIME_MAP

---

## FIELD_UNAVAILABILITIES (Blocked Dates)

```python
FIELD_UNAVAILABILITIES = {
    'Newcastle International Hockey Centre': {
        'weekends': [
            datetime(2026, 4, 4),   # Easter
            datetime(2026, 5, 16),  # Masters SC
        ],
        'whole_days': [datetime(2026, 4, 25)],  # ANZAC Day
        'part_days': [],  # For future: specific timeslots blocked
    },
    ...
}
```

**Effect:** Completely removes all game variables for these dates - HARD constraint.

---

## Friday Night Configuration

Friday night games are configured across multiple existing config structures (no separate `FRIDAY_NIGHT_CONFIG` dict):

- **Game counts**: `FORCED_GAMES` entries — `{grade='PHL', day='Friday', field_location=<venue>, count=N, constraint='equal'|'lesse'}`. Per-venue count budgets live in season config so they are discoverable, reportable, and per-season tweakable. See `docs/PERENNIAL_RULES.md` Rule 3.
- **Timeslots**: `PHL_GAME_TIMES` controls which Friday times exist at each venue (8pm Gosford, 7pm Maitland, 7pm NIHC EF only)
- **Forced dates**: `FORCED_GAMES` forces specific Friday dates at Gosford and specific matchups at NIHC. The multi-scope match (commit `cd8a338`) means each Friday game counts toward both its date/pair scope AND the per-venue count scope simultaneously.
- **Date filtering**: `BLOCKED_GAMES` prevents non-confirmed Gosford Friday dates, and blocks non-Gosford clubs from Maitland Fridays
- **Pre-solver consistency**: `validate_game_config` Phase 21 checks subset-related FORCED scopes for compatible counts (e.g. fatal if a narrower `equal 3` sits inside a broader `equal 2`).
- **Legacy reference**: `CONSTRAINT_DEFAULTS['gosford_friday_games']`/`['maitland_friday_games']`/`['max_friday_broadmeadow']` keys still exist; they're consumed by archived classes (`original.py`/`ai.py` and the legacy `_phl_times_hard()` parity reference) and retire when those archive in Phase 7c. New code should NOT read from them — use the FORCED_GAMES entries directly.

---

## PREFERENCE_NO_PLAY (Soft No-Play Requests)

```python
PREFERENCE_NO_PLAY = {
    'Crusaders_6th_Masters': {
        'club': 'Crusaders',
        'grade': '6th',           # Single grade
        'dates': [datetime(2026, 4, 17), ...],
        'reason': 'Masters State Championships',
    },
    'Souths_U18_SC': {
        'club': 'Souths',
        'grades': ['PHL', '2nd'],  # Multiple grades
        'dates': [datetime(2026, 5, 24)],
        'reason': 'U18 State Championships',
    },
    # spec-012: time-only (and day-only, field-only, …) filters.
    # Entries WITHOUT a `date`/`dates` key apply across EVERY playable week,
    # filtered by the remaining keys (time / day / field_name / field_location
    # / grade / grades). The structured format already accepts `'club'`; the
    # spec-012 work added pass-through of `time`, `day`, `day_slot`, `week`,
    # `field_name`, `field_location` from the entry into the matched
    # restriction dict.
    'maitland_no_8_30am': {
        'club': 'Maitland',
        'time': '08:30',                # filters X-vars where key[5] == '08:30'
        'description': 'Maitland teams prefer not to play at 08:30',
    },
    'port_stephens_no_8_30am': {
        'club': 'Port Stephens',
        'time': '08:30',
        'description': 'Port Stephens teams prefer not to play at 08:30',
    },
}
```

**Supported entry keys** (structured 2026 format, `value` is a dict with `'club'`):

| Key | Purpose | Required? |
|---|---|---|
| `club` | The club whose teams are penalised. Must match `Club.name` exactly. | yes |
| `grade` | Single grade filter (`'PHL'`, `'2nd'`, …). | no |
| `grades` | List of grades (mutually exclusive with `grade`). | no |
| `date` | Single date (string `'YYYY-MM-DD'` or `datetime`). | one of `date` / `dates` / a non-date filter |
| `dates` | List of dates. | (see above) |
| `time` | Time filter `'HH:MM'`, matches X-key column `time`. | no |
| `day` | Day filter (`'Sunday'`, `'Friday'`). | no |
| `day_slot` | Day-slot integer (1-indexed). | no |
| `week` | Week-number integer. | no |
| `field_name` | Field name (`'EF'`, `'WF'`, `'Maitland Main Field'`, …). | no |
| `field_location` | Venue (`'Maitland Park'`, `'Newcastle International Hockey Centre'`, …). | no |
| `reason` / `description` | Human-readable note, ignored by solver. | no |

An entry must have either at least one date OR at least one non-date filter
— a bare `{'club': 'X'}` entry is silently a no-op (it would penalise every
game and is almost certainly a config typo).

**Effect:** Adds penalty to `PreferredTimesConstraint` (severity 5) — SOFT constraint.

---

## Spacing slack keys (spec-008)

Two HARD spacing atoms each own an independent slack channel:

| Slack key (used in `constraint_slack` and `--slack N`) | Atom | Default base slack (in `CONSTRAINT_DEFAULTS`) | Effect |
|---|---|---|---|
| `EqualMatchUpSpacingConstraint` | `EqualMatchUpSpacing` (legacy class name preserved) | `spacing_base_slack` (default 0) | Each unit reduces S (the "free rounds between meetings" threshold) by 1, clamped at 0. |
| `BalancedByeSpacing` | `BalancedByeSpacing` (spec-008 Part B atom) | `bye_spacing_base_slack` (default 0) | Each unit reduces S for the bye-rounds threshold by 1, clamped at 0. |

The CLI `--slack N` flag (see `run.py::run_generate`) writes `N` into both keys
(and into every other slack-aware constraint's key). Set just one of them in
`data['constraint_slack']` to loosen one dimension without the other.

Math reference: `constraints/atoms/_spacing.py::ideal_gap(T)` and
`ideal_bye_gap(R, byes)`. See `docs/system/COUNT_ADJUSTERS.md` (spec-008
note section) for the full semantics.

---

## TEAM_PAIR_NO_CONCURRENCY (spec-007 soft per-pair conflict)

```python
CONSTRAINT_DEFAULTS['TEAM_PAIR_NO_CONCURRENCY'] = [
    ('Tigers 4th', 'Tigers 6th'),                  # base weight (multiplier=1)
    ('Maitland PHL', 'Maitland 5th', 5),           # weight multiplier 5
]
```

Each entry is `(team_a, team_b)` or `(team_a, team_b, weight_multiplier)`.

**Effect:** soft constraint enforced by the `TeamPairNoConcurrency` atom
(severity 3). For every `(week, day_slot)` where both teams could appear, a
penalty `raw = max(0, sum(vars_team_a) + sum(vars_team_b) - 1)` is added.
With the optional multiplier, the contribution is `multiplier * raw`. The
solver minimises the `TeamPairNoConcurrency` penalty bucket; the base bucket
weight comes from `PENALTY_WEIGHTS['TeamPairNoConcurrency']` (default 1000).

**Use case:** siblings in non-adjacent grades, particular coach conflicts,
or any per-team-pair "should not play at the same time" the convenor wants
the solver to honour when feasible.

**Important — spec-007 behaviour change:** the legacy
`ClubGradeAdjacencyConstraint` automatically forbade adjacent-grade
concurrency (e.g. PHL + 2nd) for every club. That hard rule has been
**removed entirely** — convenor experience was that it was over-restrictive
and caused infeasibility on tight weeks. If you still want any specific
adjacent-grade pair to avoid concurrency, declare it explicitly here. The
genuinely-fundamental same-grade-same-club rule (multiple Tigers 3rd-grade
teams cannot play simultaneously) is now the dedicated hard atom
`SameGradeSameClubNoConcurrency` (severity 1) and applies automatically.

---

## PREFERRED_WEEKENDS (spec-006)

Soft preferences for scheduling (or avoiding) games at a specific away venue on specific dates. Read by the `PreferredWeekendsAwayGround` atom (severity 5, `soft_optimisation` stage). Never blocks feasibility.

```python
PREFERRED_WEEKENDS = [
    {
        # Required: one of 'date' (single) or 'dates' (list). Mutually exclusive.
        'date':           '2026-04-05',          # YYYY-MM-DD
        # 'dates':        ['2026-04-05', '2026-04-26'],  # alternative: list of dates

        # Required: venue name matching field_location in decision variable keys.
        'field_location': 'Maitland Park',

        # Optional: specific field at the venue. Omit for venue-level (any field).
        # 'field_name':   'Maitland Main Field',

        # Required: 'avoid' (penalise games here) or 'prefer' (penalise missing games).
        'mode':           'avoid',

        # Optional: per-entry weight override. Defaults to PENALTY_WEIGHTS['preferred_weekends_away_ground'].
        # 'weight':       2000,

        # Human-readable; ignored by solver.
        'description':    'NRL Knights vs Raiders at Maitland',
    },
]
```

**mode semantics:**
- `avoid`: for each game scheduled at `field_location` (and `field_name` if set) on `date`, the solver incurs `weight × 1` penalty. Use when the convenor wants the solver to prefer scheduling games elsewhere that weekend.
- `prefer`: solver is penalised for *not* having a game at the venue on that date. Penalty = `weight × max(0, target_count - actual_games)`. Default `target_count = 1`. Use when the convenor wants the solver to prefer scheduling games at that venue.

**Conflicting entries** (prefer + avoid same date/venue): no crash — penalties accumulate independently. Solver finds the least-total-penalty assignment.

**Hard block vs soft avoid:** `PREFERRED_WEEKENDS` with `mode: avoid` is a *soft* preference. For a hard guarantee of no games at a venue on a date, use `BLOCKED_GAMES` instead.

**Default weight:** `PENALTY_WEIGHTS['preferred_weekends_away_ground']` (default `1000`). Per-entry `weight` key overrides this.

**2026 config:** The 6 NRL-Knights-at-Maitland dates are pre-configured in `config/season_2026.py::PREFERRED_WEEKENDS` as `avoid` entries for `field_location: 'Maitland Park'`.

---

## CLUB_DAYS (Special Events)

```python
CLUB_DAYS = {
    'Crusaders': datetime(2026, 6, 14),  # All teams play back-to-back, same field
}
```

**Effect:** ClubDayConstraint ensures all club teams play on that date, on same field, in contiguous slots.

---

## PREFERRED_GAMES (spec-020)

Soft, weighted analogue of `FORCED_GAMES`: same scope/team/club grammar **plus**
an optional `weight`, but a penalty-on-deviation instead of a hard constraint. A
preference that can't be met costs objective, never feasibility. Replaces the
deleted `PHL_PREFERENCES` / `PreferredDates`.

```python
PREFERRED_GAMES = [
    # Softly prefer exactly one PHL game on a marquee date (the old
    # PreferredDates behaviour, migrated):
    {'grade': 'PHL', 'date': '2026-04-19', 'constraint': 'equal', 'count': 1,
     'weight': 10000, 'description': 'marquee PHL date'},
    # Softly prefer at most 2 games at a venue on a date:
    {'field_location': 'Maitland Park', 'date': '2026-05-03',
     'constraint': 'lesse', 'count': 2},
]
```

**Penalty per `constraint` type** (target `N = count`):

| `constraint` | penalty |
|---|---|
| `equal` | `\|sum − N\|` |
| `lesse` | `max(0, sum − N)` |
| `greatere` | `max(0, N − sum)` |
| `greater` | `max(0, (N+1) − sum)` |
| `less` | `max(0, sum − (N−1))` |

**Default weight:** `PENALTY_WEIGHTS['preferred_games']` (default `10000`). A
single shared bucket holds all entries; an optional per-entry `weight` is a
multiplier on top of the default (`max(1, weight // default)`), used only when
one preference must out-rank the rest.

**Non-fatal:** an empty / zero-candidate scope produces 0 penalty + a logged
warning (never `sys.exit`). For a hard guarantee, use `FORCED_GAMES` instead.

See `docs/system/FORCED_GAMES_AS_COUNT_RULES.md` for the full grammar and
implementation notes.

---

## SPECIAL_GAMES

```python
SPECIAL_GAMES = {
    'taree_game': {
        'teams': ['Tigers', 'Souths'],
        'grades': ['PHL', '2nd'],
        'month': 5,  # May
        'date': None,  # TBC
    },
}
```

**Note:** Currently informational only. Actual scheduling requires manual intervention or constraint addition.

---

## FORCED_GAMES (Partial Key Matching)

Collects all variables matching a partial key specification and applies a constraint (`sum == 1` by default). Variables that DON'T match the scope are left alone — they are NOT eliminated.

```python
FORCED_GAMES = [
    # Each entry specifies scope fields to match against decision variable keys.
    # All matching variables are collected, then sum(matching) is constrained.

    # Example: Force exact matchup on a specific Friday (sum == 1)
    {
        'teams': ['Norths', 'Maitland'],
        'grade': 'PHL',
        'date': '2026-05-08',
        'day': 'Friday',
        'field_location': 'Newcastle International Hockey Centre',
        'description': 'NIHC Friday Night - Norths vs Maitland',
    },

    # Example: Scope-only (no teams) — force any game at this venue/date
    {
        'grade': 'PHL',
        'date': '2026-03-27',
        'day': 'Friday',
        'field_location': 'Central Coast Hockey Park',
        'description': 'Gosford Friday Night - Mar 27',
    },

    # Example: At most 1 game (constraint: 'lesse' for sum <= 1)
    {
        'grade': 'PHL',
        'date': '2026-05-17',
        'day': 'Sunday',
        'field_location': 'Newcastle International Hockey Centre',
        'constraint': 'lesse',
        'description': 'Masters SC weekend - max 1 PHL game at NIHC',
    },
]
```

### Field Reference

| Field | Type | Maps to Key Index | Description |
|-------|------|-------------------|-------------|
| `teams` | list | 0, 1 (team1, team2) | 1-2 club names. Auto-resolved to full team names using grade. Optional — omit to match all teams. |
| `grade` | str/list | 2 | Grade name(s). Also used to resolve club names to team names. |
| `day` | str | 3 | Day of week (e.g., 'Friday', 'Sunday') |
| `day_slot` | int | 4 | Time slot index |
| `time` | str | 5 | Game time (e.g., '19:00') |
| `week` | int | 6 | Week number |
| `date` | str | 7 | Date string 'YYYY-MM-DD' |
| `round_no` | int | 8 | Round number |
| `field_name` | str | 9 | Field name (e.g., 'EF', 'WF') |
| `field_location` | str | 10 | Venue name |
| `constraint` | str | — | Equality type: `'equal'` (default, sum==1), `'lesse'` (<=1), `'greatere'` (>=1), `'greater'` (>1), `'less'` (<1) |
| `description` | str | — | Logging only, not used for matching |

### How It Works

1. **Scope** = all specified scope fields (grade, date, day, venue, etc.)
2. **Team matchers** = teams resolved to full names (or `('all',)` if no teams specified)
3. All variables matching scope + team matcher are collected into a group
4. Constraint applied: `model.Add(sum(group) <op> 1)` where `<op>` is determined by `constraint` field
5. If a forced game scope matches **zero** variables, the solver exits with a diagnostic error

### Team Name Resolution

Club names in `teams` are auto-resolved to full team names:
- `'Maitland'` + `grade='PHL'` → `'Maitland PHL'`
- `'Colts'` + `grade='5th'` → `['Colts Gold 5th', 'Colts Green 5th']` (both teams match)
- No `teams` or `club` → matches ALL teams in scope

### Implementation

- Config: `FORCED_GAMES` list in `config/season_{year}.py`
- Passed through: `SEASON_CONFIG['forced_games']` → `build_season_data()` → `data['forced_games']`
- Filtering: `_build_forced_game_rules()` and `_check_forced_game_status()` in `utils.py`
- Called from: `generate_X()` — checked for each variable before creation
- Recorded in: draw JSON `metadata.forced_games` + `metadata.forced_game_outcomes` (with `satisfied` flag)

---

## BLOCKED_GAMES (No-Play Variable Elimination)

Eliminates variables matching scope + team matchers. If no teams/club specified, blocks ALL variables matching the scope.

```python
BLOCKED_GAMES = [
    # Block a specific club+grade on a date
    {
        'club': 'Crusaders',
        'grade': '6th',
        'date': '2026-06-28',
        'description': 'Crusaders 6th - NSW Masters at Tamworth',
    },
    # Block multiple grades for a club
    {
        'club': 'Souths',
        'grades': ['PHL', '2nd'],
        'date': '2026-05-24',
        'description': 'Souths PHL/2nd - U18 State Championships',
    },
    # Block ALL variables matching scope (no teams/club = block everything)
    {
        'grade': '2nd',
        'date': '2026-05-17',
        'field_location': 'Maitland Park',
        'description': 'Masters SC weekend - Maitland PHL only',
    },
]
```

### Field Reference

Same scope fields as `FORCED_GAMES` (see above), plus:

| Field | Type | Description |
|-------|------|-------------|
| `club` | str | Club name — resolved to all matching team names. Use instead of `teams` for club-wide blocks. |
| `grades` | list | Multiple grades — used with `club` to resolve specific graded teams. |

### How It Works

1. **Scope** = all specified scope fields (grade, date, day, field_location, etc.)
2. **Team matchers** = club/teams resolved to full team names. If neither specified, matcher list is empty.
3. For each variable in `generate_X()`:
   - If it matches ALL scope fields AND (matches a team matcher OR no team matchers exist) → **eliminated**
   - Otherwise → unaffected

### Implementation

- Config: `BLOCKED_GAMES` list in `config/season_{year}.py`
- Passed through: `SEASON_CONFIG['blocked_games']` → `build_season_data()` → `data['blocked_games']`
- Filtering: `_build_blocked_game_rules()` and `_is_blocked_by_no_play()` in `utils.py`
- Called from: `generate_X()` — checked for each variable after forced games check
- Recorded in: draw JSON `metadata.blocked_games` + `metadata.blocked_game_outcomes` (with `respected` flag)

---

## LOCKED_PAIRINGS (Date-Pin Config, spec-025)

Pins a pairing to its **date** (playing day) with `sum == 1`, leaving the time, day_slot, and field entirely FREE for the solver. It is the sister config to `FORCED_GAMES` for mechanical date-pins: after a draw is published and some pairings are decided but times/fields may need adjusting, `LOCKED_PAIRINGS` holds those decided pairings so a regeneration run (spec-026) keeps them on their weekends without re-opening the matchup choices.

> **Rule:** if all you want is "this pairing, this weekend, any time" — use `LOCKED_PAIRINGS`, not `FORCED_GAMES`. Reserve `FORCED_GAMES` for count rules, marquee games that also lock a venue/time, or any entry that uses `count`, `constraint`, `time`, `day_slot`, `field_name`, `field_location`, or `day`.

```python
LOCKED_PAIRINGS = [
    # Each entry pins a pairing to a date; time/slot/field are chosen by the solver.
    {
        'teams': ['Norths', 'Maitland'],  # or team1/team2 keys
        'grade': 'PHL',
        'date': '2026-05-08',             # YYYY-MM-DD
        'description': 'Locked wk7 — Norths vs Maitland PHL',  # optional
    },
    {
        'team1': 'Gosford',
        'team2': 'Tigers',
        'grade': '2nd',
        'date': '2026-04-12',
    },
]
```

### Allowed fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `teams` | list | one of `teams` / `team1`+`team2` | Two club or team names. Resolved the same way as FORCED_GAMES `teams`. |
| `team1` / `team2` | str | (alternative to `teams`) | Specify the two teams individually. |
| `grade` | str | yes | Grade name (e.g. `'PHL'`, `'2nd'`). Used for team-name resolution. |
| `date` | str | yes | Date string `'YYYY-MM-DD'`. |
| `description` | str | no | Human-readable note; ignored by solver. |

### Forbidden fields — FATAL if present

The following keys are **not permitted** in a `LOCKED_PAIRINGS` entry. Presence of any is a validation FATAL (they would partially re-lock the time or slot, defeating the config's purpose):

`time`, `day_slot`, `field_name`, `field_location`, `day`, `week`, `round_no`, `count`, `constraint`

### Behaviour

- **Enforcement:** `generate_X` builds a second scope-count pass parallel to FORCED. Each pin results in `model.Add(sum(candidate_vars_on_date) == 1)`. The sum covers all valid times/slots/fields for the pairing on that date; the solver picks exactly one.
- **Separate scope dict:** `locked_pairing_scope_vars` is kept separate from `forced_scope_vars`. A variable may satisfy both a FORCED scope and a LOCKED_PAIRINGS scope; both constraints register it and both apply (additive, no elimination).
- **Empty scope = FATAL:** a pin whose pairing has zero candidate variables on its date exits with a `LOCKED_PAIRINGS`-labelled diagnostic naming the `(teams, grade, date)`.
- **Locked-week filtering:** entries whose `date` falls in a hard-locked week are stripped before the solver runs (those weeks are already pinned by the locked-weeks machinery; double-applying the pin would be redundant and could conflict).
- **Metadata:** draw JSON gains a `locked_pairing_outcomes` section (per pin: matched-var count, resolved time/slot/field, `satisfied: true/false`) alongside `forced_game_outcomes`.
- **Tester:** `analytics/tester.py::_check_locked_pairings` (registered in `constraints/registry.py` as `LockedPairings`, `tester_only`, severity 1) flags any pinned pairing not present on its date in the finished draw.

### Implementation

- Config: `LOCKED_PAIRINGS` list in `config/season_{year}.py` (default `[]` in `config/defaults.py`)
- Passed through: `SEASON_CONFIG['locked_pairings']` → `build_season_data()` → `data['locked_pairings']`
- Enforcement: `generate_X()` — second scope-count pass after FORCED, using `_build_scope_count_rules(..., label='LOCKED_PAIRINGS', unique_per_entry=True)`
- Validation: `validate_game_config` — forbidden-field check + locked-week filter + empty-scope pre-check
- Recorded in: draw JSON `metadata.locked_pairing_outcomes` (with `satisfied` flag)
- Regen tooling: spec-026 (`--regen-from`) reads and writes to this list automatically
