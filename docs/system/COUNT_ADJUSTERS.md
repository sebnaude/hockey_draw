# FORCED/BLOCKED Count Adjusters (Phase 4)

> **Status:** framework shipped (commit `1521c9b`). The 5 adjuster formulas
> below are **approved spec** — implement them as written. Flag deviations
> only if you find a real bug in the formula while reading the related atom
> code. Implementation order is suggested, not strict.

## What it is

Some constraints care about expected *counts* of things — pair meetings per
season, home weekends per club, games per team. When `FORCED_GAMES` or
`BLOCKED_GAMES` removes flexibility (forces or eliminates specific
variables), those expected counts can change.

**The adjuster mechanism:** every `ConstraintInfo` can register a
`forced_blocked_adjuster` callable. The engine runs every adjuster once
during `UnifiedConstraintEngine.build_groupings()` (after FORCED/BLOCKED have
been parsed but before atom `apply()` runs), and stashes the result under
`data['count_adjustments'][canonical_name]`. Atoms read their adjustment by
canonical name.

```python
def adjuster(data: dict, forced_games: list, blocked_games: list) -> dict | None:
    """Return adjustment metadata keyed however the constraint cares.

    Return value is stored at data['count_adjustments'][canonical_name].
    Returning None signals "no adjustment needed".
    """
```

The framework is **generic** — it doesn't care what an adjuster does. Each
constraint that needs FORCED/BLOCKED awareness declares its own adjuster.

## API

```python
from constraints.registry import (
    CONSTRAINT_REGISTRY, ConstraintInfo, run_count_adjusters,
)

# Register on an existing entry:
CONSTRAINT_REGISTRY['EqualGames'].forced_blocked_adjuster = my_adjuster

# Engine dispatches automatically — atoms just read:
data['count_adjustments']['EqualGames']  # whatever my_adjuster returned
```

`run_count_adjusters(data)` is also exposed for tests; the engine calls it
exactly once.

## Adjuster formulas (approved spec)

These are the catalog from `ATOMIZATION_PLAN.md` Phase 4 + the user's
example use cases. Each formula is the **proposed** behaviour; the actual
implementations land only after sign-off.

### 1. `EqualGames` (split out from `EqualGamesAndBalanceMatchUps`)

**Problem:** if a team has FORCED entries pinning it into N specific games,
the `sum(team_vars) == num_rounds[grade]` constraint is unaffected (FORCED
just sets some terms to 1; the sum is unchanged).

**Likely conclusion:** **no adjuster needed.** The legacy formula is
self-consistent. Flag here for explicit confirmation.

### 2. `EqualMatchUpSpacing`

**Problem:** if `FORCED_GAMES` pins matchup `(t1, t2, grade)` into N
specific weeks, the spacing budget for that pair is reduced (fewer free
rounds in which to space the remaining T-2-N rounds).

**Proposed formula:**
```
forced_rounds_per_pair = {(t1,t2,grade): set of forced rounds}
# Atom uses this to skip pairs whose flexibility is exhausted, or to
# tighten the min_gap calculation:
effective_T = T - len(forced_rounds_per_pair[pair])
```

#### spec-008 note — intuitive gap semantics

As of spec-008 (Part A) the input number `S` for `EqualMatchUpSpacing`
means **"S played rounds between meetings"** (not the old "calendar
distance" interpretation). The hard rule now forbids `gap = r2 - r1 <= S`
(equivalently `r2 - r1 - 1 < S`), not the old `gap < min_gap`.

The math is unified in `constraints/atoms/_spacing.py`:

- `ideal_gap(T) = legacy_min_gap(T) - 1` — the default S for grade size T.
- `effective_spacing(T, base_slack, config_slack)` — clamped at 0, used
  by both the solver (`unified.py::_matchup_spacing_hard`) and the
  tester (`analytics/tester.py::_check_equal_matchup_spacing`).
- `CONSTRAINT_DEFAULTS['spacing_base_slack']` reduces S by `base_slack`
  rounds; CLI `--slack N` (via `data['constraint_slack'][...]`) reduces
  by another `N`. Each unit of slack lets the gap shrink by one round
  from ideal; the physical schedule a healthy solver produces at default
  slack is **unchanged** (forbidden-gaps set is identical).
- Each FORCED meeting in `forced_rounds_per_pair[pair]` surfaces as
  **negative net slack** to `effective_spacing` so S grows above
  `ideal_gap(T)` — i.e. tighter — rather than relaxing it.

Part B adds a sibling atom `BalancedByeSpacing` with its own helper
`ideal_bye_gap(R, byes)` and its own slack key (`bye_spacing_base_slack`
default + `constraint_slack['BalancedByeSpacing']` from `--slack N`).
The two atoms are independent dimensions and share `_spacing.py` only
for the shape of the math.

### 3. `MaitlandHomeGrouping` (→ `NonDefaultHomeGrouping`) — REMOVED (spec-018)

The home-grouping constraint and its `maitland_home_grouping_adjuster` were
**deleted in spec-018** — the convenor no longer wants any consecutive-home-
weekend sequencing. This adjuster no longer exists.

### 4. `AwayAtMaitlandGrouping` (→ `AwayAtNonDefaultGrouping`) — REMOVED (spec-018)

The away-clubs-per-weekend constraint and its
`away_at_maitland_grouping_adjuster` were **deleted in spec-018**. This
adjuster no longer exists.

### 5. `ClubVsClubCoincidence` (the user's worked example)

**Problem:** ClubVsClubCoincidence counts how many rounds a club-pair
coincides on across grades. If FORCED forces 2 PHL games of (Maitland,
Norths) onto Friday Gosford (so NOT on Sunday), then for the lower-grade
alignment block (which only sees Sunday games) only `total_meetings - 2`
PHL games of that pair exist on Sunday. The expected coincidences should
match that reduced number, not the full season total.

**Proposed formula (per the plan):**
```
expected_meetings[grade][club_pair] = (
    total_meetings_in_grade
    - count_forced_off_sunday_for_pair
    - count_blocked_on_sunday_for_pair
)
# Atom uses expected_meetings instead of num_games when computing
# min_required = max(0, expected_meetings - slack).
```

This is the most subtle one. The user flagged it as the canonical
motivating example for the whole adjuster mechanism.

## Implementation status

| # | Adjuster | Status |
|---|---|---|
| 1 | `EqualGames` | ✅ no adjuster needed — `sum(team_vars) == num_rounds` is unaffected by FORCED entries (FORCED just pins terms to 1; the sum is unchanged). |
| 2 | `EqualMatchUpSpacing` | ✅ shipped — `equal_matchup_spacing_adjuster` in `constraints/atoms/_adjusters.py`. Returns `{(t1, t2, grade): set(forced_rounds)}`. `_matchup_spacing_hard` reads it to tighten `min_gap`. |
| 3 | `MaitlandHomeGrouping` (→ `NonDefaultHomeGrouping`) | ❌ **REMOVED (spec-018)** — constraint deleted; `maitland_home_grouping_adjuster` no longer exists. |
| 4 | `AwayAtMaitlandGrouping` (→ `AwayAtNonDefaultGrouping`) | ❌ **REMOVED (spec-018)** — constraint deleted; `away_at_maitland_grouping_adjuster` no longer exists. |
| 5 | `ClubVsClubCoincidence` | ✅ shipped — `club_vs_club_coincidence_adjuster` in `constraints/atoms/club_vs_club_coincidence.py`. Returns `{grade: {club_pair: expected_meetings}}`, reducing for FORCED off-Sunday and BLOCKED on-Sunday entries. The atom reads `expected` instead of `num_games` when computing `min_required = max(0, expected - slack)`. **Verified end-to-end 2026-05-18 (spec-009)** — `tests/atoms/test_cvc_coincidence_phl_friday_adjuster.py`: adjuster math, engine dispatch via `run_count_adjusters`, and atom feasibility all confirmed correct. |

Each adjuster ships as one commit on `final-form` with: (a) the adjuster
callable assigned to `CONSTRAINT_REGISTRY[name].forced_blocked_adjuster`, (b)
the atom change to read `data['count_adjustments'][canonical_name]`, (c)
unit tests covering both the adjuster math (synthesised FORCED/BLOCKED) and
the atom's behaviour change. (Adjusters #3 and #4 were removed in spec-018
along with the home-grouping constraints they served.)

**Where to wire it:** the adjuster registers on the `ConstraintInfo` for the
*atom* canonical name (e.g. `ClubVsClubCoincidence`, not the parent
`ClubVsClubAlignment`). Atoms read `data['count_adjustments'][self.canonical_name]`
inside their `apply()`.

**Test pattern:**

```python
def test_adjuster_reduces_expected_meetings(self):
    data = {...synthetic data...}
    data['forced_games'] = [{
        'grade': 'PHL', 'day': 'Friday', 'field_location': GOSFORD,
        'teams': ['Maitland PHL', 'Norths PHL'], 'count': 2, 'constraint': 'equal',
    }]
    data['blocked_games'] = []
    out = my_adjuster(data, data['forced_games'], data['blocked_games'])
    assert out[('Maitland', 'Norths')]['PHL'] == total_meetings - 2
```

## 6. Per-pair FORCED-Friday helper + AwayClub home-Sunday derived range (spec-005 + spec-037)

`ClubVsClubStackedWeekends` (spec-005) and `AwayClubHomeWeekendsCount`
(spec-004, redesigned in spec-037) consume a shared pure-function helper
module: `constraints/atoms/_phl_forced_friday_helper.py`. The helper avoids
the `forced_blocked_adjuster` callback because it needs to be queryable from
multiple call sites WITHOUT pre-running an engine-managed dispatch step.

The helper now exports three functions:

```
phl_forced_friday_meetings(data, club_a, club_b) -> int     # spec-005
    Number of PHL Friday games FORCED to be played between this SPECIFIC pair
    of clubs. Umbrella scopes (no `team1/team2/teams` filter naming both
    clubs) contribute 0 — they don't guarantee any Friday is BETWEEN this
    pair, they only guarantee a total per-club. The per-pair helper
    deliberately UNDER-counts (only credits entries whose scope names both
    clubs) so spec-005's PHL Sunday budget
    `total_pair_meetings - phl_forced_friday_meetings` is a LOWER bound on
    Sunday meetings — the solver may schedule MORE Sundays for the pair, but
    not fewer.

away_club_min_sundays_home(data, club) -> int                # spec-037
    max(num_rounds[g] // 2 for g in non-PHL grades the club fields),
    or 0 if the club fields no non-PHL grade.

    The floor of the derived Sunday-home weekend range for an away-based
    club. Non-PHL grades have no Friday alternative — every home game must
    land on a Sunday — so this is the strict minimum Sunday-home count.

away_club_max_sundays_home(data, club) -> int                # spec-037
    max(upper(g) for g in ALL grades the club fields),
    where upper(g) = (num_rounds['PHL'] + 1) // 2 if g == 'PHL'
                   = num_rounds[g] // 2           otherwise,
    or 0 if the club has no teams.

    The ceiling of the derived Sunday-home weekend range. PHL can lend
    Sundays to forced Fridays without pushing the total above its raw home
    count, so the ceiling caps at the max across all grades.
```

### spec-005 PHL Sunday budget formula (unchanged)

`ClubVsClubStackedWeekends` uses `phl_forced_friday_meetings` to compute
the **per-pair** Sunday budget for PHL stacking:

```
pair_grade_sunday_meetings(data, pair, 'PHL')
    = total_phl_meetings(pair) - phl_forced_friday_meetings(data, *pair)

pair_grade_sunday_meetings(data, pair, other_grade)
    = total_meetings(pair, other_grade)   # other grades don't play Friday
```

where `total_phl_meetings(pair)` comes from
`per_pair_grade_meeting_counts(data, pair)['PHL']` (= matchups × per_matchup,
where per_matchup = `num_rounds['PHL'] // (T-1)` for even T or `// T` for
odd T). The atom pins `sum_w play[PHL, w] for Sunday vars == sunday_budget`
HARD.

### Why per-pair (NOT per-club) for the umbrella + per-pair entry case

A convenor with these two FORCED entries:

```python
{'grade': 'PHL', 'day': 'Friday', 'club': 'Maitland',
 'count': 2, 'constraint': 'equal'},  # umbrella: 2 Maitland Fridays anywhere
{'grade': 'PHL', 'day': 'Friday',
 'teams': ['Maitland', 'Tigers']},   # per-pair: 1 Mait-vs-Tigers Friday
```

has committed to **2 Maitland Friday PHL games total** (not 3). The
per-pair entry forces one specific matchup; that matchup is one of the
umbrella's two. `phl_forced_friday_meetings(data, 'Maitland', 'Tigers')`
returns 1 — only the per-pair entry credits the (Maitland, Tigers) pair,
since the umbrella doesn't guarantee any Tigers-vs-Maitland Friday. (The
spec-037 redesign of `AwayClubHomeWeekendsCount` no longer reads FORCED
config; FORCED_GAMES carries the per-club Friday count instead.)

### spec-037 AwayClub home-Sunday derived range

`AwayClubHomeWeekendsCount` (spec-004, redesigned in spec-037) enforces a
single two-sided range on the number of Sunday-home weekends for each
away-based club:

```
min_sundays_home(data, club) <= sum(sunday_home_indicators) <= max_sundays_home(data, club)
```

The atom previously pinned THREE hard equalities (Friday count, Sunday
count, total weekends) and read FORCED_GAMES to compute the Friday side.
That coupled two solver mechanisms encoding the same fact (FORCED_GAMES
already enforces "exactly N Maitland-Park Fridays" via partial-key count
entries) and over-constrained the model on forced-free configs. The
redesign drops the Friday-side enforcement entirely and clamps Sunday-home
to a derived range that PHL forced Fridays naturally fit beneath.

**Worked example — Maitland on `season_test` (forced-free):**

`num_rounds`: PHL=20, 3rd=18, 4th=18, 5th=16, 6th=18. Maitland fields
PHL/3rd/4th/5th/6th.

| Grade | Home games (`num_rounds // 2`, PHL uses ceil) | Must be Sunday? |
|---|---|---|
| PHL | 10 (ceil(20/2)) | No (PHL has a Friday option) |
| 3rd | 9 | Yes |
| 4th | 9 | Yes |
| 5th | 8 | Yes |
| 6th | 9 | Yes |

- `away_club_min_sundays_home = max(9, 9, 8, 9) = 9` (max across non-PHL — the floor)
- `away_club_max_sundays_home = max(10, 9, 9, 8, 9) = 10` (max across all incl. PHL — the ceiling)

With 0 forced Fridays, PHL's 10 home games land on 10 Sundays → total = 10.
With 3 forced Fridays at MP via `FORCED_GAMES`, PHL contributes 7 home
Sundays + 3 home Fridays → distinct home-Sunday weeks = max(9, 7) = 9 (the
non-PHL floor binds). Both inside `[9, 10]`. ✓

**Gosford special case** (PHL-only club): `min = 0` (no non-PHL grades),
`max = 10`. The atom skips the lower bound when min == 0 (vacuous) and
emits only `sum <= 10`. Forced Fridays from FORCED_GAMES (e.g. 8 PHL
Fridays at CCHP) reduce PHL Sunday demand to 2; the atom still allows up
to 10, no conflict.
