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

### 3. `MaitlandHomeGrouping` (→ `NonDefaultHomeGrouping` in Phase 6)

**Problem:** if FORCED forces a Maitland home weekend into week W, the
sliding-window consecutive-home-weeks calculation needs to account for it
already being a 1.

**Proposed formula:**
```
forced_home_weeks_per_club = {club: set(weeks)}
# Atom uses this to clamp specific home indicators to 1 instead of letting
# the solver choose. Sliding-window math still works on top.
```

### 4. `AwayAtMaitlandGrouping` (→ `AwayAtNonDefaultGrouping`)

**Problem:** if FORCED forces an away match at Maitland for club C in
week W, the per-week away-clubs count for that week is at least 1.

**Proposed formula:**
```
forced_away_per_week = {(week, venue): set of away clubs forced there}
# Atom uses this to baseline the per-week count.
```

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
| 1 | `EqualGames` | ✅ approved: **no adjuster** — formula above explains why |
| 2 | `EqualMatchUpSpacing` | 🟡 approved spec — implement |
| 3 | `MaitlandHomeGrouping` (`NonDefaultHomeGrouping`) | 🟡 approved spec — implement (will rename in Phase 6) |
| 4 | `AwayAtMaitlandGrouping` (`AwayAtNonDefaultGrouping`) | 🟡 approved spec — implement (will rename in Phase 6) |
| 5 | `ClubVsClubCoincidence` | 🟡 approved spec — implement (the user's worked example) |

Each adjuster ships as one commit on `final-form` with: (a) the adjuster
callable assigned to `CONSTRAINT_REGISTRY[name].forced_blocked_adjuster`, (b)
the atom change to read `data['count_adjustments'][canonical_name]`, (c)
unit tests covering both the adjuster math (synthesised FORCED/BLOCKED) and
the atom's behaviour change. The two Phase-6 renames mean adjusters #3 and
#4 should be implemented against the new `NonDefaultHomeGrouping` /
`AwayAtNonDefaultGrouping` atoms, not the legacy names.

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
