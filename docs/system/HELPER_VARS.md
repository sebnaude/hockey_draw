# Helper-Variable Registry

Atoms and the engine create the shared helper variables they need via `HelperVarRegistry` instead of hand-rolling them. The registry deduplicates by pool key so multiple callers asking for the same helper share one variable.

Defined in `constraints/helper_vars.py`. Wired into `UnifiedConstraintEngine` as `self.registry` (and aliased as `self.pool` for legacy code).

There is a **single** pathway: pool-style `get_or_create_*`. (spec-022 removed the vestigial declarative `declare`/`freeze`/`get_declared` API — it had zero production callsites, and a second store risked two CP-SAT variables for one concept.)

## Usage (pool-style)

```python
ind = registry.get_or_create_bool(key, vars_list, label)        # max-equality channeling
ind = registry.get_or_create_presence(key, vars_list, label)    # BoolOr/BoolAnd channeling
registry.register(key, custom_var)
v = registry.get(key)        # cache lookup; None if missing
v = registry.lookup(key)     # same as .get()
```

The cache is keyed by the raw pool `key`. **Key convention:** a shared helper's pool key is `(kind, *discriminators)` with `kind` ∈ `HELPER_VAR_CATALOG` — keeping `kind` first keeps the catalog meaningful and lets two callers asking for the same logical helper land on the same cache entry. Both the engine's `_*_hard()` / `_*_soft()` methods and new atoms (inside `apply()`) use this same path.

## Helper-var catalog

The set of allowed `kind` strings is `HELPER_VAR_CATALOG` in `constraints/registry.py`. An atom's `ConstraintInfo.required_helpers` is validated against this set at registry-load time (via `validate_required_helpers()`), so a typo fails immediately.

Current catalog (extend as Phase 3 atoms need new kinds):

| Kind | Key shape | Built by | Used by (planned atoms) |
|---|---|---|---|
| `is_slot_used` | `(week, day, location, day_slot)` | sums vars at that slot, ≥1 ⇒ 1 | `EnsureBestTimeslotChoices`, `ClubGameSpread` |
| `is_field_used` | `(week, day, location, field, day_slot)` | sums vars at that field+slot, ≥1 ⇒ 1 | `EnsureBestTimeslotChoices` |
| `weekend_used` | `(week, location)` | sums all venue vars in week | `EnsureBestTimeslotChoices` |
| `team_plays_in_week` | `(team, week)` | sum of team's vars in that week | `NoDoubleBookingTeams`, adjacency atoms |
| `pair_plays_in_week` | `(team1, team2, week, grade)` | OR over all timeslot vars | `EqualMatchUpSpacing`, `ClubVsClubAlignment` |
| `pair_plays_on_date` | `(team1, team2, date, grade)` | OR over date's timeslot vars | new `ClubVsClubCoincidence` atom |
| `club_plays_on_date_at_field` | `(club, date, field)` | OR over teams from club at that field | `ClubDaySameField` atom |
| `club_grade_in_slot` | `(club, grade, week, day, day_slot)` | OR over teams of (club,grade) in slot | `ClubGradeAdjacency` |
| `home_game` | `(team, week, opponent)` | indicator: team played at home | `AwayClubPerOpponentAndAggregateHomeBalance`, `AwayClubHomeWeekendsCount` (the `NonDefaultHomeGrouping` consumer was removed in spec-018) |
| `is_phl_friday` | `(week,)` | indicator: PHL plays Friday this week | PHL Friday count atoms |
| `club_day_field_used` | `(club, field_name)` | indicator: any club_day game on field_name | `ClubDaySameField` |
| `club_day_slot_used` | `(club, day_slot)` | indicator: any club_day game at day_slot | `ClubDayContiguousSlots` |
| `venue_slot_used` | `(week, date, location, day_slot)` | OR across a venue's fields at that slot | `VenueEarliestSlotFill` (spec-021) |
| `club_spread_slot_used` | `(club, week, day, field, day_slot)` | OR over a club's games at that field+slot | `ClubGameSpread` per-field contiguity (spec-024) |
| `phl_2nd_btb_pair` | `(clubs, round_no, field, slot1, slot2)` | back-to-back same-field indicator | `PHLAnd2ndBackToBackSameField` |

## Shared contiguity primitive (spec-021)

`constraints/atoms/_contiguity.py` factors out the `slot_used`-indicator pattern used by every "don't leave holes between used timeslots" rule, so the atoms share one cheap encoding without merging (GOALS §2 — extract a helper, don't merge):

- `slot_used_indicators(registry, vars_by_slot, kind, *key_prefix)` — channels one `slot_used` BoolVar per slot via the pool API (`get_or_create_bool`), keyed `(kind, *key_prefix, slot)`.
- `enforce_no_gaps(model, slot_inds)` — **floating**: the used block is contiguous but may start anywhere (`prev + next ≤ 1` when the middle slot is empty). Used by `ClubDayContiguousSlots` and `ClubGameSpread`.
- `enforce_monotone_fill(model, slot_inds)` — **anchored**: use slot `s` ⇒ use slot `s-1`, so games pack into the *earliest* slots (strictly stronger than no-gaps). Used by `VenueEarliestSlotFill`.

Both reduce to an implication/coincidence chain over BoolVar indicators — no `AddDivisionEquality` / range / min-max IntVars.

## API reference

```python
from constraints.helper_vars import HelperVarRegistry

registry = HelperVarRegistry(model)

# Pool-style (the single pathway)
var = registry.get_or_create_bool(key, vars_list, label)
var = registry.get_or_create_presence(key, vars_list, label)
registry.register(key, custom_var)
maybe_var = registry.get(key)          # cache lookup, None if missing
maybe_var = registry.lookup(key)       # alias for .get()

# Diagnostics
registry.diagnostics()                 # dict: pool_created, pool_hits, created, hits, pool_size
```

## Rules

1. **Create/look-up helpers in `apply()`** via `get_or_create_bool` / `get_or_create_presence` / `register`. The first caller for a key builds and channels the var; later callers with the same key get it back from the cache.
2. **Re-requesting the same pool key is a cache hit** — first var wins. Stat counter `pool_hits` tracks this.
3. **`kind` is part of the pool key** — by convention the first element. Distinct kinds with the same discriminators get distinct helpers; same `(kind, *discriminators)` share one.
4. **`required_helpers` on `ConstraintInfo` must reference catalog kinds.** `validate_required_helpers()` returns offending pairs.
5. **Don't reach past the registry** — never call `model.NewBoolVar` and `AddMaxEquality` directly inside an atom for a helper that another atom might want. Ask the registry via the pool-style API.

## Tests

- `tests/test_helper_var_registry.py` — pool-style API (dedup, empty-list-forces-zero, max/presence channeling, register/lookup), the single-pathway guard (no `declare`/`freeze`/`get_declared`/`HelperVar`/`Atom.declare_helpers`), diagnostics shape, and the `required_helpers` ∈ catalog key-convention check.

## `SharedVariablePool` alias

`SharedVariablePool` is a `from constraints.helper_vars import SharedVariablePool` alias for `HelperVarRegistry`. Existing call sites (`pool.get_or_create_bool`, `pool.register`, `pool.get`) keep working. New code uses `HelperVarRegistry` directly via the pool-style API.
