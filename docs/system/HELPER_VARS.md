# Helper-Variable Registry

Atoms (Phase 3+) declare the helper variables they need via `HelperVarRegistry` instead of hand-rolling them. The registry deduplicates by `(kind, key)` so multiple atoms asking for the same helper share one variable.

Defined in `constraints/helper_vars.py`. Wired into `UnifiedConstraintEngine` as `self.registry` (and aliased as `self.pool` for legacy code).

## Two usage modes

### 1. Declarative (atoms — preferred)

```python
def declare_helpers(self, registry, data):
    for week, day, location, slot in self._needed_slots(data):
        registry.declare(
            kind='is_slot_used',
            key=(week, day, location, slot),
            builder=lambda m, X, d, w=week, day=day, loc=location, s=slot: (
                _build_slot_used_indicator(m, X, w, day, loc, s)
            ),
            description=f'slot {slot} at {location} on week {week} {day}',
        )

def apply(self, model, X, data, registry):
    for week, day, location, slot in self._needed_slots(data):
        ind = registry.get_declared('is_slot_used', (week, day, location, slot))
        # use ind as a normal CP-SAT BoolVar
```

The engine flow:
1. Each atom's `declare_helpers()` runs — registry collects intents.
2. `registry.freeze(X, data)` runs once — every distinct helper var is built exactly once.
3. Each atom's `apply()` runs and calls `registry.get_declared(kind, key)` to fetch.

### 2. Pool-style (legacy, retained for back-compat)

```python
ind = registry.get_or_create_bool(key, vars_list, label)        # max-equality channeling
ind = registry.get_or_create_presence(key, vars_list, label)    # BoolOr/BoolAnd channeling
registry.register(key, custom_var)
v = registry.get(key)        # cache lookup; None if missing
v = registry.lookup(key)     # same as .get()
```

Pool-style cache is keyed by raw `key` (no `kind` discriminator). The legacy `UnifiedConstraintEngine._*_hard()` / `_*_soft()` methods use this path; new atoms must use the declarative path.

## Helper-var catalog

The set of allowed `kind` strings is `HELPER_VAR_CATALOG` in `constraints/registry.py`. An atom's `ConstraintInfo.required_helpers` is validated against this set at registry-load time (via `validate_required_helpers()`), so a typo fails immediately.

Current catalog (extend as Phase 3 atoms need new kinds):

| Kind | Key shape | Built by | Used by (planned atoms) |
|---|---|---|---|
| `is_slot_used` | `(week, day, location, day_slot)` | sums vars at that slot, ≥1 ⇒ 1 | `EnsureBestTimeslotChoices`, `ClubGameSpread` |
| `is_field_used` | `(week, day, location, field, day_slot)` | sums vars at that field+slot, ≥1 ⇒ 1 | `EnsureBestTimeslotChoices`, `MinimiseClubsOnAFieldBroadmeadow` |
| `weekend_used` | `(week, location)` | sums all venue vars in week | `EnsureBestTimeslotChoices` |
| `team_plays_in_week` | `(team, week)` | sum of team's vars in that week | `NoDoubleBookingTeams`, adjacency atoms |
| `pair_plays_in_week` | `(team1, team2, week, grade)` | OR over all timeslot vars | `EqualMatchUpSpacing`, `ClubVsClubAlignment` |
| `pair_plays_on_date` | `(team1, team2, date, grade)` | OR over date's timeslot vars | new `ClubVsClubCoincidence` atom |
| `club_plays_on_date_at_field` | `(club, date, field)` | OR over teams from club at that field | `ClubDaySameField` atom |
| `club_grade_in_slot` | `(club, grade, week, day, day_slot)` | OR over teams of (club,grade) in slot | `ClubGradeAdjacency` |
| `home_game` | `(team, week, opponent)` | indicator: team played at home | `FiftyFifty`, `NonDefaultHomeGrouping` |
| `is_phl_friday` | `(week,)` | indicator: PHL plays Friday this week | PHL Friday count atoms |
| `club_day_field_used` | `(club, field_name)` | indicator: any club_day game on field_name | `ClubDaySameField` |
| `club_day_slot_used` | `(club, day_slot)` | indicator: any club_day game at day_slot | `ClubDayContiguousSlots` |
| `phl_2nd_btb_pair` | `(clubs, round_no, field, slot1, slot2)` | back-to-back same-field indicator | `PHLAnd2ndBackToBackSameField` |

## API reference

```python
from constraints.helper_vars import HelperVarRegistry

registry = HelperVarRegistry(model)

# Declarative
registry.declare(kind, key, builder, description='')
registry.freeze(X, data)               # idempotent
var = registry.get_declared(kind, key) # raises if not declared
registry.declared_kinds()              # introspection
registry.declared_count(kind=None)     # counts

# Pool-style (legacy)
var = registry.get_or_create_bool(key, vars_list, label)
var = registry.get_or_create_presence(key, vars_list, label)
registry.register(key, custom_var)
maybe_var = registry.get(key)          # cache lookup, None if missing
maybe_var = registry.lookup(key)       # alias for .get()

# Diagnostics
registry.diagnostics()                 # dict: declared, pool_created, pool_hits, pool_size, frozen, ...
```

## Rules

1. **Declare in `declare_helpers`, fetch in `apply`.** Calling `declare` after `freeze` raises `RuntimeError`.
2. **Re-declaring `(kind, key)` is a no-op** — first builder wins. Stat counter `redeclared_same_kind` tracks this.
3. **Distinct kinds with the same key get distinct helpers** — kind is part of the registry key.
4. **`required_helpers` on `ConstraintInfo` must reference catalog kinds.** `validate_required_helpers()` returns offending pairs.
5. **Don't reach past the registry** — never call `model.NewBoolVar` and `AddMaxEquality` directly inside an atom for a helper that another atom might want. Ask the registry.

## Tests

- `tests/test_helper_var_registry.py` — 15 unit tests covering both APIs, declared/built lifecycle, diagnostics, back-compat alias.

## Migration path from `SharedVariablePool`

`SharedVariablePool` is now a `from constraints.helper_vars import SharedVariablePool` alias for `HelperVarRegistry`. Existing call sites (`pool.get_or_create_bool`, `pool.register`, `pool.get`) keep working. New code should use `HelperVarRegistry` and the declarative API.
