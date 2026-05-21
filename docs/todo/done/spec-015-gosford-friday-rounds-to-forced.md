<!-- status: done -->
<!-- owner: session=goal-final-form claimed=2026-05-21 -->
<!-- depends_on: none (shares config/defaults.py DEFAULT_STAGES + constraints/registry.py with spec-014/016/017/018; rebase + re-run validate_solver_stages before merge) -->

# spec-015 — Delete `GosfordFridayRoundsForced`; rely on generic FORCED_GAMES count rules

## Why

`GosfordFridayRoundsForced` (`constraints/atoms/gosford_friday_rounds.py`) is a bespoke atom
that hard-codes a season-specific count rule (`sum == 1` PHL Gosford-Friday game in each of
rounds `{2,4,5,9,10}`). This is exactly the kind of count rule that belongs in
`FORCED_GAMES`, not in code — code should only provide the **generic capability** to express
"force N games matching a scope with an equality type," and the season config supplies the
specifics. Baking specific rounds/venues into an atom is an engineering smell: it duplicates
the FORCED_GAMES machinery and creates a second source of truth.

This is an **engineering** task. We are NOT touching season configs or curating which
specific Gosford Fridays happen — that's the convenor's `FORCED_GAMES` data. We are: (1)
deleting the atom, and (2) guaranteeing the FORCED_GAMES engine generically supports the
shape the convenor described — **specify a scope (e.g. `field_location` + `day`), a `count`,
and an equality `constraint` type, and the solver forces exactly that outcome** — with tests
that prove it at the engine level on synthetic data.

Research (already done) shows the capability exists today: `utils.py` `_SCOPE_FIELDS`
includes `field_location`, `day`, `round_no`, `grade`; an entry with only scope fields (no
teams) sums over all matching variables; and the `constraint`/`count` interpreter supports
`equal|lesse|greatere|greater|less`. This spec deletes the redundant atom and pins that
generic behaviour with engineering tests so it can't regress.

## Definition of Done

1. `GosfordFridayRoundsForced` is fully removed: atom file
   `constraints/atoms/gosford_friday_rounds.py` deleted; removed from
   `constraints/atoms/__init__.py` (import, `PHL_TIMES_ATOMS`, `__all__`); removed from
   `constraints/unified.py::_PHL_HARD_ATOMS`; removed from `DEFAULT_STAGES`
   `critical_feasibility`; registry entry removed; `iter_phl_keys` import cleaned if now
   unused there.
2. A generic engineering test (`tests/test_forced_games_count_rules.py` or extend
   `tests/test_forced_games_multi_scope.py`) proves, on **synthetic** data (no season config,
   no mocks, GWT, hand-computed oracle), that a FORCED_GAMES entry with only
   `{field_location, day, constraint, count}` forces exactly that many games at that
   location/day:
   - `equal`/`count:N` ⇒ generated `X` constrained so `sum(matching vars) == N` (solve yields
     exactly N).
   - `lesse`/`count:N` ⇒ `sum <= N`. `greatere`/`count:N` ⇒ `sum >= N`.
   - An entry with no teams matches **all** vars whose `field_location` and `day` equal the
     entry's values (assert the matched variable set by hand against a tiny fixture).
   Hand oracle: build a fixture with a known number of Friday games at a synthetic venue,
   apply a `{venue, Friday, equal, 2}` entry, assert the solved schedule has exactly 2.
3. `len(CONSTRAINT_REGISTRY)` decremented and its count test updated;
   `validate_solver_stages(DEFAULT_STAGES)` == `[]`; full suite green.
4. Any test referencing `GosfordFridayRoundsForced` (e.g. `tests/atoms/test_phl_atoms*.py`,
   `tests/test_solver_stages_dispatch.py`) updated/removed. No code references the deleted
   symbol (grep clean across `constraints/`, `config/`, `tests/`).
5. `CONSTRAINT_DEFAULTS['gosford_friday_rounds']` removed (it only existed to feed the atom).
   The season's actual Gosford-Friday intent already lives in `season_2026.py` FORCED_GAMES
   (the `count:8` total + specific dates) — left exactly as-is, NOT modified by this spec.

## Implementation units

### Unit A — Delete the atom
- Files: `constraints/atoms/gosford_friday_rounds.py` (delete),
  `constraints/atoms/__init__.py`, `constraints/unified.py` (`_PHL_HARD_ATOMS`),
  `constraints/registry.py`, `config/defaults.py` (DEFAULT_STAGES + remove
  `gosford_friday_rounds` default).
- Test: import smoke; `validate_solver_stages == []`; registry count test; grep-clean.

### Unit B — Pin generic FORCED count behaviour with engineering tests
- Files: `tests/test_forced_games_count_rules.py` (new) or extend
  `tests/test_forced_games_multi_scope.py`. No production change expected (capability already
  exists); if a test reveals the `{field_location, day}`-only scope does NOT force correctly,
  that's a real bug — STOP and expand this spec to fix `utils.py`.
- Depends on Unit A.
- Test: per DoD 2 — one test per equality type, plus the no-teams-matches-all-vars assertion.

## Doc registry

- `docs/system/FORCED_GAMES_AS_COUNT_RULES.md` — document the generic
  `(scope + count + constraint-type)` capability with the location/day example; note the
  Gosford atom is retired in favour of it.
- `docs/system/CONSTRAINT_INVENTORY.md` — remove `GosfordFridayRoundsForced` row; update §3 count.
- `CLAUDE.md` — remove `GosfordFridayRoundsForced` from the PHL/2nd cluster + Friday-limits
  notes; point Friday counts at the FORCED_GAMES mechanism.
- `docs/todo/GOALS.md` — add spec-015 row.

## Out of scope

- Editing `season_2026.py` FORCED_GAMES or adding per-round entries — explicitly NOT doing
  season curation here; this is engine engineering only.
- Maitland Friday counts, Broadmeadow Friday max — untouched.
- The `count:8` Gosford season-total entry (already in config, unchanged).
