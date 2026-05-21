<!-- status: ready -->
<!-- owner: session=none claimed=none -->
<!-- depends_on: none (shares config/defaults.py DEFAULT_STAGES + constraints/registry.py + utils.py FORCED machinery with spec-014..018; rebase + re-run validate_solver_stages before merge) -->

# spec-020 — `PreferredGames`: a soft, weighted analogue of the whole FORCED_GAMES grammar (and delete `PreferredDates`)

## Why

`PreferredDates` is a narrow, legacy, PHL-only soft constraint: it reads
`phl_preferences['preferred_dates']` and penalises `|sum(PHL vars on date) − 1|`. It is one
hardcoded special case ("exactly one PHL game on date X, softly") of a much more general
idea: **express any FORCED_GAMES-style target as a soft, weighted preference instead of a
hard rule.** Rather than keep accreting bespoke soft constraints (`PreferredDates`,
`PreferredWeekendsAwayGround`, …), we build **one** generic mechanism that mirrors the entire
FORCED_GAMES scope grammar but applies a penalty rather than a hard constraint.

This is the modern, atomized infrastructure the project is built around (GOALS §2): a config
list + a single registered soft atom + the shared scope machinery, instead of N one-off
penalty classes.

### Research findings (FORCED_GAMES machinery, final-form)

The FORCED system has two cleanly separable halves (verified in `utils.py`):

1. **Scope/team parsing — fully reusable.** `_build_forced_game_rules(forced_games, teams)`
   (`utils.py:581`) returns `(scope_groups, constraint_types, constraint_counts)`:
   - Scope = the non-team fields from `_SCOPE_FIELDS` (`grade/grades, day, day_slot, time,
     week, date, round_no, field_name, field_location`), mapped to key indices via
     `_KEY_INDEX`.
   - Team matchers: `('pair', t1, t2)`, `('any', t)`, or `('all',)`. `teams`/`team1`/`team2`/
     `club` are resolved via `_resolve_team_name` (club name → all that club's teams at the
     effective grade). Pair/club/team entries get a unique `_entry_idx` in the scope so they
     don't merge; team-less entries (`('all',)`) merge by scope.
   - `constraint` ∈ `_VALID_CONSTRAINT_TYPES = {equal, lesse, less, greater, greatere}`;
     optional `count` (default 1).
   - `_get_matching_forced_scopes(key, rules)` (`utils.py:719`) returns every scope a given
     X-key satisfies (a var can match several scopes).

2. **Enforcement — this is the half we replace.** Inside `generate_X` (`utils.py:3535-3758`):
   as each var is created it's appended to `forced_scope_vars[scope_key]`; then per scope
   `model.Add(sum(vars) <op> count)` (`==/<=/>=/>/<`). Two behaviours a SOFT version must NOT
   copy:
   - **FATAL on empty scope** (`utils.py:3691-3741`): if any FORCED scope matched zero vars,
     `sys.exit(1)`. A soft preference with no candidates must be a **no-op + warning**, never
     fatal.
   - **No variable elimination.** FORCED leaves non-matching vars alone (only BLOCKED
     eliminates). Good — soft also touches no variables, so it can run as a post-hoc atom over
     the finished `X` dict with zero interaction with the generation pipeline.

3. **Precedent.** `PreferredWeekendsAwayGround` (spec-006) already chose exactly this
   architecture — a *separate* `PREFERRED_WEEKENDS` list + a soft atom, explicitly NOT a `soft:
   true` flag on FORCED_GAMES, "to avoid branching the variable-elimination pipeline and its
   validators." `spec-020` generalises that decision: one soft atom covering the FULL FORCED
   grammar, of which `PreferredDates` and `PreferredWeekendsAwayGround` are special cases.

### Design (the soft mechanism)

- **New config list `PREFERRED_GAMES`** (season config + empty default), entry grammar =
  FORCED_GAMES grammar **plus** an optional `weight` (per-entry penalty weight; falls back to
  `PENALTY_WEIGHTS['preferred_games']`). `constraint`/`count` reused unchanged.
- **Reuse the parser.** Rename `_build_forced_game_rules` → `_build_scope_count_rules(entries,
  teams, *, label='FORCED_GAMES')` (keep `_build_forced_game_rules` as a thin alias for
  back-compat / FORCED callers). The soft atom calls it on `data['preferred_games']`. Zero
  duplication of scope/team/club logic. To guarantee a distinct penalty + weight per entry,
  the soft path forces a unique `_entry_idx` on every entry (no scope merging) and carries a
  parallel `constraint_weights` dict.
- **New soft atom `PreferredGames`** (`constraints/atoms/preferred_games.py`): scans `X` once,
  uses `_get_matching_forced_scopes` to bucket vars per scope, and for each scope adds a
  weighted penalty on the **deviation** from `count` per the constraint type:
  | `constraint` | penalty term |
  |---|---|
  | `equal` (count N) | `|sum − N|` (two-sided; `AddAbsEquality`) |
  | `lesse` (≤ N) | `max(0, sum − N)` |
  | `greatere` (≥ N) | `max(0, N − sum)` |
  | `greater` (> N) | `max(0, (N+1) − sum)` |
  | `less` (< N) | `max(0, sum − (N−1))` |

  Penalty IntVars go into `data['penalties']['preferred_games']` (weight = entry weight).
  Empty scope → no penalty + a logged warning (never fatal). Locked-week scopes skipped
  (mirror `PreferredDates`/`PreferredWeekendsAwayGround` locked handling).
- **Stage:** `soft_optimisation`. **Severity:** 5. Registered in `registry.py`
  (`has_soft_component=True`, tester check method).
- **NOT a hard-count adjuster source.** FORCED entries feed Phase-4 count adjusters; preferred
  entries are soft and must NOT alter any hard constraint's expected counts.

## Definition of Done

1. `_build_forced_game_rules` refactored to a shared `_build_scope_count_rules(entries, teams,
   *, label, unique_per_entry=False)` returning `(scope_groups, constraint_types,
   constraint_counts, constraint_weights)`; `_build_forced_game_rules` kept as a back-compat
   wrapper so all existing FORCED callers and tests are unchanged (verified by the existing
   FORCED test suite passing untouched).
2. New `PREFERRED_GAMES` config list: empty default in `config/defaults.py` (documented in the
   "things that belong here / season" comment), readable as `data['preferred_games']` via the
   config loader. New `PENALTY_WEIGHTS['preferred_games']` default.
3. New atom `constraints/atoms/preferred_games.py::PreferredGames`:
   - Reads `data['preferred_games']`; no-op (returns 0) when empty.
   - Builds rules via the shared parser with `unique_per_entry=True`.
   - Adds the correct deviation penalty per `constraint` type (table above), weighted per
     entry, into `data['penalties']['preferred_games']`.
   - Empty/zero-candidate scope → 0 penalty + warning, never `sys.exit`.
   - Skips locked-week scopes.
   - Registered in `registry.py` (severity 5, soft) and wired into `DEFAULT_STAGES`
     `soft_optimisation`.
4. `PreferredDates` deleted and migrated: atom file removed; removed from
   `constraints/atoms/__init__.py`, `PHL_TIMES_ATOMS`, `__all__`, `unified.py::_PHL_SOFT_ATOMS`,
   `DEFAULT_STAGES soft_optimisation`, registry. The marquee-PHL-date behaviour is expressed as
   a `PREFERRED_GAMES` entry `{grade:'PHL', date:X, constraint:'equal', count:1, weight:10000}`;
   a test proves the new atom produces the **same** `|sum−1|` penalty for that entry as
   `PreferredDates` did (hand oracle on a fixture).
5. `tests/atoms/test_preferred_games.py` (no mocks, GWT, hand-computed oracle) covers each
   constraint type:
   - `equal count 1` on a date with 3 candidate PHL vars: solve that pins sum=0 → penalty 1;
     sum=1 → penalty 0; sum=2 → penalty 1 (two-sided), each × weight.
   - `lesse count 2`: sum=3 → penalty 1; sum=2 → 0.
   - `greatere count 1`: sum=0 → penalty 1; sum=1 → 0.
   - team-pair scope, club scope (expansion), and `('all',)` venue+day scope each bucket the
     right vars (assert the matched set by hand).
   - empty scope → returns without error, no penalty, warning emitted.
   - locked-week scope → skipped.
6. Draw metadata gains `preferred_game_outcomes` (per entry: matched var count, target,
   constraint type, incurred penalty / satisfied) alongside the existing
   `forced_game_outcomes`, written by `save_solver_output` / the tester.
7. Validation: `PREFERRED_GAMES` entries validated (reuse the FORCED field/date/location
   validator, allow `weight`, and do NOT apply the FORCED zero-candidate feasibility FATAL).
8. `len(CONSTRAINT_REGISTRY)` net change correct (−`PreferredDates` +`PreferredGames` = 0) and
   count test updated; `validate_solver_stages(DEFAULT_STAGES)` == `[]`; full suite green.

## Implementation units

### Unit A — Shared scope-count parser (refactor, behaviour-preserving)
- Files: `utils.py` (`_build_forced_game_rules` → `_build_scope_count_rules` + back-compat
  wrapper + `constraint_weights` return; `generate_X` call site adjusted to ignore weights).
- Test: existing FORCED suite (`tests/test_forced_games_multi_scope.py`,
  `tests/test_forced_total_plus_per_pair.py`, `tests/test_perennial_*`) passes UNCHANGED;
  add a unit test that the parser returns per-entry weights when `unique_per_entry=True`.

### Unit B — `PreferredGames` atom + config + registry + stage + validation
- Files: `constraints/atoms/preferred_games.py` (new), `constraints/atoms/__init__.py`,
  `constraints/registry.py`, `config/defaults.py` (`PREFERRED_GAMES=[]`,
  `PENALTY_WEIGHTS['preferred_games']`), `config/__init__.py` (load `preferred_games` into
  data), `config/season_2026.py` (`PENALTY_WEIGHTS` entry), `utils.py` (validation),
  `DEFAULT_STAGES soft_optimisation`.
- Depends on Unit A.
- Test: `tests/atoms/test_preferred_games.py` per DoD 5.

### Unit C — Delete `PreferredDates`, migrate marquee-date behaviour
- Files: `constraints/atoms/preferred_dates.py` (delete), `constraints/atoms/__init__.py`,
  `constraints/unified.py` (`_PHL_SOFT_ATOMS`), `constraints/registry.py`, `DEFAULT_STAGES`,
  `config/*` (move any `preferred_dates` usage to a `PREFERRED_GAMES` entry),
  `tests/atoms/test_phl_atoms.py` (remove `TestPreferredDates`), any `phl_preferences` plumbing
  that's now dead.
- Depends on Unit B.
- Test: equivalence test (DoD 4) — the migrated entry yields the identical penalty
  `PreferredDates` produced for the same fixture/date.

### Unit D — Metadata + tester soft-pressure reporting
- Files: `analytics/storage.py` / `analytics/versioning.py` (`preferred_game_outcomes`),
  `analytics/tester.py` (soft-pressure check + registry tester_check_method).
- Depends on Unit B.
- Test: a generated draw records `preferred_game_outcomes`; the tester reports preferred-game
  deviations as soft pressure (not hard violations); `test_every_drawtester_check_in_registry`
  passes.

## Doc registry

- `docs/system/FORCED_GAMES_AS_COUNT_RULES.md` — add a "soft variant" section: `PREFERRED_GAMES`
  = same grammar + `weight`, penalty-on-deviation semantics table.
- `docs/system/CONSTRAINT_INVENTORY.md` — remove `PreferredDates` row; add `PreferredGames`
  (severity 5, soft) with the behaviour table; update §3 count.
- `CLAUDE.md` — replace the `PHL_PREFERENCES only supports preferred_dates` pitfall with the
  `PREFERRED_GAMES` mechanism; note marquee PHL dates are now a soft preferred entry.
- `docs/ai/CONFIGURATION_REFERENCE.md` + `docs/CONFIGURATION_REFERENCE.md` — document
  `PREFERRED_GAMES` and `PENALTY_WEIGHTS['preferred_games']`.
- `docs/todo/GOALS.md` — add spec-020 row.

## Out of scope

- **Folding `PreferredWeekendsAwayGround` (spec-006) into `PreferredGames`.** It IS a special
  case (`{field_location, date, lesse/greatere, count, weight}`), but migrating it is a
  separate consolidation — spawn a follow-on plan once `PreferredGames` is proven. Note it in
  GOALS as a future consolidation, do NOT migrate it here.
- **`PreferredTimes`/`preference_no_play`** — that's avoidance with rich per-team matchers and
  a different (penalise-the-var) model; not part of this generalisation.
- Making FORCED itself support a `soft` flag — explicitly rejected (keeps the
  variable-elimination pipeline and its validators single-mode, per spec-006's rationale).
- The lost 7 pm penalty and the contiguity-primitive unification — separate findings, separate
  specs.
