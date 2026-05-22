<!-- status: done -->
<!-- owner: session=goal-final-form claimed=2026-05-22 -->
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

2. **Enforcement — this is the half we replace.** Inside `generate_X` (`utils.py:3535-3748`):
   as each var is created it's appended to `forced_scope_vars[scope_key]`; then per scope
   `model.Add(sum(vars) <op> count)` (`==/<=/>=/>/<`). Two behaviours a SOFT version must NOT
   copy:
   - **FATAL on empty scope** (`utils.py:3680-3729`): the empty-scope check starts at ~line 3680
     (`if scope_key not in forced_scope_vars ...`), prints diagnostics, and calls `sys.exit(1)`
     at line 3729. Constraint application runs at 3731-3746. A soft preference with no candidates
     must be a **no-op + warning**, never fatal.
     (review fix — H3: plan claimed 3691-3741; actual empty-scope check 3680-3729, constraints 3731-3746)
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
  FORCED_GAMES grammar **plus** an optional `weight`. `constraint`/`count` reused unchanged.
  **Weighting model: a SINGLE shared bucket by default.** All preferred entries penalise into
  ONE `data['penalties']['preferred_games']` bucket carrying the single default weight
  `PENALTY_WEIGHTS['preferred_games']`. A per-entry `weight` is an OPTIONAL multiplier on top of
  that default (same pattern as `PreferredWeekendsAwayGround`:
  `multiplier = max(1, entry_weight // default_weight)`), used only when one preference must
  out-rank the rest. With no per-entry `weight`, every preferred entry has equal pull.
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

  **(review C2 — penalty IntVar bound, real crash risk):** `PreferredDates` gets away with
  `NewIntVar(0, len(vars), …)` only because N is always 1. For arbitrary N the bound must cover
  the deviation's actual max, or CP-SAT rejects the model mid-solve. Compute the upper bound per
  type: `equal` → `max(N, len(candidates))`; `greatere`/`greater` → `max(0, N+1)`; `lesse`/`less`
  → `len(candidates)`. A convenor entry like `{grade:'PHL', date:X, equal, count:3}` where only
  2 PHL vars exist for that date (others filtered by `PHL_GAME_TIMES`) makes `|0−3|=3` exceed a
  naive `len=2` bound → solve crash. `count=0` (`equal`) is valid and deliberate ("prefer nothing
  here") — bound is `len(candidates)`, no special case needed.
  **(review m4 / utils.py:3740-3746):** the `greater`/`less` mappings match FORCED's strict
  semantics exactly — `model.Add(sum > N)` is `sum >= N+1` in CP-SAT, so penalty 0 iff
  `sum >= N+1`. Keep the +1/−1 shifts.
  (review fix — Low: line ref was 3752-3755; actual constraint application is 3731-3746)

  Penalty IntVars go into the SINGLE `data['penalties']['preferred_games']` bucket (one default
  weight for all entries; optional per-entry `weight` applied as a multiplier on the raw
  penalty IntVar before it enters the bucket — so the bucket weight stays uniform and only
  flagged entries are scaled). Empty scope → no penalty + a logged warning (never fatal).
  Locked-week scopes skipped (mirror `PreferredDates`/`PreferredWeekendsAwayGround` handling).
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
2. New `PREFERRED_GAMES` config list: empty list default in `config/defaults.py` as a
   top-level constant (like `PREFERRED_WEEKENDS=[]`), readable as `data['preferred_games']`.
   New `PENALTY_WEIGHTS['preferred_games']` key added to **each season's `PENALTY_WEIGHTS`
   dict** (2025, 2026, 2027, template) — there is NO shared `PENALTY_WEIGHTS` in
   `config/defaults.py` (verified: defaults.py has no such dict); `build_season_data` reads
   `config.get('penalty_weights', {})` and passes the whole dict through. The default fallback
   value when the atom reads `data.get('penalty_weights', {}).get('preferred_games', N)` must
   match the intent; choose a sensible default (e.g. 10_000 matching `phl_preferences`).
   (review fix — M6: plan said "new default in config/defaults.py" — defaults.py has no
   PENALTY_WEIGHTS dict at all; the per-season dicts are the only place this can live)
   **(review C1 — critical wiring gap):** the injection point is `utils.py::build_season_data`
   (~line 4040, beside `forced_games`/`preferred_weekends`), NOT `config/__init__.py`. Add
   `'preferred_games': config.get('preferred_games', [])` there, or the atom reads `[]` on every
   run and is a silent no-op. A test must assert `load_season_data(2026)['preferred_games']`
   round-trips a configured entry.
3. New atom `constraints/atoms/preferred_games.py::PreferredGames`:
   - Reads `data['preferred_games']`; no-op (returns 0) when empty.
   - Builds rules via the shared parser with `unique_per_entry=True`.
   - Adds the correct deviation penalty per `constraint` type (table above) into the SINGLE
     `data['penalties']['preferred_games']` bucket (one default weight); optional per-entry
     `weight` scales only that entry's raw penalty as a multiplier.
   - Empty/zero-candidate scope → 0 penalty + warning, never `sys.exit`.
   - Skips locked-week scopes.
   - Registered in `registry.py` (severity 5, soft) and wired into `DEFAULT_STAGES`
     `soft_optimisation`.
4. `PreferredDates` deleted and migrated: atom file removed; removed from
   `constraints/atoms/__init__.py`, `PHL_TIMES_ATOMS`, `__all__`,
   `unified.py::_PHL_SOFT_ATOMS` (line 503) AND `unified.py::_phl_times_atoms_soft` dispatch
   method (lines 511-514, the LIVE path — not to be confused with the parity-only
   `_phl_times_soft` at line 942 which is explicitly "Not called"),
   `DEFAULT_STAGES soft_optimisation`, registry.
   (review fix — H4: plan referenced "dead `_phl_times_soft` reference" but the LIVE dispatch
   path is `_phl_times_atoms_soft` at unified.py:511-514 → `_PHL_SOFT_ATOMS`. Both must be
   cleaned up; the parity-only `_phl_times_soft` at ~942 can stay as reference if desired.)
   The marquee-PHL-date behaviour is expressed as a `PREFERRED_GAMES` entry
   `{grade:'PHL', date:X, constraint:'equal', count:1, weight:10000}`;
   a test proves the new atom produces the **same** `|sum−1|` penalty for that entry as
   `PreferredDates` did (hand oracle on a fixture).
   NOTE: `PreferredDates` is currently registered at `severity_level=1` in `registry.py`
   (line 151), with `atom_group='PHLAndSecondGradeTimes'` and `has_soft_component=True` —
   despite living in the `soft_optimisation` stage. The replacement `PreferredGames` must be
   registered at `severity_level=5` (soft), NO `atom_group`, with `has_soft_component=True`.
   Do not carry over the severity-1 / PHLAndSecondGradeTimes group from the old entry.
   (review fix — C_sev: registry severity mismatch would silently give PreferredGames
   wrong severity in validate_solver_stages checks)
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
7. Validation: `PREFERRED_GAMES` entries validated by adding a Phase-N call to
   `_validate_entry_fields(preferred_games, 'PREFERRED_GAMES', ...)` inside
   `validate_game_config` (utils.py ~line 3296), reusing the FORCED field/date/location
   validator with the `PREFERRED_GAMES` label so is_forced=False (warnings, not fatals, for
   unknown dates/venues, per the soft/non-fatal contract). Also allow `weight` field to pass
   without triggering "unexpected field" warnings. Do NOT apply the FORCED zero-candidate
   feasibility FATAL. Add `'preferred_games': data.get('preferred_games', [])` to the call so
   the existing FORCED validation phases don't accidentally process PREFERRED_GAMES entries.
   **(review C3):** the constraint-type check in `_validate_entry_fields` (`utils.py:1216-1221`)
   is gated on `is_forced = (label == 'FORCED_GAMES')`, so a `PREFERRED_GAMES` label skips it —
   a typo like `'constraint':'equall'` would pass validation then silently produce no penalty.
   Fix: extend the gate condition to also validate `constraint` when `label == 'PREFERRED_GAMES'`
   (warn, don't fatal). Concrete change: `if is_forced or label == 'PREFERRED_GAMES':` at the
   ctype check, but append to `warnings` not `fatals` for the preferred case.
8. `len(CONSTRAINT_REGISTRY)` net change correct (−`PreferredDates` +`PreferredGames` = 0) and
   count test updated; `validate_solver_stages(DEFAULT_STAGES)` == `[]`; full suite green.

## Implementation units

### Unit A — Shared scope-count parser (refactor, behaviour-preserving)
- Files: `utils.py` (`_build_forced_game_rules` → `_build_scope_count_rules` + back-compat
  wrapper + `constraint_weights` return).
- NOTE (review M1): the back-compat `_build_forced_game_rules` wrapper returns only the
  original 3-tuple, so **no `generate_X` call-site change is needed** — and there are 12 FORCED
  callers (`utils.py:3535`, `_phl_forced_friday_helper.py:203`, and ~10 tests) that unpack a
  3-tuple; they must all keep working unchanged. Do NOT thread a 4th value through them.
- NOTE (review m1): `_build_scope_count_rules(unique_per_entry=True)` must not double-inject
  `('_entry_idx', idx)` for entries that already qualify under the team-matcher branch
  (`utils.py:644-648`) — frozenset dedups it, but keep the logic clean (inject once).
- Test: existing FORCED suite (`tests/test_forced_games_multi_scope.py`,
  `tests/test_forced_total_plus_per_pair.py`, `tests/test_perennial_*`, `test_utils_coverage.py`)
  passes UNCHANGED; add a unit test that the parser returns per-entry weights when
  `unique_per_entry=True`.

### Unit B — `PreferredGames` atom + config + registry + stage + validation
- Files: `constraints/atoms/preferred_games.py` (new), `constraints/atoms/__init__.py`,
  `constraints/registry.py`,
  `config/defaults.py` (`PREFERRED_GAMES=[]` constant — review fix M6: PENALTY_WEIGHTS is
  NOT in defaults.py, so only add the list default here, not the penalty weight),
  `config/season_2026.py` (`PENALTY_WEIGHTS['preferred_games']` entry — also add to
  `config/season_2025.py`, `config/season_2027.py`, `config/season_template.py`),
  **`utils.py::build_season_data`** (~line 4040, inject
  `'preferred_games': config.get('preferred_games', [])` — review C1),
  `utils.py` (validation in `validate_game_config` — add Phase-N call + ctype-gate fix —
  review C3),
  `DEFAULT_STAGES soft_optimisation`.
- Locked weeks (review M4): do a **per-variable** week check (skip vars whose `key[6]` is in
  `data['locked_weeks']`) like `PreferredWeekendsAwayGround` (`preferred_weekends_away_ground.py:108`),
  NOT an entry-level date→week check. X includes locked-week vars; the per-var filter is the
  correct and simpler match.
- Depends on Unit A.
- Test: `tests/atoms/test_preferred_games.py` per DoD 5.

### Unit C — Delete `PreferredDates`, migrate marquee-date behaviour
- Files: `constraints/atoms/preferred_dates.py` (delete), `constraints/atoms/__init__.py`,
  `constraints/registry.py`, `DEFAULT_STAGES`,
  `tests/atoms/test_phl_atoms.py` (remove `TestPreferredDates`).
- **`constraints/unified.py` dead-code cleanup (review C4):** The LIVE dispatch path is
  `_phl_times_atoms_soft` (lines 511-514) which dispatches `_PHL_SOFT_ATOMS = (PreferredDates,)`
  (line 503). Remove BOTH the `_PHL_SOFT_ATOMS` tuple and the entire `_phl_times_atoms_soft`
  method. Also remove the now-orphaned `build_groupings` plumbing that reads `phl_preferences`:
  — `self.preferred_date_vars` (actual decl line 197, not ~206)
  — the `pref_dates` build from `phl_preferences` (actual lines 215-219, not ~224-228)
  — its population (~line 320-321)
  The parity-only `_phl_times_soft` method (~line 942, marked "Not called") may optionally be
  removed too but is not required (it doesn't run). Otherwise `build_groupings` keeps reading a
  deleted config key (`phl_preferences`) every run — silent breakage until `phl_preferences` is
  also removed from `build_season_data`.
  (review fix — H4+H5: corrected line numbers and identified live dispatch path vs parity copy)
- **Season configs (review C5):** remove/migrate `PHL_PREFERENCES` from:
  - `config/season_2026.py` (lines 460-468 note comment + definition, line 1168 dict entry;
    also line 1104 `PENALTY_WEIGHTS['phl_preferences']`)
  - `config/season_2025.py` (lines ~101, ~133)
  - `config/season_2027.py` (lines ~114, ~169, ~222)
  - `config/season_template.py` (lines ~163, ~210)
  Also drop `phl_preferences` injection from `utils.py::build_season_data` (line 4015 and 4030)
  once `preferred_date_vars` plumbing in `unified.py` is gone — otherwise `build_season_data`
  reads a key that nothing consumes.
  Drop the orphaned `PENALTY_WEIGHTS['phl_preferences']` from each season config
  (review m2 — after PreferredDates is gone this weight is dead).
  The marquee-date intent moves to a `PREFERRED_GAMES` entry where the convenor actually
  wants it (2026 only if currently used; other season configs can have empty `PREFERRED_GAMES`).
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
- `docs/operator-ai/CONFIGURATION_REFERENCE.md` (the real path — verified at
  `docs/operator-ai/CONFIGURATION_REFERENCE.md`; also update
  `docs/operator-human/RULES.md` if any player-facing description of preferred dates exists) —
  document `PREFERRED_GAMES` and `PENALTY_WEIGHTS['preferred_games']`.
  (review fix — Low: plan referenced `docs/ai/CONFIGURATION_REFERENCE.md` which does not exist
  at that path; the actual file is `docs/operator-ai/CONFIGURATION_REFERENCE.md`)
- `.github/copilot-instructions.md` (~line 116) — remove the `PHL_PREFERENCES` section (review
  C6) so autocomplete stops suggesting the deleted key.
- `docs/todo/GOALS.md` — add spec-020 row.

## Precondition (review)

spec-018 must be fully landed first (registry + `DEFAULT_STAGES` + `stages.py` + `_adjusters.py`
+ severity all consistent, `validate_solver_stages(DEFAULT_STAGES) == []`, full suite green).
spec-020's DoD §8 asserts `validate_solver_stages == []`; do not start spec-020 against a
half-applied spec-018 baseline.

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
