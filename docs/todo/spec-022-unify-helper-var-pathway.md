<!-- status: ready -->
<!-- owner: session=none claimed=none -->
<!-- depends_on: none (touches constraints/helper_vars.py, constraints/atoms/base.py, constraints/stages.py, tests/test_helper_var_registry.py — minimal overlap with spec-014..019; rebase before merge) -->

# spec-022 — One pathway for shared helper variables: remove the vestigial declarative API

## Why

`HelperVarRegistry` (`constraints/helper_vars.py`) exposes **two** internal stores for
"shared helper variable" deduplication, and they do not cross:

- **Pool-style** → `_cache`, keyed by a raw tuple, via `get_or_create_bool` /
  `get_or_create_presence` / `register` / `get` / `lookup`.
- **Declarative** → `_declared`, keyed by `(kind, key)`, via `declare` / `freeze` /
  `get_declared` (+ the `declare_helpers(registry, data)` hook on `Atom`, `HelperVar`
  dataclass, `declared_kinds`, `declared_count`, `_frozen`).

The whole point of the registry is that two callers asking for the same logical helper get
**one** CP-SAT variable (no model bloat). Two stores defeat that guarantee at the boundary:
if a future atom `declare()`s a helper that an engine method already built via
`get_or_create_bool()`, they land in different dicts and the model gets **two variables for
one concept**. The discipline that prevents bloat is "one pathway, one key convention per
logical helper" — not "everything must be an atom."

Research (this session) shows the declarative pathway is **entirely vestigial**:
- No atom overrides `declare_helpers` — only the no-op default in `constraints/atoms/base.py`.
- Nothing calls `declare_helpers`, `registry.declare(...)`, or `get_declared(...)` in
  production code (the only hits are docstrings + error strings in `helper_vars.py` itself,
  `atoms/base.py`, `atoms/__init__.py`, and `nihc_fill_wf_before_ef.py`).
- The sole **production** `freeze()` callsite is `constraints/stages.py:322`
  (`_ephemeral_registry`) on an **empty** `_declared` — a no-op.
  (review fix — C1/H1: was "line 319"; verified line is 322. Also: `freeze()` is used in
  11 atom test fixtures as a test-harness pattern — see C1 below — those are NOT declarative
  API tests but they must be cleaned up as part of this spec.)
- No code outside `helper_vars.py` consumes the declarative diagnostics
  (`declared_total`, `declared_kinds`, `redeclared_same_kind`).
- The engine (`unified.py`) never calls `freeze` or `declare_helpers`; it uses pool-style
  exclusively (`self.pool is self.registry`, one instance — verified `unified.py:102-103`).
- `tests/test_helper_var_registry.py` tests the declarative API explicitly.
- 11 atom test fixtures (`tests/atoms/test_phl_atoms.py`,
  `test_phl_2nd_adjacency.py`, `test_nihc_field_fill_order.py`,
  `test_preferred_weekends_away_ground.py`, `test_soft_lex_matchup_ordering.py`,
  `test_double_up_handling.py`, `test_same_grade_same_club_no_concurrency.py`,
  `test_team_pair_no_concurrency.py`, `test_balanced_bye_spacing.py`,
  `test_away_club_home_weekends_count.py`, `test_away_club_home_balance.py`) call
  `r.freeze({}, {})` as a "ready a blank registry before calling atom.apply()" idiom —
  NOT as declarative API tests. These calls must be removed (they break once `freeze` is
  deleted). Unit B must list all of them.

So the "majority convention" is pool-style `_cache`, and the minority has **zero production
callsites to migrate**. This spec removes the dead second pathway and adds a guard so a
second pathway can't silently reappear.

`required_helpers` (registry metadata) + `HELPER_VAR_CATALOG` + `validate_required_helpers()`
are KEPT — they are advisory string metadata (which helper *kinds* an atom uses), not the
declarative var-builder, and a registry test depends on them. We additionally bless the
convention that a pool key's first element IS that kind string, so the catalog stays
meaningful.

## Definition of Done

1. The declarative API is removed from `constraints/helper_vars.py`: methods `declare`,
   `freeze`, `get_declared`, `declared_kinds`, `declared_count`; the `_declared` dict, the
   `_frozen` flag, the `HelperVar` dataclass, and the declarative `_stats` keys
   (`declared`, `redeclared_same_kind`). The module retains ONLY the pool-style API
   (`get_or_create_bool`, `get_or_create_presence`, `register`, `get`, `lookup`, `_cache`)
   plus `diagnostics()` (trimmed to pool keys: `created`/`hits`/`pool_size`).
   `SharedVariablePool = HelperVarRegistry` alias kept.
2. `declare_helpers` removed from `constraints/atoms/base.py` (and its mention in the class
   docstring + `constraints/atoms/__init__.py` module docstring), since nothing overrides or
   calls it. The Atom contract becomes: `apply(model, X, data, registry)` creates/looks-up
   shared helpers via the pool-style API only.
3. `constraints/stages.py::_ephemeral_registry` no longer calls `freeze` — it just constructs
   and returns `HelperVarRegistry(model)`. `apply_solver_stage` unchanged otherwise (it never
   used the declarative path).
4. A **guard test** asserts the single-pathway invariant and prevents regression:
   `HelperVarRegistry` has no attributes `declare` / `freeze` / `get_declared` / `_declared`;
   and `Atom` has no `declare_helpers`. (So a future contributor reintroducing the declarative
   API trips a red test.)
5. `tests/test_helper_var_registry.py` rewritten to pool-only: delete the declarative tests
   (`test_declare_*`, `test_redeclare_*`, `test_declared_kinds_listing`,
   `test_distinct_kinds_with_same_key_get_separate_helpers`, `test_helper_var_dataclass_fields`,
   the declarative half of `test_diagnostics_shape`); keep + extend the pool tests
   (`get_or_create_bool` dedup, empty-list-forces-zero, `register`/`lookup`, `presence` dedup,
   alias). All GWT, no mocks.
6. Key-convention documented + lightly guarded: a test asserts that every `required_helpers`
   kind in the registry is a non-empty string in `HELPER_VAR_CATALOG` (keep existing
   `validate_required_helpers` test) AND a doc line states the convention "a shared helper's
   pool key is `(kind, *discriminators)` with `kind` ∈ `HELPER_VAR_CATALOG`." (Full static
   enforcement that every `get_or_create_*` call uses a catalog kind is Out of scope — noted.)
7. Full suite green; import smoke (`import constraints.helper_vars, constraints.unified,
   constraints.stages, constraints.atoms`); grep-clean across `constraints/`, `analytics/`,
   `tests/` for `declare(`, `get_declared`, `declare_helpers`, `HelperVar(` , `.freeze(`
   (none remain except this spec's guard test referencing their absence).
   (review fix — C1: the grep-clean for `.freeze(` cannot pass unless the 11 atom test
   fixtures listed in the Why section above have their `r.freeze({}, {})` call removed.
   The implementer must treat those atom-test files as Unit B deliverables, not just
   `test_helper_var_registry.py`. Removing `r.freeze({}, {})` from atom tests is a
   one-line deletion per file — no logic change is needed since `freeze` on an empty
   declarative store was always a no-op.)
8. No behavioural change to any constraint: a generated draw (or a checkpoint re-test) on the
   2026 fixture produces the same variable count and same violations as before this spec —
   confirm the registry change is purely structural (the `_cache` path is untouched).

## Open decision (recommendation baked in)

- **Which API survives?** The original design intent (per the `helper_vars.py` docstring) was
  that *atoms* would use the declarative API and the *engine* the pool API, unifying later.
  In practice everything converged on **pool-style**. Recommendation: **keep pool-style, delete
  declarative** — it's the de-facto universal convention, zero migration. The opposite
  (migrate all ~25 engine + atom callsites to declarative `declare`/`freeze`/`get_declared`)
  is a large, risky rewrite for a strictly nicer-looking API with no functional gain. Only do
  that if the team explicitly prefers the declarative style; default is pool-style.

## Implementation units

> Single subsystem; run as one sequenced worktree, commit per unit.

### Unit A — Strip the declarative API
- Files: `constraints/helper_vars.py` (remove declarative methods/fields/dataclass; trim
  diagnostics), `constraints/atoms/base.py` (remove `declare_helpers` + docstring), 
  `constraints/atoms/__init__.py` (docstring), `constraints/stages.py`
  (`_ephemeral_registry` at line 322: drop `reg.freeze({}, {})` call — just return
  `HelperVarRegistry(model)` directly),
  `constraints/unified.py` (line 100 comment: remove "declarative API for atoms"
  wording — update to pool-style-only description).
  (review fix — H1: stages.py line is 322, not 319.)
  (review fix — M3: unified.py line 100 comment says "declarative API for atoms; pool-style
  methods for legacy engine" — this becomes wrong after the spec; add it to Unit A file list.)
- Test: import smoke; `validate_solver_stages(DEFAULT_STAGES) == []`; a generation/checkpoint
  re-test shows unchanged variable count + violations (DoD 8).

### Unit B — Tests: pool-only + single-pathway guard
- Files: `tests/test_helper_var_registry.py` (rewrite per DoD 5), new guard test per DoD 4,
  key-convention test per DoD 6. Check `tests/test_constraint_registry.py` keeps
  `validate_required_helpers` (file is at `tests/test_constraint_registry.py` — confirmed
  present; the `validate_required_helpers` test is at line 273 — leave unchanged).
  **Additionally** (review fix — C1): remove the `r.freeze({}, {})` harness call from each
  of the following 11 atom test files (one-line deletion per file; no other logic change):
    - `tests/atoms/test_phl_atoms.py` (line 32)
    - `tests/atoms/test_phl_2nd_adjacency.py` (line 25)
    - `tests/atoms/test_nihc_field_fill_order.py` (line 26)
    - `tests/atoms/test_preferred_weekends_away_ground.py` (line 146)
    - `tests/atoms/test_soft_lex_matchup_ordering.py` (line 118)
    - `tests/atoms/test_double_up_handling.py` (line 160)
    - `tests/atoms/test_same_grade_same_club_no_concurrency.py` (line 33)
    - `tests/atoms/test_team_pair_no_concurrency.py` (line 28)
    - `tests/atoms/test_balanced_bye_spacing.py` (line 179)
    - `tests/atoms/test_away_club_home_weekends_count.py` (lines 254, 281, 319, 364, 380 —
      five occurrences; all are `registry.freeze(X, data)` harness calls, not declarative
      API tests)
    - `tests/atoms/test_away_club_home_balance.py` (lines 189, 219, 239, 261 — four
      occurrences; all are `registry.freeze(X, data)` harness calls)
- Depends on Unit A.
- Test: full suite green; the guard test fails if `declare`/`get_declared`/`declare_helpers`
  are reintroduced (verify by temporarily re-adding one locally, then removing).

## Doc registry

- `docs/system/HELPER_VARS.md` — rewrite to describe the single pool-style pathway + the
  `(kind, *discriminators)` key convention; remove the declarative "Phase 3" mode. Specifically:
  delete the "Declarative (atoms — preferred)" usage section (lines 9–27); collapse
  Rule 5 ("Don't reach past the registry") to cover pool-style only; update the API
  reference table to remove `declare`, `freeze`, `get_declared`, `declared_kinds`,
  `declared_count`; update the "Two usage modes" heading to "Usage (pool-style)";
  remove the "Migration path from SharedVariablePool" note that says "New code should
  use `HelperVarRegistry` and the declarative API."
  (review fix — H3: current HELPER_VARS.md line 44 says "new atoms must use the declarative
  path" — this must become "new atoms must use the pool-style API in `apply()`.")
- `constraints/helper_vars.py` module docstring — collapse the "two usage modes" section to
  one (pool-style only).
- `constraints/atoms/base.py` + `constraints/atoms/__init__.py` docstrings — atoms create
  helpers via the pool API in `apply()`; drop `declare_helpers` lifecycle step.
  Also remove the comment in `constraints/atoms/nihc_fill_wf_before_ef.py` (approx. line 92)
  that says "the engine freezes after `declare_helpers`" — pool-style is always valid; there
  is no freeze lifecycle constraint. (review fix — H3: this comment becomes misleading once
  freeze is removed.)
- `docs/todo/GOALS.md` §2:
  - Line 37: change "declared via `HelperVarRegistry`" to "registered with `HelperVarRegistry`
    via the pool-style API (`get_or_create_bool`, `get_or_create_presence`, `register`)".
  - Step 3 in the How-to: change "Declare helpers via `data['helper_registry']`" to "Register
    shared helpers via the `registry` arg passed to `atom.apply()` — use `get_or_create_bool`,
    `get_or_create_presence`, or `register`." (review fix — M4: `data['helper_registry']` is
    not a real data-dict key; the registry arrives as the fourth arg to `apply()`, not via
    `data`. Pre-existing stale doc; correct it here since we're touching this section.)
  - Add spec-022 row to the spec table (already planned).
  (review fix — M2: GOALS.md prose must change, not just gain a spec-022 table row.)
- `docs/operator-ai/CONSTRAINT_APPLICATION.md` — grep confirms no `declare_helpers` reference
  in this file; no change needed.
  (review note — Low: confirming this file is clean so implementer does not waste time.)
- `docs/system/HARNESS.md` — line 93–96 says "HelperVarRegistry (declarative) +
  SharedVariablePool-style `.get()` (legacy)"; change to "HelperVarRegistry (pool-style API)
  + `SharedVariablePool` alias for back-compat." Remove the word "declarative" throughout.
  (review fix — L1: HARNESS.md was missing from the doc registry.)

## Out of scope

- Static enforcement that EVERY `get_or_create_*` call uses a `HELPER_VAR_CATALOG` kind as
  `key[0]` (an AST/lint pass). Worth doing later if key drift becomes a problem — spawn a new
  plan. This spec only documents + lightly guards the convention.
- Atomising the remaining monolithic engine methods (`_equal_games_balanced_matchups`,
  spacing, legacy club/maitland methods). Confirmed (this session) to give NO bloat benefit
  since the engine already shares via the single `_cache`; purely organizational, deferred.
- Renaming pool-key prefixes to match catalog kinds where they differ — cosmetic, not now.
