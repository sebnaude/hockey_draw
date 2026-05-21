<!-- status: ready -->
<!-- owner: session=none claimed=none -->
<!-- depends_on: none (touches constraints/helper_vars.py, constraints/atoms/base.py, constraints/stages.py, tests/test_helper_var_registry.py — minimal overlap with spec-014..019; rebase before merge) -->

# spec-020 — One pathway for shared helper variables: remove the vestigial declarative API

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
- Nothing calls `declare_helpers`, `registry.declare(...)`, or `get_declared(...)` (the only
  hits are docstrings + error strings).
- The sole `freeze()` callsite is `constraints/stages.py:319` (`_ephemeral_registry`) on an
  **empty** `_declared` — a no-op.
- No code outside `helper_vars.py` consumes the declarative diagnostics
  (`declared_total`, `declared_kinds`, `pool_size`, `redeclared_same_kind`).
- The engine (`unified.py`) never calls `freeze` or `declare_helpers`; it uses pool-style
  exclusively (`self.pool is self.registry`, one instance — verified `unified.py:102-103`).
- Only `tests/test_helper_var_registry.py` exercises the declarative API.

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
  (`_ephemeral_registry`: drop `freeze`).
- Test: import smoke; `validate_solver_stages == []`; a generation/checkpoint re-test shows
  unchanged variable count + violations (DoD 8).

### Unit B — Tests: pool-only + single-pathway guard
- Files: `tests/test_helper_var_registry.py` (rewrite per DoD 5), new guard test per DoD 4,
  key-convention test per DoD 6. Check `tests/test_constraint_registry.py` keeps
  `validate_required_helpers` (unchanged).
- Depends on Unit A.
- Test: full suite green; the guard test fails if `declare`/`get_declared`/`declare_helpers`
  are reintroduced (verify by temporarily re-adding one locally, then removing).

## Doc registry

- `docs/system/HELPER_VARS.md` — rewrite to describe the single pool-style pathway + the
  `(kind, *discriminators)` key convention; remove the declarative "Phase 3" mode.
- `constraints/helper_vars.py` module docstring — collapse the "two usage modes" section to
  one.
- `constraints/atoms/base.py` + `constraints/atoms/__init__.py` docstrings — atoms create
  helpers via the pool API in `apply()`; drop `declare_helpers`.
- `docs/todo/GOALS.md` §2 — adjust the "Helper variables ... declared via HelperVarRegistry"
  wording to the pool-style mechanism; add spec-020 row to the spec table.
- `docs/operator-ai/CONSTRAINT_APPLICATION.md` — if it documents `declare_helpers`, update.

## Out of scope

- Static enforcement that EVERY `get_or_create_*` call uses a `HELPER_VAR_CATALOG` kind as
  `key[0]` (an AST/lint pass). Worth doing later if key drift becomes a problem — spawn a new
  plan. This spec only documents + lightly guards the convention.
- Atomising the remaining monolithic engine methods (`_equal_games_balanced_matchups`,
  spacing, legacy club/maitland methods). Confirmed (this session) to give NO bloat benefit
  since the engine already shares via the single `_cache`; purely organizational, deferred.
- Renaming pool-key prefixes to match catalog kinds where they differ — cosmetic, not now.
