<!-- status: ready -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: none -->
<!-- reviewed: adversarial Sonnet review 2026-05-30 — fixes applied inline -->

# spec-040 — Slack & limit provenance: persist the effective constraint config, replay a draw faithfully

**Spec source:** convenor request (this session) — "the slack is a config setting the draw is run
on and the analysis should also be run with that level of slack leniency (including base slack from
config where applicable) otherwise it will throw errors when it shouldn't." This spec makes the
analysis run at the *exact* leniency the draw was solved under, sourced from the draw's own stored
provenance, including for historical draws.

## Why

Post-hoc analysis (`DrawTester`, and the soft-outcome engine in spec-041, and the report in spec-042)
must reconstruct the **effective** per-constraint slack and limits the draw was solved with.
"Effective" is `base slack/limit from config` **combined with** `CLI --slack override`. Concretely,
the spacing check computes `effective_slack = spacing_base_slack + constraint_slack['EqualMatchUpSpacingConstraint']`,
and the bye-spacing check does the analogous `bye_spacing_base_slack + constraint_slack['BalancedByeSpacing']`
(verified in `analytics/tester.py` this session). Several constraints similarly fold a config base
limit (`club_game_spread_max_gap`, `club_game_spread_max_fields`, `phl_2nd_cross_venue_min_minutes`, etc.)
with the CLI slack.
(review fix — C1: `away_maitland_max_clubs`, `maitland_max_consecutive_home`, `max_clubs_per_field`
do NOT exist on final-form. They were deleted in spec-018 and spec-024 respectively. Replaced with
the keys that actually exist in `CONSTRAINT_DEFAULTS` and are read by the tester.)

The persistence is **lossy**: `_build_draw_metadata()` in `analytics/versioning.py` writes only
`metadata.solver_config.constraint_slack` — the CLI override dict. The **base** slacks/limits in
`data['constraint_defaults']` are NOT persisted. At analysis time the tester re-reads them from the
*current live* season config. So:

- If `constraint_defaults` changes between solve and analysis (e.g. a convenor bumps
  `spacing_base_slack` 2 → 5 for a later run), re-testing an *old* draw computes
  `effective = 5 + stored_cli` instead of the `2 + stored_cli` it was actually solved at → **false
  violations the convenor never created**. This is exactly the "throw errors when it shouldn't" the
  convenor flagged.
- For a genuinely historical draw (pre-provenance), there is nothing stored to reconstruct from at
  all; the analysis silently assumes today's config applied then.

The fix is to persist the full effective constraint-config provenance at solve time, add a resolver
that reconstructs it (preferring stored provenance, falling back to current config with a logged
warning for legacy draws), and make `DrawTester` load it automatically so faithful replay is the
default rather than something the caller must remember to wire.

## Definition of Done

### Provenance persistence (solve time)

1. **`_build_draw_metadata()` (`analytics/versioning.py`) persists a complete
   `metadata.solver_config.constraint_config` block** containing both:
   - `constraint_slack` — the CLI override dict (as today; key preserved for back-reading).
   - `constraint_defaults` — a snapshot of the slack/limit-relevant subset of
     `data['constraint_defaults']` at solve time (at minimum every base-slack/base-limit key any
     slack-aware or limit-folding constraint reads: `spacing_base_slack`, `bye_spacing_base_slack`,
     `club_game_spread_max_gap`, `club_game_spread_max_overlap`, `club_game_spread_max_fields`,
     `phl_2nd_cross_venue_min_minutes`, plus any others enumerated by grepping the atoms for
     `constraint_defaults.get(...)`). **DO NOT include** `away_maitland_max_clubs`,
     `maitland_max_consecutive_home`, or `max_clubs_per_field` — these keys were deleted from
     `CONSTRAINT_DEFAULTS` in spec-018/spec-024 and do not exist on final-form. Storing the whole
     `constraint_defaults` dict is acceptable and simpler; the snapshot must be JSON-serialised via
     the existing `_serialize_for_json` helper.
     (review fix — C1: key list corrected to match the live `CONSTRAINT_DEFAULTS` in
     `config/defaults.py`.)
   - `groups_selected` — the resolved constraint set / group selection the draw was solved with.
     Note: `groups_selected` is **already persisted** by the existing `_build_draw_metadata()` at
     line ~720 (`'groups_selected': list(data.get('_groups_selected', ['default']))`). The
     implementer must NOT add a duplicate key — confirm the existing field is present and correct,
     then move it inside the new `constraint_config` block (or reference it from there), rather than
     creating a second top-level copy. (review fix — L2: existing persistence noted.)
2. **The same provenance block is written to checkpoint `metadata.json` and `run_metadata.json`**
   (`main_staged.py` checkpoint/run-metadata writers), so a checkpoint can be analysed as faithfully
   as a saved draw.
3. **`constraint_slack` remains readable at its existing path** `metadata.solver_config.constraint_slack`
   (forward-only does not mean breaking the back-read; the new `constraint_config` block *contains*
   it, and the old top-level key is migrated into the block, with all live readers updated — no dual
   path left behind).

   **Important reader distinction the implementer must handle (review fix — M2):**
   - In **draw JSON** (`metadata.solver_config.constraint_slack`): read by `DrawTester.__init__`
     (line ~443) and `_extract_metadata` (line ~503) via `solver_config.get('constraint_slack', {})`.
   - In **checkpoint `metadata.json`**: `constraint_slack` is written at the **top level** (not nested
     under `solver_config`) by `save_stage()` in `main_staged.py`. The `from_checkpoint` method reads
     it at the top level (`meta.get('constraint_slack', {})`). These are TWO DIFFERENT JSON shapes.
     Unit A must persist the `constraint_config` block consistently in BOTH shapes, and Unit B must
     read from the correct path for each. Do not assume checkpoint metadata has the same nesting as
     draw metadata.

### Resolver (analysis time)

4. **New module `analytics/slack_provenance.py` with `resolve_effective_constraint_config(draw,
   current_data) -> EffectiveConstraintConfig`** that returns the effective slack + base
   slacks/limits the draw was solved under:
   - If the draw's metadata carries the `constraint_config` provenance block → reconstruct purely
     from it (the stored `constraint_slack` + stored `constraint_defaults`). This is the
     "read from historical config" capability.
   - If the draw predates provenance (legacy draw, block absent) → fall back to
     `current_data['constraint_defaults']` + any stored `constraint_slack`, and emit a **WARNING**
     through the project logger naming the draw and that current config was assumed.
   - Returns a structured `EffectiveConstraintConfig` exposing, per constraint, the effective slack
     value the tester should use (`base + cli`) and the effective limits.
5. **`EffectiveConstraintConfig` can be applied onto a data dict** (`apply_to(data)`), producing the
   `data['constraint_slack']` + `data['constraint_defaults']` the tester needs, so any analysis
   consumer gets faithful replay by passing the draw through the resolver once.

### Tester auto-load (faithful replay by default)

6. **`DrawTester.from_file` and `DrawTester.from_checkpoint` auto-resolve provenance** via the
   resolver and apply it, so a caller who does `DrawTester.from_file(path, data)` gets the draw's own
   effective slack/limits WITHOUT having to know about `use_metadata`. The current partial behaviour
   (caller `data['constraint_slack']` overrides metadata slack) is preserved as an explicit
   override path, but the default is faithful replay.

   **Implementer note on `use_metadata` (review fix — M1):** The current `from_file` signature has
   `use_metadata: bool = False` (line 508 of `analytics/tester.py`). Unit B should NOT simply flip
   the default to `True` — it should instead call the new resolver (`resolve_effective_constraint_config`)
   and `apply_to(data)` unconditionally before constructing the tester, replacing the old `use_metadata`
   partial path. The `use_metadata=True` branch only merges slack, not `constraint_defaults` — the
   resolver is the correct complete replacement. The `use_metadata` parameter may be kept for back-compat
   but its behaviour becomes a no-op (the resolver already covers it).

   **`from_X_solution` limitation (review fix — H1):** `DrawTester.from_X_solution` has NO access
   to draw metadata (it constructs a DrawStorage from a raw dict, not a persisted file). Faithful
   replay is therefore only achievable via `from_file` and `from_checkpoint`. Callers using
   `from_X_solution` (e.g. in-solver post-checks, `save_solver_output`'s tester call at line ~633)
   must still set `data['constraint_slack']` correctly before calling, as today. This limitation
   must be stated in the doc registry update to `HARNESS.md` and `CLAUDE.md`.
7. **The tester's effective-slack formulae read the resolved base slacks**, not the ambient live
   config, when provenance is present (i.e. the `spacing_base_slack` / `bye_spacing_base_slack` /
   limit reads come from the resolved config). No code path computes effective slack from a mix of
   stored CLI override + live base.

### Proof

8. **No-mock divergence test** (`tests/test_slack_provenance.py`):
   - *Given* a draw solved and saved with `constraint_defaults['spacing_base_slack'] = 2` and
     `constraint_slack['EqualMatchUpSpacingConstraint'] = 3` (construct a real saved draw via the
     normal save path, not a hand-edited JSON), *when* the live config is then changed to
     `spacing_base_slack = 5` and the draw is re-tested via `DrawTester.from_file`, *then* the
     spacing check uses effective slack **5** (2 + 3 from provenance), NOT **8** (5 + 3) — and a
     pair that is a **genuine violation at effective_slack=5** (but not at effective_slack=8) reports
     a violation when tested with provenance, and reports **no** violation when tested without
     provenance (wrong live config with base_slack=5).

   **Hand-oracle (review fix — C2: the original oracle formula and direction were wrong):**
   The real formula on final-form is `S = effective_spacing(T, base_slack, config_slack) =
   max(0, ideal_gap(T) - base_slack - config_slack)` where `ideal_gap(T) = max(0, _legacy_min_gap(T) - 1)`
   (see `constraints/atoms/_spacing.py`). A meeting gap `g` violates if `g <= S`.

   - At `effective_slack=5` (correct provenance): `S = max(0, ideal_gap(T) - 5)`.
   - At `effective_slack=8` (wrong live config): `S = max(0, ideal_gap(T) - 8)`.
   - Since `ideal_gap(T) - 8 < ideal_gap(T) - 5`, there exist gaps where `ideal_gap(T)-8 < g <= ideal_gap(T)-5` — these gaps are violations at effective_slack=5 (gap ≤ S₅) but NOT violations at effective_slack=8 (gap > S₈).
   - **The false-positive scenario:** bumping live base_slack from 2→5 means the tester uses eff_slack=8 instead of eff_slack=5, producing a LOWER S, which means FEWER violations flagged. The bug is **missed violations (false negatives)**, not false positives. This is still the "throws errors when it shouldn't" scenario — specifically, a draw that was solved at the tighter effective_slack=5 contains genuine violations at that leniency; bumping the config silently hides them.
   - **Concrete fixture for T=10:** `_legacy_min_gap(10) = max(1, 9 - max(1, 9-6)) = max(1, 9-3) = 6`; `ideal_gap(10) = 6-1 = 5`. S₅ = max(0, 5-5)=0. Wait — at eff_slack=5, S=0, which means no violations are possible. Try T=6: `ideal_distance=5`, `hardcoded_slack=max(1,5-2*5//3)=max(1,5-3)=2`, `_legacy_min_gap=max(1,5-2)=3`, `ideal_gap(6)=2`. S₅=max(0,2-5)=0; still 0. Use a larger T with smaller cli slack to get a meaningful S. Try T=10, cli=0, base=0: S=ideal_gap(10)=5. With base=2,cli=3: S=max(0,5-5)=0. **The implementer must choose T and slacks so that both S₅>0 and S₈>0 and they differ.** For example: T=10, base_slack=0, cli_slack=2: S₂=max(0,5-2)=3; with "wrong" base_slack=5, cli=2: S₇=max(0,5-7)=0. Pick a pair with gap=2: violation at S=3 (gap≤3), compliant at S=0 (gap>0). This demonstrates the false-negative. Assert that with provenance (base=0, cli=2): the gap-2 pair IS flagged; with wrong config (base=5, cli=2): it is NOT flagged.
   - *Given* a legacy draw with no `constraint_config` block, *when* resolved, *then* the resolver
     returns current-config values AND a WARNING is logged naming the draw.
9. **Round-trip test:** a freshly generated draw, re-tested immediately via `from_file`, reports
   identical violations to the in-memory `from_X_solution` check the solver ran — proving persistence
   loses no slack information.

## Implementation units

### Unit A — Persist provenance at solve time

- **Files touched:** `analytics/versioning.py` (`_build_draw_metadata` — write the `constraint_config`
  block; migrate the top-level `constraint_slack` into it and update all readers within this file),
  `main_staged.py` (`save_stage()` for checkpoint `metadata.json` at line ~370; `_write_run_metadata()`
  for `run_metadata.json` at line ~288 — add a `constraint_config` block alongside the existing
  top-level `constraint_slack` key in BOTH writers. See note M2: these use a DIFFERENT JSON shape than
  draw metadata — `constraint_slack` lives at top level in checkpoint JSON, so the block is added as a
  PEER key, not nested under `solver_config`).
  **Complete reader list** (grep `constraint_slack` before touching anything and update ALL of):
  - `analytics/tester.py` line ~443 (`draw.metadata...solver_config...constraint_slack`) — draw JSON reader
  - `analytics/tester.py` line ~503 (`_extract_metadata` — `solver_config.get('constraint_slack')`) — draw JSON
  - `analytics/tester.py` line ~577 (`from_checkpoint` — `meta.get('constraint_slack', {})`) — checkpoint JSON top-level
  - `analytics/versioning.py` line ~703 (`self._serialize_for_json(data.get('constraint_slack', {}))`) — already inside `solver_config`, gets wrapped by the new block
  - `main_staged.py` line ~288 (`constraint_slack` in run_metadata) — top-level, stays top-level, also write block peer
  - `main_staged.py` line ~370 (`constraint_slack` in stage metadata) — top-level, stays, also write block peer
- **Change summary:** additive metadata persistence + a forward-only migration of the slack key into
  the new block (update every reader; leave no dual path). Key point: the two metadata shapes (draw
  JSON vs checkpoint JSON) have `constraint_slack` at different nesting levels — the `constraint_config`
  block must be added without breaking the existing shape in either format.
- **Depends on:** none within plan.
- **Suggested executor:** Opus (metadata schema change with multiple writers + readers across
  versioning and checkpointing; getting the migration complete — no orphan readers — needs care).
- **No-mock test outline:** covered by Unit B's DoD-9 round-trip (generate → save → reload → compare).

### Unit B — Resolver + tester auto-load + proof tests

- **Files touched:** `analytics/slack_provenance.py` (new — `resolve_effective_constraint_config`,
  `EffectiveConstraintConfig`), `analytics/tester.py` (`from_file`/`from_checkpoint` auto-resolve;
  effective-slack formulae source resolved config), `tests/test_slack_provenance.py` (new).
- **Change summary:** the resolver + faithful-replay default + the divergence/round-trip proofs.
- **Depends on:** Unit A merged (resolver reads the block Unit A writes; tests need the save path
  writing provenance).
- **Suggested executor:** Opus (tester replay semantics + the legacy fallback + the divergence oracle
  are subtle).
- **No-mock test outline:** DoD-8 and DoD-9 above, both driving the real save/load/test path on a
  real-data fixture (reuse spec-034's real-data draw fixtures); no JSON hand-editing, no mocks.

## Doc registry

- `docs/system/HARNESS.md` — document the new `metadata.solver_config.constraint_config` provenance
  block in the data-dict / draw-metadata key reference; note the resolver as the canonical way to
  reconstruct effective slack.
- `docs/system/CONSTRAINT_INVENTORY.md` — for each slack-aware / limit-folding atom, note that its
  effective limit is `base (constraint_defaults) + cli (constraint_slack)` and is now provenance-stored.
- `CLAUDE.md` (final-form copy) — update the "Analyzing Checkpoints" / slack guidance to point at the
  resolver instead of the hand-built `data['constraint_slack']` dict in the doc example.
- `docs/todo/00-dependency-tree.md` — add spec-040 as a live entry.
- `docs/todo/GOALS.md` — add the spec-040 row.

## Out of scope

- **Sweeping slack across a range / robustness curves.** The convenor clarified slack is a fixed
  provenance, not a parameter to vary; the analysis runs at the draw's own level only. No
  slack-proliferation sweep is built.
- **Re-solving at any slack** — analysis is post-hoc on the existing draw; no solver re-runs.
- **The alignment gate** — spec-039 (independent, parallel).
- **Soft-outcome scoring** — spec-041 (consumes this resolver so scores are computed at the right leniency).
- **Backfilling provenance into already-saved historical draws.** Existing draws stay legacy; the
  resolver handles them via the logged current-config fallback. (If the convenor later wants a
  one-off backfill, that is its own spec.)

## Dependencies

- `depends_on: none`. Independent of spec-039 (no shared edited file). Parallelisable now.
- Within this plan: Unit B depends on Unit A (resolver + tests need the persisted block).

## Risks & blast radius

- **Metadata schema migration completeness.** Moving `constraint_slack` under `constraint_config`
  risks an orphan reader still looking at the old top-level path. Mitigation: grep every reader of
  `solver_config'].get('constraint_slack'` / `['constraint_slack']` and update all in Unit A; the
  round-trip test (DoD-9) fails if any reader gets stale data. Forward-only: no compatibility shim
  for the old path — all readers move together.
- **`constraint_defaults` snapshot scope.** If the persisted subset omits a base-limit key some atom
  reads, the tester silently falls back to live config for that key. Mitigation: persist the *whole*
  `constraint_defaults` dict (simplest, safe) rather than a curated subset; the snapshot is small.
- **Legacy-fallback masking.** A draw that *should* have provenance but lost it would silently use
  current config. Mitigation: the fallback always logs a WARNING; spec-042's report surfaces
  "provenance: stored" vs "provenance: assumed-from-current-config (legacy)" so a human sees it.
- **Tester behaviour change ripples to existing callers.** Making faithful-replay the default could
  change violation counts for callers that previously (accidentally) ran at live-config slack.
  Mitigation: DoD-9 round-trip proves fresh draws are unchanged; the only deltas are exactly the
  missed violations (false negatives from an inflated live base_slack) and false positives (from a
  reduced live base_slack) this spec exists to remove, and those are the intended change.
  (review fix — direction note: the bug produces false NEGATIVES when the convenor bumps base_slack
  upward, since higher effective slack → smaller S → fewer violations reported. The spec removes
  BOTH false negatives and false positives depending on which direction the live config has drifted.)

## Open Questions

0 — the convenor settled the semantics ("run at the draw's own slack, including base slack");
persistence location and resolver behaviour are fully determined by the existing config + metadata model.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Requires an explicit user go-ahead to START (a `ready` plan does not self-start). Once authorised, run the units end-to-end, pausing only on `blocked` or an unrecoverable failure. -->

0. **Do NOT start without an explicit user instruction to implement this plan.** `ready` ≠ "build now".
1. Status must be `ready` (carries a `reviewed:` stamp). If under review, wait. If `blocked`, STOP.
2. Only after the user says to implement: stamp `building`, claim `owner`. Orchestrator = Opus.
3. **Unit A** on `spec040-unitA` worktree off `final-form`: delegate to Opus. Gates: type-check;
   AST sweep; `pytest` on the touched versioning/checkpoint tests; `/adversarial` Mode B. Merge →
   push → tear down.
4. **Unit B** after A on `spec040-unitB`: delegate to Opus. Gates: type-check; `pytest
   tests/test_slack_provenance.py -v` (DoD-8 divergence + DoD-9 round-trip green); AST sweep;
   `/adversarial` Mode B. Merge → push → tear down.
5. When both units pass: stamp `done`, archive to `docs/todo/done/`, update
   `docs/todo/00-dependency-tree.md` (spec-041 now one dependency closer).
