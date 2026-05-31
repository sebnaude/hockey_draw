<!-- status: ready -->
<!-- reviewed: adversarial Sonnet review 2026-05-30 — fixes applied inline -->
<!-- severity: S2 -->
<!-- open_questions: 0 -->
<!-- depends_on: none -->

# spec-039 — Machine-checked registry ↔ tester alignment gate

**Spec source:** convenor request (this session) — "ensure the draw we produce can be properly
analysed to assess that the constraints ARE doing what they should be doing … assess the alignment
of the draw tests to the atoms now as they may have fallen out of alignment." This spec is the
foundation of the analysis-engine plan-set (spec-039…042): it makes atom↔check alignment a
self-enforcing invariant so the report (spec-042) can be trusted, and so soft-outcome measurement
(spec-041) has a verified surface to build on.

## Why

`constraints/registry.py` is already the single source of truth that maps each constraint atom to
its post-hoc verification: each `ConstraintInfo` entry carries `tester_check_methods`,
`tester_violation_names`, `slack_key`, `has_soft_component`, `tester_only`, and `groups`. A reverse
index `_METHOD_TO_CANONICAL` and `get_checks_for_applied_constraints()` already consume it.

A manual audit this session found the mapping **currently aligned** for hard constraints (every
production atom's `tester_check_methods` resolve to real `DrawTester._check_*` methods; the eight
slack-aware checks read the right keys). But that alignment is enforced by *nobody* — it is one
rename, one new atom, or one deleted tester method away from silently drifting, and the only thing
that catches drift today is a human reading both files side by side. The convenor is explicitly
asking for confidence that the analysis is faithful; that confidence has to be a gate, not a vibe.

Two concrete drift-surfaces exist right now:

1. **Soft atoms with empty `tester_check_methods`.** `PreferredWeekendsAwayGround` and
   `SoftLexMatchupOrdering` (the two production atoms in the `default`/`production` group) plus 13
   `regen_soft` atoms produce penalty terms but have no post-hoc check. Today that is
   indistinguishable (to any tooling) from "someone forgot to wire the check." There is no field that
   says *why* an atom has no check, so a gate cannot tell a deliberate omission from a bug.
   (review fix — C1: `AwayClubHomeWeekendsCount` is NOT in this set. It is a HARD atom with
   `has_soft_component=False` and it maps to `_check_fifty_fifty_home_away`. The plan's claim that
   its "soft per-club component" has an empty check is incorrect — confirmed by reading
   `constraints/registry.py` line 93–100 and `analytics/tester.py`. The 13 `regen_soft` atoms are
   the additional empty-check set; they share the `is_pure_soft` exemption already in
   `test_all_entries_have_required_fields`. All 15 empty-check atoms are enumerated in DoD-2.)
2. **Penalty-weight orphans.** `has_soft_component` and PENALTY_WEIGHTS membership are maintained by
   hand in two files; nothing asserts they agree (a soft atom whose penalty bucket is missing from
   PENALTY_WEIGHTS contributes nothing to the objective and would never be noticed).

The fix is a registry-integrity test that runs in the normal suite, plus one new registry field that
lets an atom legitimately declare "no post-hoc pass/fail check, and here is why."

## Definition of Done

### Registry surface

1. **New `ConstraintInfo` field `no_tester_check_reason: Optional[str] = None`** added to the
   dataclass in `constraints/registry.py`. Semantics: when an atom legitimately has no
   `tester_check_methods`, this field carries a one-line human reason (e.g. `"symmetry tie-break —
   no observable pass/fail outcome"`). An atom with empty `tester_check_methods` AND
   `no_tester_check_reason is None` is a **gate failure**.
2. **Every currently-empty-check atom gets a `no_tester_check_reason`** set honestly.
   The exact set to label (verified against live `constraints/registry.py` — 15 atoms total):

   **Production atoms (appear in `default`/`production` groups):**
   - `SoftLexMatchupOrdering` → `"alphabetical tie-break; no observable post-hoc outcome"`.
   - `PreferredWeekendsAwayGround` → `"soft outcome measured by the analysis engine (spec-041), not a pass/fail tester check"`.

   **Regen-soft atoms (appear only in the `regen` group via `regen_soft`):**
   All 13 `*RegenSoft` atoms (`PHLAnd2ndAdjacencyRegenSoft`, `AwayClubHomeWeekendsCountRegenSoft`,
   `ClubVsClubStackedWeekendsRegenSoft`, `ClubVsClubStackedCoLocationRegenSoft`,
   `EqualMatchUpSpacingRegenSoft`, `BalancedByeSpacingRegenSoft`,
   `ClubDayParticipationRegenSoft`, `ClubDayIntraClubMatchupRegenSoft`,
   `ClubDayOpponentMatchupRegenSoft`, `ClubDaySameFieldRegenSoft`,
   `ClubDayContiguousSlotsRegenSoft`, `ClubGameSpreadRegenSoft`,
   `VenueEarliestSlotFillRegenSoft`) → `"regen-soft objective-only analogue; no pass/fail tester
   check — post-hoc outcome measured by deviation from corresponding hard constraint's check"`.

   Note: `AwayClubHomeWeekendsCount` is **NOT** in this set — it is a hard atom
   (`has_soft_component=False`) with `tester_check_methods=['_check_fifty_fifty_home_away']`.
   (review fix — C1: corrects the original plan's misidentification of AwayClubHomeWeekendsCount
   as a soft atom with empty checks.)

   Setting this reason is a *labelling* act only — it does not add measurement (that is spec-041).
3. **No behaviour change to constraint application, the objective, or solving.** This spec edits the
   registry dataclass + data and adds tests/docs only. Grep confirms `no_tester_check_reason` is read
   only by the new gate (and later by spec-041).

### Alignment gate (the deliverable)

4. **New test module `tests/test_registry_tester_alignment.py`** that runs in the default suite
   (picked up by `scripts/run_green_suite.py`) and asserts ALL of:
   - **(a) Forward resolvability.** For every registry entry, each name in `tester_check_methods`
     resolves to a real callable on `DrawTester` (`hasattr` + `callable`). Hand-oracle: the audit
     enumerated 24 live `_check_*` methods; the test recomputes the set from the class, never hard-codes it.
     Note: `tests/test_constraint_registry.py::test_every_tester_check_method_covered` already
     asserts this invariant. The new test module re-asserts it through `audit_registry_alignment`
     for spec-042's consumption — the gate is not removing the existing test.
   - **(b) Reverse coverage.** Every `DrawTester` method matching `^_check_` is referenced by at
     least one registry entry's `tester_check_methods`, OR appears in an explicit, documented
     allow-list constant `NON_CONSTRAINT_CHECK_METHODS` (for genuine helpers that aren't atom
     checks). A `_check_*` method in neither place is a gate failure (catches an orphaned/renamed check).
     Note: `tests/test_constraint_registry.py::test_every_drawtester_check_in_registry` already
     asserts this invariant without the allow-list; as of this writing `NON_CONSTRAINT_CHECK_METHODS`
     will be an empty set (all 24 `_check_*` methods are currently in the registry). The allow-list
     is future-proofing for helpers that might be added with a `_check_` prefix. The existing test
     is NOT superseded — the new module adds structured output for spec-042.
     (review note — Low: implementer should start with `NON_CONSTRAINT_CHECK_METHODS = frozenset()`
     and add an inline comment explaining it will remain empty unless a genuine non-constraint helper
     is added with a `_check_` prefix.)
   - **(c) Slack-key integrity.** Every non-None `slack_key` on a registry entry is (i) a key the
     solver actually populates — cross-checked against the `constraint_slack` dict constructed in
     `run.py` (the slack-aware key set) — and (ii) a key the tester actually reads. No registry
     `slack_key` is orphaned on either side. Hand-oracle: the four solver slack keys are
     `EqualMatchUpSpacingConstraint`, `ClubGameSpread`, `ClubNoConcurrentSlot`, `BalancedByeSpacing`
     (verified in `run.py` this session); the test imports the live set rather than hard-coding it,
     and fails if registry `slack_key`s and the solver set diverge.
   - **(d) Production-group check coverage.** Every atom selected by any *production* derived group
     (`default`/`production`/`regen` — resolved via the existing `resolve_groups`) either has ≥1
     `tester_check_methods` OR a non-empty `no_tester_check_reason`. No production atom is silently
     unverifiable.
     Hand-oracle (verified against live registry): `default`/`production` has exactly 2 atoms with
     empty checks (`SoftLexMatchupOrdering`, `PreferredWeekendsAwayGround`); `regen` adds 13 more
     (`*RegenSoft` atoms). All 15 will have `no_tester_check_reason` set by Unit A, so this
     assertion passes after Unit A. (review fix — C2: the regen group also contains empty-check
     atoms; DoD-4d must cover `regen` as stated, not only `default`/`production`.)
   - **(e) tester_only consistency.** Every `tester_only=True` entry has empty `solver_class_names`;
     every `tester_only=False` entry has ≥1 `solver_class_names`.
     Hand-oracle (verified): `tester_only=True`: `ForcedGames`, `BlockedGames`, `LockedPairings`
     (all have empty `solver_class_names`). No `tester_only=False` entry has empty
     `solver_class_names`. Both halves currently pass. Note: `tests/test_constraint_registry.py::
     test_non_tester_only_have_solver_names` already asserts the second half; Unit A's
     `audit_registry_alignment` adds structured output covering both halves for spec-042.
     (review note — Low: existing test covers half of DoD-4e already; implementer should verify
     it is not being silently duplicated in the new module.)
   - **(f) Penalty-weight ↔ soft consistency.** Every atom with `has_soft_component=True` (or in the
     `soft`/`regen_soft` groups) maps to a penalty bucket present in `PENALTY_WEIGHTS`, and every
     non-structural `PENALTY_WEIGHTS` key (excluding `dummy_slots`) is claimed by at least one soft
     atom. Orphans on either side fail. The atom→bucket mapping is read from the new
     `penalty_bucket` field on `ConstraintInfo` (added by Unit A — see Risks section; confirmed
     absent from current dataclass).
     (review fix — C3: "exactly one" is too strict — `NIHCFillWFBeforeEF` and `NIHCFillEFBeforeSF`
     both map to the same `nihc_fill_order` bucket; changed to "at least one".)
     (review fix — C4: no `penalty_bucket` field exists on `ConstraintInfo` today — confirmed by
     inspecting `constraints/registry.py` dataclass fields. The audit function CANNOT read an
     atom→bucket mapping from registry metadata because it does not exist. Unit A MUST add a
     `penalty_bucket: Optional[str] = None` field to `ConstraintInfo` and populate it for every
     `has_soft_component=True` entry. The Risks section already flags this as possible; this fix
     makes it a hard Unit A deliverable, not a conditional. Known mapping to populate:
     `BalancedByeSpacing→'BalancedByeSpacing'`, `TeamConflict→'TeamConflict'`,
     `TeamPairNoConcurrency→'TeamPairNoConcurrency'` (NOTE: this bucket is ABSENT from
     `PENALTY_WEIGHTS` in `config/season_2026.py` — the atom uses a hardcoded fallback of 1000;
     Unit A must either add the bucket to `PENALTY_WEIGHTS` or explicitly handle the orphan in the
     gate), `NIHCFillWFBeforeEF→'nihc_fill_order'`, `NIHCFillEFBeforeSF→'nihc_fill_order'`,
     `ClubNoConcurrentSlot→'ClubNoConcurrentSlot'`, `SoftLexMatchupOrdering→'soft_lex_ordering'`,
     `PreferredWeekendsAwayGround→'preferred_weekends_away_ground'`,
     `PreferredGames→'preferred_games'`, and the 13 `*RegenSoft` atoms per the `regen_*` keys in
     `PENALTY_WEIGHTS`.)
     (review fix — C5: the "unclaimed PENALTY_WEIGHTS keys" side also has known orphans today:
     `EqualMatchUpSpacing`, `ClubGameSpread`, `PreferredTimesConstraint` are read by the legacy
     engine path in `constraints/unified.py`, NOT by `has_soft_component=True` atoms; and
     `ClubGradeAdjacencyConstraint` is a legacy key for the archived/obsolete constraint. The
     gate must handle these by either (a) widening the scope to include engine-path atoms
     without `has_soft_component`, or (b) introducing an `engine_penalty_bucket` concept, or
     (c) maintaining an explicit legacy-bucket allow-list constant in `analytics/registry_audit.py`
     (e.g. `LEGACY_ENGINE_PENALTY_BUCKETS = {'EqualMatchUpSpacing', 'ClubGameSpread',
     'PreferredTimesConstraint', 'ClubGradeAdjacencyConstraint'}`). Option (c) is simplest and
     most honest — implementer must pick and document one approach; this is not optional.)
5. **Each assertion failure message names the offending atom/method and which invariant broke** (so
   a future drift points straight at the fix), and is logged through the project logger at WARNING
   when the gate is run as a report rather than a test (see DoD-6).
6. **A reusable `audit_registry_alignment(data) -> AlignmentReport` function** in a new module
   `analytics/registry_audit.py` returns the structured result (lists of: unresolved methods, orphan
   tester methods, orphan slack keys, uncovered production atoms, tester_only inconsistencies,
   penalty orphans). The test in DoD-4 asserts every list is empty; spec-042's report renders this
   same `AlignmentReport` as its "analysis-integrity" panel. (One computation, two consumers — the
   gate and the report — so the report can never claim alignment the gate would reject.)

### Self-verifying count

7. **The test recomputes all sets from live code** (registry entries, `DrawTester` methods, the
   `run.py` slack-key set, `PENALTY_WEIGHTS` keys) — it hard-codes no atom list, so it cannot rot as
   atoms are added/removed in later specs. It additionally asserts the live registry entry count
   matches the count recorded in `CONSTRAINT_INVENTORY.md` (currently 49 per spec-036/037 notes —
   confirmed: `len(CONSTRAINT_REGISTRY) == 49` and `tests/test_constraint_registry.py` line 103
   already asserts this), failing loudly if the inventory doc and the registry disagree.
   Note: `tests/test_constraint_registry.py::test_registry_has_expected_entry_count` already
   asserts the 49 count against the live registry. The new module's count cross-check should
   additionally parse `CONSTRAINT_INVENTORY.md` for its stated count and compare — this is the NEW
   value-add (the existing test only asserts the constant 49 inline, not cross-referenced to the
   doc). (review note — Low: implement the CONSTRAINT_INVENTORY.md parse as a simple regex on the
   `\*\*N entries\*\*` pattern already present in that file, e.g. line 164.)

## Implementation units

### Unit A — Registry field + honest reasons + audit function

- **Files touched:** `constraints/registry.py` (add `no_tester_check_reason: Optional[str] = None`
  AND `penalty_bucket: Optional[str] = None` fields to `ConstraintInfo`; set both on all relevant
  entries per DoD-2 and DoD-4f), `analytics/registry_audit.py` (new — `audit_registry_alignment` +
  `AlignmentReport` dataclass + the `NON_CONSTRAINT_CHECK_METHODS` allow-list +
  `LEGACY_ENGINE_PENALTY_BUCKETS` allow-list — or equivalent design per DoD-4f C5).
  (review fix — C4: `penalty_bucket` field is a hard Unit A deliverable, not optional. Also add
  `TeamPairNoConcurrency` bucket to `PENALTY_WEIGHTS` in `config/season_2026.py` OR document its
  deliberate absence via the gate's allow-list mechanism — this is a known orphan that will cause
  DoD-4f to fail without resolution.)
  Additional file for `TeamPairNoConcurrency` bucket: `config/season_2026.py` (add
  `'TeamPairNoConcurrency': 1000` to `PENALTY_WEIGHTS` to match the atom's hardcoded fallback,
  OR add `TeamPairNoConcurrency` to an explicit `PENALTY_WEIGHTS_FALLBACK_ONLY` set in the audit).
- **Change summary:** purely additive registry metadata + a pure introspection function. No solver,
  objective, or tester behaviour touched.
- **Depends on:** none within plan.
- **Suggested executor:** Opus (touches the shared registry dataclass; the audit logic must reason
  about derived-group resolution and the penalty-bucket mapping correctly — erring to Opus on the line).
- **No-mock test outline:** covered by Unit B's module (the audit function is exercised end-to-end
  against the live registry, not a fixture).

### Unit B — Alignment gate test + inventory doc

- **Files touched:** `tests/test_registry_tester_alignment.py` (new), `docs/system/CONSTRAINT_INVENTORY.md`
  (add an "alignment status" column / note: each atom's check method(s) or its `no_tester_check_reason`).
- **Change summary:** the DoD-4 gate, asserting every `AlignmentReport` list is empty against the
  live registry + live `DrawTester` + live `run.py` slack set + live `PENALTY_WEIGHTS`; plus the
  DoD-7 count cross-check.
- **Depends on:** Unit A merged.
- **Suggested executor:** Sonnet (mechanical once the audit function exists; oracles are "every list empty").
- **No-mock test outline:**
  - *Given* the real `load_season_data` data dict and the live registry, *when* `audit_registry_alignment(data)`
    runs, *then* all six finding-lists are empty (hand-oracle: current codebase is aligned per the
    session audit).
  - *Given* a deliberately-broken copy of the registry (an atom whose `tester_check_methods` names a
    non-existent method, constructed inline — NOT a mock, a real `ConstraintInfo` instance), *when*
    audited, *then* the unresolved-methods list contains exactly that atom. (Proves the gate bites.)
  - *Given* a soft atom stripped of both its check and its reason, *when* audited, *then* the
    uncovered-production-atom list contains it.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — add the per-atom alignment-status column and the
  `no_tester_check_reason` values; note the new gate as the enforcement mechanism.
- `docs/system/TESTING.md` — register `tests/test_registry_tester_alignment.py` as a standing
  invariant gate (what it asserts, why it exists).
- `docs/todo/00-dependency-tree.md` — add spec-039 as a live entry.
  (review note — Low: spec-039 is ALREADY listed in `docs/todo/00-dependency-tree.md` at lines
  117–122 and in the dependency graph. The doc registry entry here is redundant but harmless —
  the unit executor should verify the entry exists and update it rather than duplicate it.)
- `docs/todo/GOALS.md` — add the spec-039 row.

## Out of scope

- **Adding the missing soft-outcome *measurements*** (PreferredWeekendsAwayGround etc.) — that is
  spec-041. This spec only *labels* such atoms with `no_tester_check_reason`; spec-041 replaces the
  label with a real metric where one is measurable.
- **Slack-value provenance / historical replay** — spec-040.
- **The report rendering** — spec-042 (which consumes this spec's `AlignmentReport`).
- **Re-deriving or changing any constraint's semantics** — this spec asserts the *mapping* is
  truthful, not that any constraint is *correct*.

## Dependencies

- `depends_on: none`. Independent of spec-040 (they share no edited file — spec-039 edits
  `registry.py` + adds `analytics/registry_audit.py`; spec-040 edits `versioning.py`/`tester.py`).
  Both can run in parallel now.
  (review fix — C4: Unit A's scope now also touches `config/season_2026.py` to resolve the
  `TeamPairNoConcurrency` bucket orphan — spec-040 also touches `tester.py` but NOT
  `season_2026.py`, so there is no new contention introduced.)
- Within this plan: Unit B depends on Unit A (needs the audit function + the field).

## Risks & blast radius

- **Allow-list abuse.** `NON_CONSTRAINT_CHECK_METHODS` could become a dumping ground that hides real
  drift. Mitigation: the list is small, each entry carries an inline comment justifying it, and Mode-B
  review checks each entry is genuinely a non-constraint helper.
- **Penalty-bucket mapping ambiguity (DoD-4f).** Confirmed: no `penalty_bucket` field exists on
  `ConstraintInfo` today (verified by reading the dataclass in `constraints/registry.py`). Unit A
  MUST add `penalty_bucket: Optional[str] = None` to the dataclass and populate it for every soft
  atom. Two additional wrinkles confirmed by code inspection: (a) `TeamPairNoConcurrency` reads
  `PENALTY_WEIGHTS['TeamPairNoConcurrency']` but that key is absent from `config/season_2026.py`
  (the atom falls back to 1000 — a silent orphan); (b) four PENALTY_WEIGHTS keys
  (`EqualMatchUpSpacing`, `ClubGameSpread`, `PreferredTimesConstraint`,
  `ClubGradeAdjacencyConstraint`) are consumed by the legacy engine path in `constraints/unified.py`
  but have no `has_soft_component=True` registry entry — they require an explicit
  `LEGACY_ENGINE_PENALTY_BUCKETS` allow-list in `analytics/registry_audit.py`. These are additive
  resolutions only.
  (review fix — C3/C4/C5: this risk is now a resolved hard requirement, not a contingency.)
- **Inventory count coupling (DoD-7).** Tying the test to the `CONSTRAINT_INVENTORY.md` count means
  every future atom add/remove must update that doc — which is already the house rule (specs 030–037
  all did). Acceptable, and arguably a feature.

## Open Questions

0 — the gate's invariants are fully determined by the existing registry schema and the session audit;
no product decision is pending.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Requires an explicit user go-ahead to START (a `ready` plan does not self-start). Once authorised, run the units end-to-end, pausing only on `blocked` or an unrecoverable failure. -->

0. **Do NOT start without an explicit user instruction to implement this plan.** `ready` means
   "authorised to be built when asked", not "build now". If you arrived here straight off
   authoring/review with no user go-ahead, STOP and ask.
1. Status must be `ready` (carries a `reviewed:` stamp). If `review_pending`/`under_review`, let
   review finish. If `blocked`, STOP.
2. Only after the user says to implement: stamp `building`, claim `owner`. You are the orchestrator (Opus).
3. **Unit A** on its own worktree+branch off `final-form` (`spec039-unitA`): delegate to Opus.
   Gates: type-check; AST dead-code sweep; `/adversarial` Mode B on the diff. Merge → push → tear down.
4. **Unit B** after A on `spec039-unitB`: delegate to Sonnet. Gates: type-check + `pytest
   tests/test_registry_tester_alignment.py -v` (must be green against the live codebase); changed-file
   lint; `/adversarial` Mode B on the diff. Merge → push → tear down.
5. When both units pass: stamp the plan `done`, archive to `docs/todo/done/`, update
   `docs/todo/00-dependency-tree.md` (drop the node; note that spec-041 is now one dependency closer).
