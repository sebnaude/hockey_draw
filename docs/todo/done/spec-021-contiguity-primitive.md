<!-- status: done -->
<!-- owner: session=goal-final-form claimed=2026-05-23 -->
<!-- tier: complex -->
<!-- Open decisions A/B/C resolved by convenor 2026-05-23 — see "Resolved decisions" section. -->
<!-- depends_on: spec-016 (NIHC fill → soft) and spec-017 (spacing→hard, the soft_only-trap fix pattern). Shares config/defaults.py DEFAULT_STAGES + constraints/unified.py + constraints/registry.py with spec-018; rebase + re-run validate_solver_stages before merge. -->

# spec-021 — Shared contiguity pattern: anchored monotone-fill (venue) + floating no-gap (club); drop the heavy IntVar encodings

## Why

Three constraints implement "don't leave holes between used timeslots" at different scopes,
in three different (and two of them expensive) ways:

- **`EnsureBestTimeslotChoices`** (archived class + engine `_best_timeslot_choices_hard`, per **venue**):
  The **archived class** (`constraints/archived/original.py`, `archived/ai.py`) uses a pure
  `AddImplication` chain (each `next_slot` indicator implies each `prev_slot` indicator across all
  fields — exactly `enforce_monotone_fill`'s semantics). No `AddDivisionEquality` there.
  The **unified engine method** `_best_timeslot_choices_hard` (`constraints/unified.py`) is the
  expensive path: it adds slot-number bounding via `AddDivisionEquality(quot, nlg, num_fields)`,
  a `BROADMEADOW_MAX_SLOTS`-triggered `eq` IntVar, slot-number IntVars per slot, plus the same
  implication chain for stacking. Both code paths implement the same anchored rule (games pack
  into the *earliest* slots); the engine version adds the slot-cap logic on top.
  (review fix — C1: original claim wrongly said the archived class used AddDivisionEquality/nts/ceil;
  those are only in the unified engine method. Both paths are replaced by VenueEarliestSlotFill —
  the intent is unchanged, the attribution was wrong.)
- **`ClubGameSpread`** (engine `_club_game_spread_hard`, per (club, week, day)): a **floating**
  bounded-gap rule (`spread = range − num_used ≤ max_gap`, default 2) PLUS a lower
  no-double-up bound (`num_games − num_used ≤ max_overlap`, default 0). Heavy: min/max
  reification + range/gap/overlap IntVars per club·week·day.
- **`ClubDayContiguousSlots`** (atom, per club-day): a **floating** strict no-gap rule, encoded
  cleanly with `slot_used` indicators and `prev + next ≤ 1 when mid empty` — no IntVar
  arithmetic. This is already the efficient pattern.

Problems:
1. **Two of the three don't run in production.** `EnsureBestTimeslotChoices` and
   `ClubGameSpread` are wired only into the `soft_only` `soft_optimisation` stage, so
   `apply_stage_1_hard()` is skipped (`stages.py::apply_solver_stage`) — their HARD parts are
   dead. The earliest-packing guarantee the convenor relies on is currently NOT enforced.
   (Same trap spec-017 fixed for spacing.)
2. **The expensive encodings are unnecessary.** Anchored fill and no-gap both reduce to an
   implication chain over `slot_used` indicators — the `ClubDayContiguousSlots` pattern — which
   drops every `AddDivisionEquality`/min/max IntVar (the costly part of the engine method).
3. **The WF soft rule is obsolete.** `_best_timeslot_choices_soft` only does the "last-slot
   single field should be WF" penalty (`BestTimeslotWF`), which duplicates `NIHCFillWFBeforeEF`
   (spec-003/016) → **drop it**.
   **7pm penalty:** The archived `EnsureBestTimeslotChoices` / `EnsureBestTimeslotChoicesAI`
   classes emit `EnsureBestTimeslotChoices_7pm` soft penalties (see `archived/original.py` line
   612 and `archived/ai.py` line 759). The unified engine's `_best_timeslot_choices_hard/_soft`
   do NOT emit a 7pm penalty — only the archived classes do. With anchored fill enforced as HARD,
   **packing into the earliest slots structurally avoids 7pm** — so no separate 7pm rule is
   needed. **Do not re-add it.** The `tests/test_best_timeslot_stacking.py` `TestSevenPmPenalty`
   tests assert `EnsureBestTimeslotChoices_7pm` in `data['penalties']`; these must be updated or
   deleted as part of Unit B (see implementation units).
   (review fix — C2: original claim said "7pm penalty was lost in the atomization" — false; it
   lives in the archived classes and is exercised by TestSevenPmPenalty tests. Correctly noted
   here so Unit B implementer knows what to clean up.)
4. **The lower no-double-up bound in `ClubGameSpread`** ("a club isn't in the same slot twice,
   across any of its grades") is **concurrency, not contiguity** — it belongs with the
   concurrency atoms, not bundled into a spread rule.

**Design principle (and the trap to avoid):** anchored (venue) and floating (club) are
genuinely different rules — merging them into one atom would be the re-monolithization
GOALS §2 explicitly forbids. So we **share the efficient pattern via a helper module**
(`constraints/atoms/_contiguity.py`), exactly like `_club_day_shared.py`, and keep the atoms
separate. We add the same cheap encoding everywhere; we do **not** force one rule to be
another.

## The shared primitive

New module `constraints/atoms/_contiguity.py` exposing the indicator pattern + the two
distinct semantics (no shared atom, just shared building blocks).

The helper uses the **pool-style** `HelperVarRegistry` API (`registry.get_or_create_bool`)
consistent with how `ClubDayContiguousSlots` currently uses it — not the declarative
`declare/freeze` API. This matches the existing pattern in `_club_day_shared.py`.

```
slot_used_indicators(registry, vars_by_slot, kind, *key_prefix) -> dict[int, BoolVar]
    # one channeled `slot_used` BoolVar per slot (OR/max of that slot's vars), via the
    # pool registry keyed (kind, *key_prefix, slot) so callers dedupe. (The existing
    # ClubDayContiguousSlots / EnsureBestTimeslotChoices indicator-building, factored out.)
    # Uses: registry.get_or_create_bool((kind, *key_prefix, slot), vars_list, label)

enforce_no_gaps(model, slot_inds) -> int        # FLOATING: strict no-hole.
    # for each consecutive triple (prev, mid, next) in sorted slots:
    #   model.Add(prev + next <= 1).OnlyEnforceIf(mid.Not())
    # block of used slots is contiguous but can start anywhere.

enforce_monotone_fill(model, slot_inds) -> int   # ANCHORED: no-hole + earliest-start.
    # for each consecutive pair (s_prev, s) in sorted slots:
    #   model.AddImplication(slot_inds[s], slot_inds[s_prev])
    # using slot s requires the earlier slot be used → packs into earliest available slots.
    # (Strictly stronger than enforce_no_gaps.)
```

Note we deliberately do NOT provide a "bounded gap ≤ N" primitive in the cheap form — see Open
decision (the only place `ClubGameSpread` differs from strict no-gap).

## Definition of Done

1. `constraints/atoms/_contiguity.py` exists with `slot_used_indicators`, `enforce_no_gaps`,
   `enforce_monotone_fill`, each unit-tested in isolation (GWT, hand oracle: 3-slot fixtures
   proving the exact constraints added and feasible/infeasible cases).
2. New atom **`VenueEarliestSlotFill`** (`constraints/atoms/venue_earliest_slot_fill.py`)
   replaces `EnsureBestTimeslotChoices`'s hard behaviour: per (week/date, venue) build a
   **combined-field** `slot_used` indicator (OR across that venue's fields per slot) and call
   `enforce_monotone_fill`. This single chain reproduces no-gaps + earliest-packing + the
   venue-level effect of cross-field stacking, with **zero** `AddDivisionEquality`/`nts`/`eq`
   IntVars (from the unified engine path) and no `BROADMEADOW_MAX_SLOTS` special-case.
   Registered at **severity 2** (HARD; it replaces a feasibility guarantee, not a preference;
   severity 5 is "VERY LOW" in the registry — inappropriate for a hard structural rule).
   (review fix — M1: original text had garbled "severity 5→ but HARD"; clarified to severity 2.
   Severity 5 = VERY LOW per `CONSTRAINT_TO_SEVERITY`; a hard earliest-fill belongs at 2 or 3.)
   Wired into a **hard** stage (not `soft_only`).
3. The soft `BestTimeslotWF` rule is **deleted** (`_best_timeslot_choices_soft`, the
   `BestTimeslotWF` penalty bucket, weight key in `config/defaults.py`). WF preference is owned
   solely by `NIHCFillWFBeforeEF` (spec-016). No 7pm penalty is added.
4. `_best_timeslot_choices_hard`/`_soft` engine methods removed from `constraints/unified.py`;
   `EnsureBestTimeslotChoices` engine-key removed from `ENGINE_HARD_KEYS` and `ENGINE_SOFT_KEYS`
   in `constraints/stages.py`; registry entry replaced by `VenueEarliestSlotFill`. `BROADMEADOW_MAX_SLOTS`
   class constant removed from `UnifiedConstraintEngine`. The three grouping dicts populated
   exclusively for the deleted methods (`slot_vars_by_location`, `slot_field_vars`,
   `games_per_location`) must be removed from `build_groupings()` in `unified.py`, and
   `self._loc_fields = None` removed from `__init__`.
   (review fix — H1: stages.py ENGINE_HARD_KEYS / ENGINE_SOFT_KEYS / ALL_ENGINE_KEYS entries for
   `EnsureBestTimeslotChoices` must also be removed; `stages.py` was missing from the file list.
   review fix — H2: three grouping dicts in build_groupings() + `_loc_fields`/__init__ cleanup
   were not mentioned but must happen to avoid dead code.)
5. `ClubDayContiguousSlots` refactored to call `slot_used_indicators` + `enforce_no_gaps`
   (behaviour-identical; it's the reference impl). Its `tests/atoms/test_club_day_atoms*.py`
   stay green unchanged.
6. `ClubGameSpread` refactored (per resolved decision A): the UPPER contiguity HARD rule allows
   `gap_cap = max(0, min(1, n - 3))` holes where `n` = the club's games in the (club, week, day)
   group — i.e. ≤3 games → 0 holes via the cheap `enforce_no_gaps` chain, ≥4 games → ≤1 hole via
   a minimal hole indicator. The heavy `_club_game_spread_hard` min/max/range/gap/spread IntVars
   are removed; the cap is encoded with `slot_used` indicators + (only when `n ≥ 4`) a single
   bounded hole count. A **soft** penalty drives residual holes toward 0 (kept, lightened to a
   pure hole-count penalty). The LOWER no-double-up bound is **extracted** to `ClubNoConcurrentSlot`
   (DoD 6b) — not in the spread rule.
6b. New atom **`ClubNoConcurrentSlot`** (`constraints/atoms/club_no_concurrent_slot.py`, hard, per
   resolved decision B): per (club, week, day_slot) the count of the club's games across all
   grades/teams ≤ `cap`, where `cap = max(1, ceil(n_loc / no_field_slots_<location>))` and `n_loc`
   = the club's games at that location that day. `no_field_slots_<location>` is a new config value
   **derived from `DAY_TIME_MAP`** (distinct game-times per location/day; NIHC Sunday = 8,
   Maitland Park = 6, Central Coast = 2). The derivation is a config-layer function (in
   `config/defaults.py` or `config/__init__.py`), not hardcoded per-venue; adding a venue or a
   time in `DAY_TIME_MAP` updates it automatically. Registered hard (severity 2) and wired into a
   hard stage. Behaviour test (GWT, hand oracle): a club with 2 games at a 2-time venue (Central
   Coast) → cap 1, both-in-one-slot INFEASIBLE; a club with 3 games at the same 2-time venue →
   cap `ceil(3/2)=2`, two-in-one-slot FEASIBLE but three-in-one INFEASIBLE.
7. Staging fixed: `VenueEarliestSlotFill` runs in a **hard** stage (so earliest-packing is
   actually enforced — the whole point). The `ClubGameSpread` hard portion likewise runs in a
   hard stage if kept hard (no longer dead in `soft_only`). `validate_solver_stages == []`.
8. Behaviour checks (GWT, no mocks, hand oracle):
   - venue with 2 games / 2 fields offered slots {1,2,3}: a solution placing a game in slot 3
     while slot 1 empty is **INFEASIBLE**; placing in slot 1 is FEASIBLE (earliest-pack).
   - venue with games in slots {1,3} but not 2 → **INFEASIBLE** (no gap).
   - Hand oracle uses holes = (max_used − min_used + 1) − num_used and gap_cap = max(0, min(1, n−3)).
   - club with **3** games offered slots {1..6}: {1,2,3} → 0 holes, FEASIBLE; {1,2,4} → 1 hole >
     gap_cap 0, INFEASIBLE.
   - club with **4** games offered slots {1..6}: {1,2,4,5} → 1 hole = gap_cap 1, FEASIBLE;
     {1,2,5,6} → 2 holes > 1, INFEASIBLE.
9. Variable-count check: on the 2026 fixture, the rewritten atoms add **strictly fewer**
   CP-SAT IntVars than the old engine methods (assert before/after `model.Proto().variables`
   for the venue + club groupings) — the efficiency win is measured, not assumed.
10. Full suite green; inventory + severity + stages docs updated; grep-clean for
    `_best_timeslot_choices`, `BROADMEADOW_MAX_SLOTS`, `BestTimeslotWF`, `AddDivisionEquality`
    (in this area), `EnsureBestTimeslotChoices_7pm`.
    (review fix — M2: added `EnsureBestTimeslotChoices_7pm` to the grep-clean list; that penalty
    key is produced by the archived classes and tested in TestSevenPmPenalty — it must vanish.)
11. **Production-wiring proof (not just unit fixtures).** An integration test loads the real
    2026 config, builds the model via the actual `DEFAULT_STAGES` dispatch, and asserts each
    rewritten rule **actually emits hard constraints in production** — i.e. it is reachable
    through `apply_stage_1_hard()`, not stranded in a `soft_only` stage. Concretely: assert the
    `VenueEarliestSlotFill` (and the kept-hard club rules) appear in a non-`soft_only` stage in
    `DEFAULT_STAGES`, and that a full-config build adds > 0 of their hard constraints. This is
    the explicit guard against the trap that currently leaves `EnsureBestTimeslotChoices` /
    `ClubGameSpread` hard parts dead.
12. **Systemic guard (key-level).** Add a test that fails if any **hard engine key** in
    `ENGINE_HARD_KEYS` is reachable in `DEFAULT_STAGES` ONLY from `soft_only` stages. The check
    must be at the engine-KEY level, not the atom level: collect, per hard key, the union of
    stages of every atom that maps to it (`atom_to_engine_key`), and assert that union includes
    at least one non-`soft_only` stage. (Key-level avoids false positives on legitimately-soft
    atoms that share a hard key with sibling hard atoms — e.g. `PreferredDates` shares
    `PHLAndSecondGradeTimes` with the concurrency atoms that DO sit in `critical_feasibility`.)
    Verified scope at plan time: the only genuinely-stranded keys are `ClubGameSpread` and
    `EnsureBestTimeslotChoices` (both fixed by this spec); `EqualMatchUpSpacing` was the third
    and is already fixed by spec-017. If the guard surfaces any other stranded key, list it in
    the report and spawn a follow-up plan — do not widen this spec.

## Resolved decisions (convenor, 2026-05-23)

All three Open decisions are now resolved by the convenor. The resolutions below are binding
on the DoD; where they tighten or extend a DoD item, the DoD item is updated to match.

- **A. Club gap → bounded by a games-derived cap (NOT a fixed 0 or 2).** The per-(club, week,
  day) contiguity HARD rule allows `gap_cap = max(0, min(1, n - 3))` holes, where `n` is the
  number of that club's games in the group (its games that day at that venue/contiguity scope).
  Concretely: a club with ≤3 games → **0 holes** (strict no-gap); a club with ≥4 games →
  **at most 1 hole**. A **soft** penalty pushes any residual hole toward 0 (so even when 1 hole
  is permitted, the solver prefers none). Rationale (convenor): with only 3 games a tight block
  is always achievable; past 4 games one unavoidable hole can occur, so forcing 0 would risk
  infeasibility. This is a bounded-gap-≤1-conditional rule, so it needs a minimal hole indicator
  (NOT the pure `enforce_no_gaps` chain) when `n ≥ 4`; for `n ≤ 3` it reduces to the cheap
  `enforce_no_gaps` chain. The heavy `ClubGameSpread` min/max/range IntVars are still removed —
  we encode the cap with `slot_used` indicators + a single bounded hole-count, not the old
  range/spread arithmetic.
- **B. Lower no-double-up → extract to hard `ClubNoConcurrentSlot`, capacity-aware.** Extract
  the cross-grade no-double-up to a new atom `ClubNoConcurrentSlot` (per (club, week, day_slot):
  the club's games across all grades/teams in one timeslot ≤ a capacity-aware cap) in a **hard**
  stage. **Caveat (convenor):** a flat "≤1 per slot" is wrong when a club has more games at a
  venue than the venue has timeslots — then double-ups are *forced*. So the cap is
  `max(1, ceil(club_games_at_location_that_day / no_field_slots_<location>))`, where
  `no_field_slots_<location>` is a **new config value derived from `DAY_TIME_MAP`** (the count of
  distinct game *times* at that location/day — e.g. NIHC Sunday = 8, Maitland Park = 6, Central
  Coast = 2). The derivation lives in config (computed from the existing field-slot-time entries),
  not hardcoded, so adding a venue or time auto-updates it. This complements
  `SameGradeSameClubNoConcurrency` (same-grade) by covering the cross-grade club case.
- **C. Venue fill → HARD, severity 2.** Confirmed. `VenueEarliestSlotFill` is hard at severity 2;
  the earliest-packing guarantee is what lets us drop the 7pm rule entirely (no late-slot soft
  penalty re-added).

## Implementation units

> Sequence: A → (B, D in parallel-ish but both touch unified.py/registry, so serialize) → C.
> One worktree, commit per unit; rebase on spec-016/017/018 before merge.

### Unit A — Shared `_contiguity` helper + tests
- Files: `constraints/atoms/_contiguity.py` (new), `tests/atoms/test_contiguity_primitive.py`.
- No production wiring yet. Test each function on tiny CP-SAT fixtures with hand oracles.
- Registry `HELPER_VAR_CATALOG` addition: add `'venue_slot_used'` and `'club_spread_slot_used'`
  as new kind strings (to distinguish from the existing `club_day_slot_used`).

### Unit B — `VenueEarliestSlotFill` atom (replace EnsureBestTimeslotChoices)
- Files:
  - `constraints/atoms/venue_earliest_slot_fill.py` (new)
  - `constraints/atoms/__init__.py`
  - `constraints/registry.py` (replace `EnsureBestTimeslotChoices` entry with `VenueEarliestSlotFill`;
    add new helper kinds to `HELPER_VAR_CATALOG`)
  - `constraints/severity.py` (replace `EnsureBestTimeslotChoices`/`AI` entries)
  - `constraints/unified.py` (remove `_best_timeslot_choices_hard/_soft` methods; remove
    `BROADMEADOW_MAX_SLOTS` class constant; remove `self._loc_fields = None` from `__init__`;
    remove `slot_vars_by_location`, `slot_field_vars`, `games_per_location` dicts from
    `build_groupings()`; remove `EnsureBestTimeslotChoices` from `apply_stage_1_hard()` and
    `apply_stage_2_soft()` dispatch blocks)
  - `constraints/stages.py` (remove `EnsureBestTimeslotChoices` from `ENGINE_HARD_KEYS`,
    `ENGINE_SOFT_KEYS`; add `VenueEarliestSlotFill` as a non-engine atom — it dispatches via
    the legacy-class fallback path in `apply_solver_stage`)
  - `config/defaults.py` (DEFAULT_STAGES: move `VenueEarliestSlotFill` to a HARD stage, remove
    `EnsureBestTimeslotChoices`; remove `BestTimeslotWF` weight key from
    `PENALTY_WEIGHTS`-style config if present; `worst_timeslot_time` default — remove only after
    confirming no other production code reads it; see review note below)
  - `analytics/tester.py` (replace `_check_ensure_best_timeslot_choices` with an earliest-fill
    check that verifies `VenueEarliestSlotFill` semantics)
  - `tests/test_best_timeslot_stacking.py` — **UPDATE OR DELETE** this file. The
    `TestSevenPmPenalty` tests assert `EnsureBestTimeslotChoices_7pm` in `data['penalties']`;
    these tests are exercising the archived classes and will need updating to reflect that:
    (a) the 7pm penalty key is gone, and (b) the new atom uses `enforce_monotone_fill` semantics.
    The stacking/contiguity tests (TestPerFieldContiguity, TestCrossFieldStacking,
    TestEarlySlotPush) must be rewritten to target `VenueEarliestSlotFill`.
    (review fix — H3: test_best_timeslot_stacking.py was missing from the file list; it directly
    tests EnsureBestTimeslotChoices/AI from archived/ and has TestSevenPmPenalty tests that will
    fail post-cleanup.)
  - `tests/test_constraint_defaults_merge.py` — update the `expected_keys` set at line 22 if
    `worst_timeslot_time` is removed from `CONSTRAINT_DEFAULTS` (test currently checks for it).
    (review fix — M3: this test file was missing; it will break if worst_timeslot_time is deleted
    without updating the test.)
- Depends on Unit A. Test per DoD 2, 3, 8 (venue cases), 9.
- (review note — Low: `worst_timeslot_time` default in `config/defaults.py` is checked by
  `tests/test_constraint_defaults_merge.py` line 22 + line 70. If it's kept as a dead default
  for backward-compat, leave the test alone. If truly removed, update the test simultaneously.)

### Unit C — `ClubGameSpread` refactor + extract `ClubNoConcurrentSlot`
- Files:
  - `constraints/unified.py` (`_club_game_spread_hard/_soft`) — replace heavy range/spread IntVars
    with `slot_used` indicators + games-derived `gap_cap = max(0, min(1, n-3))` (cheap chain when
    n≤3; single bounded hole-count when n≥4); lighten soft to a pure hole-count penalty.
  - `constraints/atoms/club_no_concurrent_slot.py` (new, per resolved decision B) — capacity-aware
    per-(club, week, day_slot) cap `max(1, ceil(n_loc / no_field_slots_<location>))`.
  - `config/defaults.py` — add a `no_field_slots_for_location(location, day='Sunday')` helper (or
    `NO_FIELD_SLOTS` dict computed from `DAY_TIME_MAP`) returning the distinct-time count per
    location; wire `ClubNoConcurrentSlot` into a hard stage; keep `club_game_spread_*` defaults
    (gap now derived, soft weight retained).
  - `config/__init__.py` — surface `no_field_slots` into the loaded `data` dict so the atom reads
    it via `data`, derived (not hardcoded). Confirm the cleanest single source at implement time.
  - `constraints/registry.py`, `constraints/severity.py` (register `ClubNoConcurrentSlot` hard sev 2).
  - `analytics/tester.py` (spread check updated to hole-count semantics; add `ClubNoConcurrentSlot`
    capacity check).
- Depends on Unit A. Test per DoD 6, 6b, 8 (club cases), 9; concurrency test for the extracted atom
  with the Central-Coast 2-time hand oracle.

### Unit D — Point `ClubDayContiguousSlots` at the shared helper
- Files: `constraints/atoms/club_day_contiguous_slots.py`.
- Depends on Unit A. Behaviour-identical; existing club-day atom tests must stay green
  (parity by construction).

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — replace the `EnsureBestTimeslotChoices` row with
  `VenueEarliestSlotFill`; update `ClubGameSpread` (lower-overlap extracted); add
  `ClubNoConcurrentSlot`; note the shared `_contiguity` helper; update §3 count + severity table.
- `docs/system/HELPER_VARS.md` — document the `slot_used` indicator kind + the contiguity
  primitive (anchored vs floating).
- `docs/system/STAGES.md` — venue-fill + club-spread now in hard stages (no longer dead in
  `soft_only`).
- `docs/operator-human/PERENNIAL_RULES.md` + `CLAUDE.md` Draw-Review-Checklist — the "last
  game on WF" and "avoid 7pm" review notes: WF is owned by `NIHCFillWFBeforeEF`; 7pm avoidance
  is now structural via earliest-fill. Update the checklist wording.
  (review fix — Low: original path `docs/PERENNIAL_RULES.md` is wrong; in this worktree the
  file is at `docs/operator-human/PERENNIAL_RULES.md`.)
- `docs/operator-human/RULES.md` or `docs/DRAW_RULES.md` — describe earliest-slot packing
  (venue) vs game clustering (club). (Check which of these exists in the worktree; only one
  will be the live operator-human rules doc.)
- `docs/todo/GOALS.md` — add spec-021 row; record the "share the pattern, keep atoms separate"
  decision as a worked example of the §2 "extract a helper, don't merge" rule.
- `docs/CONFIGURATION_REFERENCE.md` / `docs/ai/CONFIGURATION_REFERENCE.md` (whichever exists in
  this worktree) — document the new `no_field_slots_<location>` derived config (what it is, that
  it is computed from `DAY_TIME_MAP`, and that `ClubNoConcurrentSlot` reads it).

## Out of scope

- Re-adding any 7pm penalty (earliest-fill makes it moot — confirmed with the convenor).
- The per-field cross-field "balance fields within a slot" nicety (owned by
  `MaximiseClubsPerTimeslotBroadmeadow` / `MinimiseClubsOnAFieldBroadmeadow`) — untouched.
- A general "bounded gap ≤ N as a cheap primitive" — only built if Open decision A requires it.
- Atomising the *other* monolithic engine methods (`_equal_games_balanced_matchups`, etc.) —
  no bloat benefit (shared `_cache`), separate concern.
- The systemic "hard parts parked in soft_only" audit beyond these two atoms — if more are
  found, spawn a dedicated audit plan rather than widening this one.
