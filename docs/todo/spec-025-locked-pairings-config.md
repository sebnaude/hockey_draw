<!-- status: building -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: none (locked-weeks already shipped on final-form: run.py:71-79, main_staged.py:1037-1059, utils.py:3586+. Reuses the FORCED scope-count machinery in utils.py without changing it. Shares utils.py + config/defaults.py + config/season_*.py with spec-021/022/023 — rebase + re-run validate before merge.) -->
<!-- owner: session=opus-572895d-20260523T091343Z claimed=2026-05-23T09:13:43Z -->

# spec-025 — `LOCKED_PAIRINGS`: a dedicated "pin pairing to its weekend, free the time" config

## Why

When the convenor regenerates part of a published draw (a roster change drops a team a grade;
a few future weeks need re-timing), most pairings are already decided and the convenor wants
them to **stay on their weekend (their playing day) but be free to move to a different time,
slot, or field**. Today this is expressed by hand-adding a FORCED_GAMES entry per pairing —
`{teams:[t1,t2], grade, date}` → `sum==1` (the season config carries ~246 such "Locked wk7:
…" entries, `config/season_2026.py:693-938`). That **bloats FORCED_GAMES**:
(review fix — M5: actual count is ~246 per the in-file comment at line 692; line range ends at
938 not 874 — verified in draw-specs worktree 2026-05-23) the convenor's
deliberate count rules and marquee games (`{grade:'PHL', day:'Friday', count:3,
constraint:'lesse'}`, `Norths vs Wests 80th-anniversary`) are buried under a wall of
mechanical pins, and the regen tooling (spec-026) has no clean place to *write* pins without
mangling that hand-curated list.

This spec adds a **sister config `LOCKED_PAIRINGS`** with identical pin semantics but kept
physically and semantically separate from FORCED_GAMES: FORCED_GAMES = the convenor's
intentional rules; LOCKED_PAIRINGS = "freeze these already-decided pairings onto their
weekends." It is the substrate spec-026's regen mode reads from and writes to. The cost of not
doing it: FORCED_GAMES keeps growing unreadable, and regen has nowhere to put auto-extracted
pins.

### Research findings (verified against final-form)

- **The pin already works through FORCED.** A FORCED entry `{teams:[t1,t2], grade, date}` with
  no `time`/`day_slot`/`field` builds a scope of `{grade, date, _entry_idx}` + team-matcher
  `('pair', t1, t2)` and applies `model.Add(sum(matching_vars) == 1)`
  (`utils.py:3797-3812`). Because the scope omits time/slot/field, **every** variable for that
  pairing on that date (across all times/slots/fields valid for the date) is in the sum, and
  the solver freely picks exactly one — i.e. the pairing is pinned to the date with the time,
  slot, and field left free. This is exactly the behaviour we want; spec-025 reuses it.
- **The scope-count machinery is already factored for reuse.** `_build_scope_count_rules(entries,
  teams, *, label='FORCED_GAMES', unique_per_entry=False)` (`utils.py:581-748`) parses any
  FORCED-grammar list into `(scope_groups, constraint_types, constraint_counts,
  constraint_weights)`. `_get_matching_forced_scopes(key, rules)` (`utils.py:766-824`) returns
  every scope a var satisfies. `_build_forced_game_rules` (`utils.py:751-763`) is the 3-tuple
  back-compat wrapper FORCED callers use. spec-025 calls `_build_scope_count_rules` on
  `LOCKED_PAIRINGS` with `label='LOCKED_PAIRINGS', unique_per_entry=True` (every pin is its own
  scope; pins never merge).
- **Generation/enforcement site.** `generate_X` (`utils.py:3469-3834`) builds FORCED rules at
  `:3589`, registers each created var against matching scopes into `forced_scope_vars`
  (`:3724-3737`), runs the empty-scope FATAL pre-check (`:3741-3795`, `sys.exit(1)`), and
  applies the count constraints (`:3797-3812`). LOCKED_PAIRINGS will run as a **second,
  parallel scope-count pass** in the same function, with its own `locked_pairing_scope_vars`
  dict and its own apply loop hard-wired to `== 1`.
- **Locked-week filtering already exists.** `validate_game_config` strips FORCED/BLOCKED entries
  whose `date` is in a locked week (`utils.py:3313-3337`), and `generate_X` skips
  FORCED/BLOCKED matching for `in_locked_week` vars (`:3685-3737`). LOCKED_PAIRINGS gets the
  same treatment: a pin on a hard-locked past week is redundant (those weeks are already pinned
  to exact keys by `main_staged.py:1037-1056`) and must be filtered out, never double-applied.
- **Config injection point.** `build_season_data` returns the `data` dict with config lists
  injected at `utils.py:4073-4106` (`'forced_games'`, `'blocked_games'`, `'preferred_games'`,
  `'preferred_weekends'` …). LOCKED_PAIRINGS is injected here as `'locked_pairings'`.
- **Validation.** `_validate_entry_fields(entries, label, …)` (`utils.py:1194-1296`) validates
  scope fields/dates; `is_forced = (label == 'FORCED_GAMES')` decides fatal-vs-warn. Called from
  `validate_game_config` (`utils.py:3292-3464`).
  (review fix — C9: `_validate_entry_fields` is at lines 1194-1296 in the draw-specs
  worktree, not 3194-3297 as originally cited — verified 2026-05-23)

## Definition of Done

1. **New config list `LOCKED_PAIRINGS`** with an empty-list default as a top-level constant in
   `config/defaults.py` (mirroring how `PREFERRED_GAMES` is declared there — `PREFERRED_WEEKENDS`
   is NOT in defaults.py and must NOT be used as the template; see review note L1), readable as
   `data['locked_pairings']`. Entry grammar = a strict subset of the FORCED grammar:
   `teams: [t1, t2]` (or `team1`/`team2`), `grade`, `date`, optional `description`.
   **`time`, `day_slot`, `field_name`, `field_location`, `day`, `week`, `round_no`, `count`,
   `constraint` are NOT permitted** — a LOCKED_PAIRINGS entry is, by definition, "this pairing,
   this date, exactly one game, everything else free." Presence of any forbidden scope field is a
   validation FATAL (it would partially re-lock the time and defeat the config's purpose).
   (review note — Low/L1: `PREFERRED_WEEKENDS` verified absent from defaults.py 2026-05-23;
   it lives only in season_2026.py. `PREFERRED_GAMES` at defaults.py:313 is the correct model.)
2. **Injection:** `build_season_data` (`utils.py:4073-4106`) injects
   `'locked_pairings': config.get('locked_pairings', [])`. A test asserts
   `load_season_data(2026)['locked_pairings']` round-trips a configured entry.
   Also add `locked_pairings` to the season config dict key in each season file's
   `SEASON_CONFIG` dict (matching the `preferred_games` pattern already there).
   (review fix — C11 corrected by second review pass 2026-05-23: return dict is at
   lines 4073-4106 — `return {` opens at 4073 and `}` closes at 4106; the prior
   "fix" inverted the correction and introduced a wrong range 4072-4102.)
3. **Enforcement in `generate_X`:** a second scope-count pass parallel to FORCED:
   - Build rules via `_build_scope_count_rules(data.get('locked_pairings', []), teams,
     label='LOCKED_PAIRINGS', unique_per_entry=True)`.
   - As each var is created, register it against matching LOCKED_PAIRINGS scopes (reuse
     `_get_matching_forced_scopes`) into a dedicated `locked_pairing_scope_vars` dict —
     **separate from `forced_scope_vars`** so the two configs never interact and FORCED count
     math is untouched. A var may be in BOTH a FORCED scope and a LOCKED_PAIRINGS scope; both
     constraints apply to it (no elimination, additive — same as a var matching two FORCED
     scopes today).
   - Skip registration for `in_locked_week` vars (mirror `:3685-3737`).
   - Apply `model.Add(sum(vars) == 1)` per LOCKED_PAIRINGS scope.
   - Note: `_build_scope_count_rules` returns a **4-tuple**
     `(scope_groups, constraint_types, constraint_counts, constraint_weights)`. The
     LOCKED_PAIRINGS apply loop only needs `scope_groups` (the other three are
     unused — pins are always hard `== 1`). Assign all four to avoid a tuple-
     unpack error: `lp_scope_groups, _lp_ctypes, _lp_counts, _lp_weights = ...`
     (review fix — H3: return-tuple arity corrected; the FORCED back-compat
     wrapper strips to 3-tuple but direct `_build_scope_count_rules` returns 4)
4. **Empty-scope handling = FATAL with a LOCKED_PAIRINGS-labelled diagnostic** (a pin whose
   pairing has zero placeable vars on its date is a real config/feasibility error, exactly like
   FORCED). Reuse the FORCED diagnostic shape (`utils.py:3741-3795`) but labelled
   `LOCKED_PAIRINGS` and listing the offending `(teams, grade, date)`. Do NOT silently drop.
5. **Locked-week filtering:** `validate_game_config` strips LOCKED_PAIRINGS entries whose `date`
   is in a locked week (extend the existing `_not_in_locked_week` filter at `utils.py:3313-3337`
   to also filter `locked_pairings`). Verified by a test: with weeks 1-3 locked, a pin dated in
   week 2 produces no constraint and no FATAL.
6. **Validation:** `_validate_entry_fields` accepts label `'LOCKED_PAIRINGS'` and treats it like
   FORCED for fatal-vs-warn on bad dates/grades/team-resolution (`is_forced` path), PLUS the
   new forbidden-field check (DoD 1). The `is_forced` check at line 1198 must be extended from
   `is_forced = (label == 'FORCED_GAMES')` to
   `is_forced = label in ('FORCED_GAMES', 'LOCKED_PAIRINGS')`.
   Wire a `validate_game_config` phase that calls it with the `LOCKED_PAIRINGS` label and
   passes `data.get('locked_pairings', [])` so FORCED/BLOCKED phases don't accidentally process
   pins. A pin with a `time`/`field`/`day_slot` key → FATAL listing the forbidden field; a pin
   with an unknown team → FATAL (same as FORCED).
   Also update the `has_config_to_validate` guard (line 3345) to include `locked_pairings`
   in the OR-check, so validation runs when the only config is a LOCKED_PAIRINGS list.
   Also add `locked_pairings = data.get('locked_pairings', [])` read at the top of
   `validate_game_config` (alongside `forced_games` and `blocked_games` at lines 3305-3308)
   and store the filtered list back: `data['locked_pairings'] = locked_pairings` after the
   locked-week filter (mirror the store-back at lines 3336-3337).
   (review fix — H2+M1+M2+M4: is_forced extension, store-back, guard, read — all
   required but missing from original Unit A description)
7. **Metadata:** draw metadata gains `locked_pairing_outcomes` (per pin: matched var count, the
   chosen game's resolved time/slot/field, `satisfied: true/false`) alongside the existing
   `forced_game_outcomes`, written by `save_solver_output` / the tester. A regenerated draw lets
   the convenor audit that every pinned pairing kept its date.
8. **Tester check:** `analytics/tester.py` gains a `_check_locked_pairings` check (registered in
   `constraints/registry.py` as a `tester_only` entry, like `ForcedGames`/`BlockedGames`) that
   flags any pinned pairing not present on its date in the finished draw.
   `len(CONSTRAINT_REGISTRY)` goes from **37 → 38**; `test_registry_has_expected_entry_count`
   at `tests/test_constraint_registry.py:94` must be updated from `== 37` to `== 38`.
   (review fix — H4 corrected by second review pass 2026-05-23: prior fix said "38 at line 88;
   new count 39" — BOTH values were wrong. Verified in constraints/registry.py: there are 37
   canonical_name entries (spec-024 deleted 2 entries, net 37). The test assert is at line 94,
   not line 88 (line 88 is inside the docstring). Correct oracle: 37 → 38.)
9. Full suite green; `import utils, config.defaults, analytics.tester` smoke clean.

## Implementation units

> Shared file `utils.py` is touched by Units A and B; sequence them on one worktree, commit per
> unit. Units C/D are independent of each other but both depend on B.

### Unit A — Config list, default, injection, validation
- Files: `config/defaults.py` (`LOCKED_PAIRINGS = []` constant + a one-line docstring),
  `config/season_2026.py` / `season_2025.py` / `season_2027.py` / `season_template.py`
  (`LOCKED_PAIRINGS = []` placeholder so every season exposes the key; do NOT migrate the
  existing FORCED locked-pairing entries here — that migration is Unit E),
  `utils.py::build_season_data` (inject `'locked_pairings'`),
  `utils.py::_validate_entry_fields` (accept `'LOCKED_PAIRINGS'` label + forbidden-field check),
  `utils.py::validate_game_config` (add the LOCKED_PAIRINGS validation phase + extend the
  locked-week filter at `:3313-3337`).
- Test (`tests/test_locked_pairings_config.py`, GWT, no mocks, hand oracle):
  `load_season_data(2026)['locked_pairings']` round-trips an entry; a pin with `time` → FATAL
  naming the forbidden field; a pin with an unknown team → FATAL; a pin dated in a locked week →
  filtered out (no FATAL, no constraint, and `data['locked_pairings']` is empty after filter).
  Also test: `validate_game_config` with a non-empty `locked_pairings` (and empty
  forced/blocked/club_days etc.) still enters the validation body — verifying the
  `has_config_to_validate` guard was updated correctly.
  (review fix — M4: test the guard update)

### Unit B — `generate_X` enforcement pass
- Files: `utils.py::generate_X` (build LOCKED_PAIRINGS rules; register into
  `locked_pairing_scope_vars`; empty-scope FATAL; apply `sum==1`).
- Depends on Unit A.
- Test (`tests/test_locked_pairings_generate.py`, GWT, no mocks, hand oracle on a tiny CP-SAT
  fixture): GIVEN a pairing with 4 candidate vars on one date (2 times × 2 fields) and a pin on
  that date, WHEN `generate_X` runs THEN exactly one of the 4 is forced (sum==1) and the other 3
  remain free; the chosen var's time/field is solver-determined (assert any single var =1 is a
  valid solution, two =1 is infeasible). GIVEN a pin whose pairing has zero vars on its date,
  THEN `generate_X` exits FATAL with a `LOCKED_PAIRINGS` diagnostic naming the pairing. GIVEN a
  var matching both a FORCED scope and a LOCKED_PAIRINGS scope, THEN both constraints register
  it and `forced_scope_vars` is unchanged from the FORCED-only baseline (hand-count the FORCED
  sum).

### Unit C — Metadata + tester check + registry
- Files: `analytics/storage.py` / `analytics/versioning.py` (`locked_pairing_outcomes`),
  `analytics/tester.py` (`_check_locked_pairings`; also add to the ordered check-list at
  line ~1190 alongside `ForcedGames`/`BlockedGames`), `constraints/registry.py` (`tester_only`
  registry entry + count; update `test_constraint_registry.py:94` assert from `== 37` to `== 38`).
- Depends on Unit B.
- Test: a generated draw records `locked_pairing_outcomes`; the tester flags a deliberately
  date-moved pinned pairing; `test_every_drawtester_check_in_registry` and the registry-count
  test both pass with the new count of 38.
  (review fix — H4 corrected by second review pass 2026-05-23: assert is at line 94 not 88;
  pre-spec count is 37 not 38; post-spec count is 38 not 39.)

### Unit E — Migrate existing FORCED locked-pairing entries out of FORCED_GAMES
- Files: `config/season_2026.py` (move the ~246 `"Locked wk…"` `{teams, grade, date}` FORCED
  entries — `config/season_2026.py:693-938`, identify them by the absence of
  `time`/`field`/`count`/`constraint` and the `Locked` description marker — into
  (review fix — M5: ~246 entries, line range 693-938, not ~180 entries ending at 874)
  `LOCKED_PAIRINGS`; leave genuine count rules and marquee games in FORCED_GAMES).
- Depends on Unit B (the new pass must exist before the entries are moved or they'd be silently
  unenforced).
- Test (`tests/test_locked_pairings_migration_parity.py`): a full 2026 `generate_X` build AFTER
  the migration produces the **same** set of `sum==1` pinning constraints (same scopes, same
  matched var sets) as the pre-migration FORCED-only build — proving the move is behaviour-
  preserving. Hand-verify the count of pinning constraints matches the number of moved entries
  (~246, per the comment at `season_2026.py:692`). The parity test must compare both the
  `forced_scope_vars` set (FORCED path, pre-migration) against the combined
  `forced_scope_vars + locked_pairing_scope_vars` sets (post-migration).

## Doc registry

- `docs/system/FORCED_GAMES_AS_COUNT_RULES.md` — new section: `LOCKED_PAIRINGS` is the sister
  config for mechanical date-pins (pairing + date, time/slot/field free); FORCED_GAMES is for
  count rules + marquee games. State the rule "if all you want is 'this pairing, this weekend,
  any time' use LOCKED_PAIRINGS, not FORCED."
- `docs/system/CONSTRAINT_INVENTORY.md` — add the `LockedPairings` `tester_only` row; update §3
  count.
- `docs/operator-ai/CONFIGURATION_REFERENCE.md` + `docs/operator-human/RULES.md` — document the
  `LOCKED_PAIRINGS` grammar and its forbidden fields.
- `CLAUDE.md` — add `LOCKED_PAIRINGS` to the FORCED/BLOCKED table in §3 (Variable Filtering) with
  its "pin date, free time" semantics; note regen tooling (spec-026) writes to it.
- `docs/todo/GOALS.md` — spec-025 row already present (verified 2026-05-23); update status to
  `done` when complete. (review note — Low/L2: "add the row" was inaccurate; the row is there)
- `docs/todo/00-dependency-tree.md` — spec-025 already registered (verified 2026-05-23); update
  status marker when complete. (review note — Low/L2)

## Risks & blast radius

- **A second scope-count pass in `generate_X` is the hot loop.** If LOCKED_PAIRINGS rule-matching
  is run per-var alongside FORCED, build time grows with the number of pins (regen can produce
  hundreds). Risk is build-time, not correctness; surface it — the per-var match is already O(scopes)
  for FORCED and pins are disjoint single-pair scopes, so the added cost is bounded by pin count.
- **Double-counting if a var lands in both a FORCED and a LOCKED_PAIRINGS scope.** Mitigated by
  separate `locked_pairing_scope_vars` dict and a parity test (Unit B/E) asserting FORCED sums
  are byte-identical to the FORCED-only baseline.
- **Migration (Unit E) silently un-enforcing a pin** if an entry is moved but the new pass has a
  bug. Mitigated by the migration-parity test (Unit E) comparing constraint sets before/after.
- **Pre-solve feasibility gap for LOCKED_PAIRINGS.** `validate_game_config` runs
  `_check_forced_game_feasibility` (Phase 18) which simulates variable filtering to catch
  FORCED empty-scopes *before* the solver starts. No equivalent check is added for
  LOCKED_PAIRINGS by Unit A: the empty-scope FATAL fires at `generate_X` time (DoD item 4)
  rather than at validation time. This means a broken pin configuration is not caught until
  variable generation begins. Acceptable for now (same symptom, different timing), but
  Unit B could optionally add a Phase 18-style pre-check for LOCKED_PAIRINGS in
  `validate_game_config` to give earlier diagnostics. Not blocking.
  (review note — M3: known gap, deliberately deferred to implementation discretion)
- **`unique_per_entry=True` is redundant but harmless.** LOCKED_PAIRINGS entries always carry
  `teams`, so `_build_scope_count_rules` already injects `_entry_idx` via the
  `if raw_teams or has_team1_team2...` branch (utils.py:676). Passing `unique_per_entry=True`
  is a no-op but makes the intent explicit.
  (review note — Low/L3: redundancy is safe; keep for documentation clarity)

## Out of scope

- **Auto-extracting pins from an existing draw** — that is spec-026 (regen mode); spec-025 only
  provides the config + enforcement that spec-026 writes into. Hand-authored `LOCKED_PAIRINGS`
  entries are fully usable on their own once this lands.
- **Pinning to a *week* (allowing Friday↔Sunday movement) instead of a *date*** — deliberately
  excluded; the convenor wants "same playing day, different time," which a date-pin gives. A
  week-level pin is a different (and not-requested) semantic.
- **Soft pins** (penalise instead of hard-force a pairing onto its date) — that is the regen
  soft-constraint group's job (spec-027), applied to *other* constraints, not to the pins
  themselves. Pins stay hard.
- **Changing FORCED_GAMES semantics or the `soft_only`/groups machinery** — untouched here
  (spec-023's domain).

<!-- reviewed: adversarial Sonnet review 2026-05-23 (SECOND PASS — prior stamp was applied without under_review cycle, invalidating it; this is the legitimate first review) — fixes applied inline -->
<!-- review summary:
     PRIOR-REVIEW FIXES (verified correct by this pass):
       C9: _validate_entry_fields lines corrected (now 1194-1296, verified ✓)
       M5: season_2026.py locked entries 693-938 / 246 entries (verified ✓)
       H2: is_forced extension to LOCKED_PAIRINGS label (correct ✓)
       H3: 4-tuple return from _build_scope_count_rules (verified ✓)
       M1: store-back data['locked_pairings'] (correct ✓)
       M2: has_config_to_validate guard update (correct ✓)
       M4: guard-update test (correct ✓)
       M3: pre-solve feasibility gap noted (correct ✓)
       L3: unique_per_entry redundancy noted (correct ✓)
     NEW FINDINGS AND FIXES IN THIS PASS:
       C1 (Critical): "review fix C11" had inverted the correction — it claimed
         4072-4102 but actual code has return dict at 4073-4106. Fixed to 4073-4106.
       C2 (Critical): registry count oracle doubly wrong — prior fix said "38 at
         line 88, new count 39." Actual: assert == 37 at LINE 94 (spec-024 deleted
         2 entries). Post-spec count = 38, not 39. Fixed in DoD item 8 and Unit C.
       L1 (Low): PREFERRED_WEEKENDS is NOT in defaults.py (only season_2026.py);
         removed it from the analogy in DoD item 1; PREFERRED_GAMES is the correct model.
       L2 (Low): GOALS.md and 00-dependency-tree.md already have spec-025 rows;
         changed "add" to "update when done" in Doc registry. -->
