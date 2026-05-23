<!-- status: in_progress -->
<!-- reviewed: adversarial Sonnet review 2026-05-23 — fixes applied inline -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-021 (this re-scopes the ClubGameSpread hard+soft that spec-021 lands; building against the pre-spec-021 ClubGameSpread would re-introduce the heavy range/min/max IntVars spec-021 deletes). Shares constraints/registry.py, constraints/severity.py, constraints/unified.py, config/defaults.py, config/season_2026.py, analytics/tester.py with spec-021/022/023 — rebase + re-run validate_solver_stages before merge. spec-023's §1 group table lists the two removed constraints; whichever of {spec-023, spec-024} lands second drops/omits them (see Dependencies). -->
<!-- owner: session=spec-024-impl claimed=2026-05-23 -->
<!-- tier: complex -->

# spec-024 — Field-aware ClubGameSpread replaces the club-balance pair (`MaximiseClubsPerTimeslotBroadmeadow` + `MinimiseClubsOnAFieldBroadmeadow`)

## Why

The convenor is retiring two old soft "spread the clubs around" rules and replacing the
intent with a sharper, club-centric rule expressed through `ClubGameSpread`:

- **`MaximiseClubsPerTimeslotBroadmeadow`** (soft, sev 4) — at NIHC, pushes *more distinct
  clubs* into each Sunday timeslot (`min_clubs = floor(games/2) − slack`). A symmetry-breaker
  about timeslot diversity.
- **`MinimiseClubsOnAFieldBroadmeadow`** (soft, sev 4) — at NIHC, caps the number of distinct
  *clubs sharing a field* per day (`max_clubs_per_field = 5 + slack`).

Both are Broadmeadow-only, both approximate "don't let one club monopolise a field / spread
clubs sensibly" from the *venue's* point of view. The convenor wants the opposite framing —
from the *club's* point of view: a club's games on a day should sit **together on one field**,
contiguously. That is exactly what `ClubGameSpread` already does for contiguity (spec-021), but
spec-021 left it **day-scoped** (`(club, week, day)` over all fields combined) and added **no
field-spread penalty**. spec-021 explicitly listed these two rules as *out of scope* ("the
per-field cross-field balance nicety … untouched"), so closing that gap is this spec's job.

Net effect this spec delivers:
1. **Delete** `MaximiseClubsPerTimeslotBroadmeadow` and `MinimiseClubsOnAFieldBroadmeadow`
   entirely (forward-only — no shims, no archived re-wiring).
2. **Re-scope `ClubGameSpread`** from `(club, week, day)` to **`(club, week, day, field)`**: the
   contiguity hard rule (`gap_cap = max(0, min(1, n−3))`) now applies **per field** — a club's
   ≤3 games on any one field must be a contiguous block; ≥4 may carry ≤1 hole (soft pressure to
   0), mirroring spec-021's resolved decision A but per-field.
3. **Add a multi-field soft penalty** to `ClubGameSpread`: per `(club, week, day)`, the field
   holding the most of the club's games that day is its *primary* field; **every game the club
   plays on any other field that day is one unit of penalty**. This discourages a club splitting
   its day across fields and replaces the "spread clubs" intent the two deleted rules served.

Applies at **all venues** (away grounds typically have one field, so the spread penalty is
naturally zero there; the rule is general, not Broadmeadow-special-cased).

## What this is NOT

- **NOT** a change to *which timeslots* are used (earliest-fill is `VenueEarliestSlotFill`,
  spec-021) — this is purely about a club's per-field clustering.
- **NOT** a new hard cap on clubs-per-field or clubs-per-slot — those caps die with the two
  deleted rules and are **not** re-expressed as hard constraints. The only hard part added is
  per-field contiguity; field-concentration is *soft*.
- **NOT** a re-introduction of the heavy range/min/max IntVar encoding spec-021 removed. The
  per-field rule reuses the cheap `slot_used` + hole-indicator pattern from `_contiguity`.

## Design

### 1. Delete the two club-balance constraints (forward-only)

Remove every trace of `MaximiseClubsPerTimeslotBroadmeadow` and `MinimiseClubsOnAFieldBroadmeadow`
(and their `…AI` aliases and `…Soft` variants):

- `config/defaults.py` — drop both names from the `soft_optimisation` stage atom list
  (`DEFAULT_STAGES`, ~line 288).
- `config/season_2026.py` + `config/season_template.py` — drop the two `PENALTY_WEIGHTS` entries
  from `season_2026.py` (lines ~1115–1116); drop the `max_clubs_per_field` `CONSTRAINT_DEFAULTS`
  entry (season_2026 ~1022, season_template ~220) and its formula comments (~1014–1015). **Note:**
  `season_template.py` has no `PENALTY_WEIGHTS` entries for these two names — only the
  `max_clubs_per_field` CONSTRAINT_DEFAULTS entry to remove.

  > (review fix — M1: season_template.py only has `max_clubs_per_field` in CONSTRAINT_DEFAULTS,
  > not the two PENALTY_WEIGHTS entries; verified against season_template.py:220. The original
  > wording implied both files needed PENALTY_WEIGHTS edits — incorrect.)

  **`max_clubs_per_field` readers:** grep confirms it is read by: (1) `analytics/tester.py:2291`
  inside `_check_minimise_clubs_on_a_field_broadmeadow` (which Unit A deletes), (2)
  `tests/test_constraint_defaults_merge.py:17` which lists it in an `expected_keys` set that
  asserts the key must exist in `CONSTRAINT_DEFAULTS`, and (3) archived classes (not in production
  path). After deleting the tester method, `test_constraint_defaults_merge.py:17` must also be
  updated to remove `max_clubs_per_field` from `expected_keys` — it's the only remaining live
  reader. Delete the key and update that test in Unit A.

  > (review fix — H3: tester.py:2291 reads max_clubs_per_field but is deleted in Unit A; however
  > tests/test_constraint_defaults_merge.py:17 also asserts the key's presence in
  > CONSTRAINT_DEFAULTS.keys() and WILL fail if the key is removed without updating the test.
  > The original wording's conditional "iff nothing else reads it" was ambiguous — the key should
  > be removed and both its remaining readers cleaned up in Unit A.)
- `constraints/registry.py` — delete both `ConstraintInfo` entries (lines ~374–389).
- `constraints/severity.py` — delete the four severity-map entries (base + `AI`, lines ~85–88).
- `constraints/soft.py` — delete `MaximiseClubsPerTimeslotBroadmeadowSoft` (class ~867) and
  `MinimiseClubsOnAFieldBroadmeadowSoft` (class ~948), their entries in the soft-class lookup
  dict (~1139–1142) and the instantiation list (~1190–1191), and the level-4 docstring mention
  (~line 14).
- `constraints/resolver.py` — delete the imports (~192–193), the relax-map entries (~205–206),
  and the canonical-name map entries (~478–481).
- `analytics/tester.py` — delete `_check_maximise_clubs_per_timeslot_broadmeadow` (~2228) and
  `_check_minimise_clubs_on_a_field_broadmeadow` (~2281), their entries in the check-dispatch
  list (~1189–1190), and the two severity-map entries (~88–89).
- `constraints/unified.py` — delete the now-dead `bm_slot_club` and `bm_field_club` grouping
  dicts: their `__init__` declarations (lines 177–178) and the population block in
  `build_groupings()` (lines 277–285). Confirmed dead: grep shows they are *populated* but read
  by nothing in `constraints/` — the engine never had `_maximise_clubs`/`_minimise_clubs` methods
  (the rules ran only as the `soft.py` legacy soft classes). Deleting the constraints makes them
  unambiguously orphaned, so they go (forward-only).
- `constraints/stages.py` — the docstring example at line 6 names
  `MaximiseClubsPerTimeslotBroadmeadow` as a sample non-atom legacy-soft key; reword to a
  surviving example (e.g. `PreferredTimes`). Neither name is in `ENGINE_HARD_KEYS`/
  `ENGINE_SOFT_KEYS` — confirmed by reading stages.py:43-71. No key-set edits needed.

  > (review note — Low/L2: confirmed via code read that neither name is in ENGINE_HARD_KEYS or
  > ENGINE_SOFT_KEYS; the "verify" note was pre-emptively correct.)
- `constraints/archived/original.py` / `archived/ai.py` — the `Maximise…`/`Minimise…` classes
  live here (original.py ~1337/1418; ai.py ~1478/1577) and in the archived "apply all" lists
  (ai.py ~2011–2012). These are archived (blocked from production import by
  `tests/test_no_legacy_imports.py`). **Leave them** — archived code is historical and not in the
  production path; removing them is churn with no production effect. (If a registry/severity test
  asserts a class↔canonical mapping that now points at a deleted registry entry, fix the test, not
  the archive.)

> **Forward-only check:** after deletion, `grep -rn "MaximiseClubs\|MinimiseClubs\|maximise_clubs\|minimise_clubs\|max_clubs_per_field" constraints/ config/ analytics/ run.py main_staged.py` returns **only** archived-class hits and this spec's note — nothing in the live path.

> (review note — H1: `run.py:462-463` also contains hardcoded `MaximiseClubsPerTimeslotBroadmeadow`
> and `MinimiseClubsOnAFieldBroadmeadow` slack-dict entries. Unit A must also edit `run.py` to
> remove these two keys from the `constraint_slack` dict — verified at run.py:462-463. The forward-
> only grep above covers run.py, so the gap was in the Unit A file list, not this check.)

### 2. Re-scope `ClubGameSpread` to per-field contiguity

Today (`constraints/unified.py::_club_game_spread_hard`, ~1021) groups by `(club, week, day)` via
`self.by_club_week_day_slot[(club, week, day, day_slot)]` (built at unified.py ~287–291,
field-agnostic). Change the grouping key to include the field:

- New grouping `by_club_week_day_field_slot[(club, week, day, field_name)][day_slot] -> [vars]`,
  populated in `build_groupings()` **alongside** `by_club_week_day_slot` (keep the existing
  grouping — the soft off_primary pass needs per-`(club, week, day)` totals which are most
  cheaply derived from it; see the key-change consequence note below). Use
  `field_name` (the field identifier already destructured in `build_groupings`); a club's games at
  two different venues necessarily have different `field_name`s, so keying on field alone keeps
  cross-venue games in separate groups correctly (no need to also key on location).
- `_club_game_spread_hard` iterates `(club, week, day, field)` groups. For each group with ≥2
  used slots: build `slot_used` indicators (helper kind `club_spread_slot_used`, now keyed
  `(club, week, day, field, slot)` — update the key prefix passed to `slot_used_indicators`),
  build the same pref/suf/hole channels, and apply the **same** games-derived cap
  `gap_cap = max(0, min(1, n − 3)) + slack` where `n` = the club's games **on that field** that
  day. Result: ≤3 games on a field ⇒ 0 holes (strict contiguous block); ≥4 ⇒ ≤1 hole.
- The hole indicators feed the existing soft hole-count penalty (`_club_game_spread_soft`,
  ~1094) unchanged in spirit — every residual interior hole on a field is one penalty unit.

> **Key-change consequence (M2 fix):** `_club_game_spread_hard` populates `self._cgs_keys` and
> `self._cgs_hole_vars`. After the per-field rework, these must store **4-tuple** keys
> `(club, week, day, field)` — one entry per field group processed. `_club_game_spread_soft` must
> accordingly iterate 4-tuple keys for the hole-count part. For the **off_primary** computation
> (§3 below), `_club_game_spread_soft` must additionally re-aggregate per `(club, week, day)` —
> either by grouping `_cgs_keys` by the first three elements, or by maintaining a separate
> `_cgs_club_day_fields` dict populated during the hard pass (set of fields seen per
> `(club, week, day)`). Both approaches work; the simplest is grouping during the soft pass.
> `by_club_week_day_slot` may be **kept** (unchanged) alongside `by_club_week_day_field_slot`
> if the off_primary path needs it for total-game-count; alternatively, derive totals by summing
> field-group counts. Do **not** silently remove `by_club_week_day_slot` and leave the soft pass
> with no source for per-`(club, week, day)` totals.

> **Helper-catalog note:** `club_spread_slot_used` already exists in `HELPER_VAR_CATALOG`
> (registry.py ~596). Its comment must be updated from `(club, week, day, day_slot)` to
> `(club, week, day, field, day_slot)`. No new kind string is needed — the key tuple widens,
> the kind stays the same.

### 3. Add the multi-field soft penalty to `ClubGameSpread`

Within `_club_game_spread_soft` (or a new helper called from it), per `(club, week, day)`:

- Let the club's games that day be partitioned across fields `f1..fk` with counts `c1..ck`
  (`ci = sum of the club's game BoolVars on field fi that day`).
- The *primary* field is the one with the largest count. Because primary-field identity is
  itself a decision variable (the solver chooses placements), encode the penalty as
  **"games not on the most-used field"** without committing to a fixed primary:
  `off_primary = (total club games that day) − max_i(ci)`, where `max_i(ci)` is an IntVar
  constrained via `AddMaxEquality` over the per-field count IntVars. `off_primary ≥ 0` by
  construction; it is 0 when all the club's games sit on one field, and grows by one for each
  game on a non-primary field.
- Add `off_primary` (one IntVar per `(club, week, day)` that actually has games on ≥2 candidate
  fields) to the `ClubGameSpread` penalty bucket. **Reuse the existing `ClubGameSpread` penalty
  weight** — both the hole count and the off-primary count are `ClubGameSpread` soft pressure.
  (If the convenor later wants them weighted differently, that is a separate, trivial config split
  — out of scope here; one bucket now.)

> **IntVar bounds:** each per-field count IntVar has domain `[0, len(field_vars)]`; the
> `AddMaxEquality` target and `off_primary` are bounded by the club's total games that day
> (`≤ number of grades × teams`, comfortably bounded by `len(all club day vars)`). Set explicit
> upper bounds = the count of the club's vars that day so CP-SAT accepts the model.

### 4. Tester

`analytics/tester.py` — `_check_club_game_spread` (the spread check spec-021 updated to hole-count
semantics) must move to **per-field** hole counting (group games by `(club, date, field)`),
matching the solver. Add a check that flags, as a soft-pressure observation, a club using >1
field on a day (the off-primary count) so the violation report surfaces it. Keep it slack-aware
on the `ClubGameSpread` key.

## Definition of Done

1. **Both constraints deleted:** `grep -rn "MaximiseClubs\|MinimiseClubs\|maximise_clubs\|minimise_clubs"`
   over `constraints/` (excluding `archived/`), `config/`, `analytics/`, `run.py`,
   `main_staged.py` returns nothing but this spec's note. `len(CONSTRAINT_REGISTRY)` drops by
   exactly 2 from the pre-change count.
2. **No dangling references:** `import constraints.registry, constraints.severity,
   constraints.soft, constraints.resolver, constraints.unified, analytics.tester, config.defaults,
   config.season_2026` all succeed; `validate_solver_stages(DEFAULT_STAGES) == []`; full test
   suite green (update/delete any test that asserted the two deleted constraints exist — the full
   known list is in the Unit A file list above; always re-grep `tests/` for completeness).

   > (review fix — H2: the original named only 4 test files in this DoD item; 14 files are
   > actually affected. See the Unit A file list for the authoritative enumeration.)
3. **`bm_slot_club`/`bm_field_club` removed** from `unified.py` `__init__` and `build_groupings`;
   grep-clean in `constraints/`.
4. **ClubGameSpread is per-field (hard):** `_club_game_spread_hard` groups by
   `(club, week, day, field)`; the contiguity cap is `gap_cap = max(0, min(1, n−3)) + slack`
   where `n` is the club's games **on that field** that day.
5. **Multi-field penalty (soft):** `_club_game_spread_soft` adds, per `(club, week, day)` with the
   club on ≥2 fields, an `off_primary = total − max_field_count` IntVar to the `ClubGameSpread`
   penalty bucket (correct bounds; ≥0; 0 ⇔ single field).
6. **Behaviour tests (GWT, no mocks, hand oracle)** in `tests/atoms/` (or the existing
   ClubGameSpread test file):
   - *Per-field contiguity, ≤3:* a club with 3 games **on field EF** offered slots {1..6}:
     placements {1,2,3} → 0 holes, FEASIBLE; {1,2,4} → 1 hole > gap_cap 0 → **INFEASIBLE**.
     (Hole oracle: holes = (max_used − min_used + 1) − num_used, per field.)
   - *Per-field contiguity, ≥4:* 4 games on EF, slots {1..6}: {1,2,4,5} → 1 hole = gap_cap 1 →
     FEASIBLE; {1,2,5,6} → 2 holes > 1 → **INFEASIBLE**.
   - *Per-field independence:* a club with 2 games on EF in slots {1,2} and 2 on WF in slots
     {5,6} → each field contiguous → 0 holes total, FEASIBLE (no cross-field hole), but
     `off_primary` penalty = 2 (two games on the non-primary field; primary holds 2, total 4).
   - *Single-field optimum:* same club, all 4 games on EF slots {1,2,3,4} → 0 holes,
     `off_primary` = 0.
7. **Tester parity:** `_check_club_game_spread` reports per-field holes; a fixture with a
   cross-field gap that is contiguous *per field* yields **no** spread violation (regression vs the
   old day-scoped check, which would have falsely flagged it). The off-primary count appears in the
   violation report as soft pressure.
8. **Production-wiring proof:** an integration test loads the real 2026 config, builds the model
   via `DEFAULT_STAGES`, and asserts (a) neither deleted constraint is applied (0 of their
   penalties in `data['penalties']`), and (b) `ClubGameSpread` emits >0 hard per-field constraints
   AND a non-empty `ClubGameSpread` soft penalty bucket containing both hole and off-primary terms.
9. **Variable-count sanity:** on the 2026 fixture, the rewritten per-field `ClubGameSpread` plus
   the two deleted constraints together add **no more** CP-SAT IntVars than the pre-change set
   (assert before/after `model.Proto().variables` over the relevant groupings) — the efficiency
   posture spec-021 established is preserved.
10. Docs updated per the doc registry; full suite green.

## Implementation units

> Heavily-shared files (`unified.py`, `registry.py`, `severity.py`, `defaults.py`,
> `season_2026.py`, `analytics/tester.py`). Sequence as one worktree, commit per unit; rebase on
> spec-021 (and spec-022/023 if landed) before merge.

### Unit A — Delete `MaximiseClubsPerTimeslotBroadmeadow` + `MinimiseClubsOnAFieldBroadmeadow`
- Files: `config/defaults.py`, `config/season_2026.py`, `config/season_template.py`,
  `constraints/registry.py`, `constraints/severity.py`, `constraints/soft.py`,
  `constraints/resolver.py`, `analytics/tester.py`, `constraints/unified.py` (drop
  `bm_slot_club`/`bm_field_club`), `constraints/stages.py` (docstring reword), **`run.py`**
  (remove the two slack-dict entries at lines ~462–463), and every test that asserts their
  existence (grep `tests/`).

  > (review fix — H1: `run.py` was missing from the file list. It has two hardcoded slack-key
  > entries at lines 462-463 inside the `--slack` handling block. These must be removed. Verified
  > via grep run.py.)

  **Known test files that need editing** (do not rely on this list being exhaustive — always grep
  `tests/` before closing Unit A):
  - `tests/test_constraint_registry.py` — asserts the two canonical names in the registry
  - `tests/test_severity_relaxation.py` — imports and tests `MaximiseClubsPerTimeslotBroadmeadowAI` / `MinimiseClubsOnAFieldBroadmeadowAI` at level 4
  - `tests/test_solver_stages_dispatch.py` — `atom_to_engine_key('MaximiseClubsPerTimeslotBroadmeadow') is None` assertion (test input must change)
  - `tests/test_tester_coverage.py` — `TestCheckMaximiseClubsPerTimeslotBroadmeadow` and `TestCheckMinimiseClubsOnAFieldBroadmeadow` classes; also uses `max_clubs_per_field` in `make_data`
  - `tests/test_unified_engine.py` — `test_broadmeadow_slot_club_populated` asserts `len(mini_engine.bm_slot_club) > 0` (delete or rewrite once `bm_slot_club` is gone)
  - `tests/test_constraint_defaults_merge.py` — `expected_keys` set at line 17 includes `max_clubs_per_field`; remove from expected set after deleting the config key
  - `tests/test_constraints_comprehensive.py` — imports and tests `MaximiseClubsPerTimeslotBroadmeadow`; class `TestMaximiseClubsPerTimeslotBroadmeadow`
  - `tests/test_constraints.py` — imports `MinimiseClubsOnAFieldBroadmeadow`/`MaximiseClubsPerTimeslotBroadmeadow`; class `TestMinimiseClubsOnAFieldBroadmeadow`
  - `tests/test_constraints_equivalence.py` — imports both AI variants; class `TestMinimiseClubsOnAFieldEquivalence`
  - `tests/test_ai_constraints_comprehensive.py` — imports all four class names; `TestMaximiseClubsPerTimeslotBroadmeadowAI` class + integration tests that include both classes
  - `tests/test_infeasibility_resolver.py` — imports `MaximiseClubsPerTimeslotBroadmeadowAI` / `MinimiseClubsOnAFieldBroadmeadowAI`
  - `tests/test_run_coverage.py` — includes both names in `constraint_slack` dict; asserts `len(constraint_slack) == 5` (must drop count to 3 after removal)
  - `tests/test_run_cli.py` — uses `MaximiseClubsPerTimeslotBroadmeadow` as test input to `_diagnose_group_atoms`; asserts it appears in the result groups (replace with a surviving non-engine atom)
  - `tests/test_constraints_realdata.py` — contains `TestMaximiseClubsPerTimeslotBroadmeadow` and `TestMinimiseClubsOnAFieldBroadmeadow` classes

  > (review fix — H2: the DoD enumeration named only 4 test files. grep tests/ finds 14 test
  > files needing edits. The full list above was verified; archived-class tests (test_constraints.py,
  > test_ai_constraints_comprehensive.py etc.) import from `constraints/archived/` — those imports
  > stay valid, but any assertion about the REGISTRY or STAGES or TESTER that references the deleted
  > names must be removed or adapted.)

- No dependency on other units (pure removal). Land first so the registry count is correct before
  the ClubGameSpread rework re-counts.
- Test: DoD 1, 2, 3 — import smoke, registry-count drop by 2, `validate_solver_stages == []`,
  grep-clean, full suite green.

### Unit B — Field-aware `ClubGameSpread` (per-field contiguity + multi-field penalty)
- Files: `constraints/unified.py` (`build_groupings` new `by_club_week_day_field_slot`;
  `_club_game_spread_hard` per-field; `_club_game_spread_soft` off-primary penalty), 
  `constraints/registry.py` (update `club_spread_slot_used` catalog comment to include field),
  `analytics/tester.py` (`_check_club_game_spread` → per-field + off-primary observation).
- Depends on Unit A (so registry/grouping edits sit on the cleaned tree; both touch `unified.py`
  and `registry.py` → serialize).
- Test: DoD 4–9 (the GWT contiguity + off-primary fixtures, tester parity, production-wiring,
  var-count).

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — delete the `MaximiseClubsPerTimeslotBroadmeadow` and
  `MinimiseClubsOnAFieldBroadmeadow` rows; update the `ClubGameSpread` row to "per-field
  contiguity + multi-field (off-primary) soft penalty, all venues"; update §3 count + severity
  table (two fewer constraints).
- `docs/system/STAGES.md` — remove the two names from the `soft_optimisation` listing; note
  `ClubGameSpread` carries the field-concentration soft penalty now.
- `docs/system/HELPER_VARS.md` — update the `club_spread_slot_used` entry key to include `field`.
- `docs/operator-human/RULES.md` (and `docs/DRAW_RULES.md` if it carries the rule) — replace the
  "spread clubs across timeslots / cap clubs per field" wording with "a club's games on a day sit
  contiguously on one field; splitting across fields is penalised."
- `docs/operator-ai/AI_OPERATIONS_MANUAL.md` / `docs/operator-ai/CONSTRAINT_APPLICATION.md` —
  drop references to the two removed constraints; describe the field-aware `ClubGameSpread`.
- `config/season_2026.py` inline comments — remove the `MaximiseClubs…`/`MinimiseClubs…` formula
  comments (~1014–1015) and `max_clubs_per_field` comment if the key is removed.
- `CLAUDE.md` — the Constraint-System severity table lists `MaximiseClubsPerTimeslotBroadmeadow`
  and `MinimiseClubsOnAFieldBroadmeadow` at Level 4; remove both rows. Update any `--slack`
  bullet that names them. (Note: the live `CLAUDE.md` in this worktree may differ from the
  feat/ai-updates copy — edit the worktree copy.)
- `docs/todo/GOALS.md` — add the spec-024 row to the Specifications table (done in this filing;
  implementer flips status to `done` and moves the file to `done/` on completion).
- `seasons/RULES.md` — grep shows it references `MaximiseClubsPerTimeslotBroadmeadow` or
  `MinimiseClubsOnAFieldBroadmeadow`; update or remove the reference there.
- `.github/copilot-skills/hockey-draw-scheduler.md` — similarly references one or both; update.

  > (review note — Low/L3: grep found 43 files total containing references. `seasons/RULES.md` and
  > `.github/copilot-skills/hockey-draw-scheduler.md` contain references but were absent from the
  > doc registry. Low priority (non-production docs) but should be cleaned for completeness.)

## Out of scope

- **Splitting the `ClubGameSpread` penalty bucket** into separate hole vs off-primary weights —
  one bucket now; a weighted split is a trivial later config change, not scoped here.
- **A hard cap on clubs-per-field or clubs-per-slot** — the deleted rules' caps are intentionally
  *not* re-expressed as hard constraints; field concentration is soft only (convenor's framing).
- **`VenueEarliestSlotFill` / earliest-slot packing** (spec-021) — untouched; this spec only
  changes a club's per-field clustering, not which slots a venue fills.
- **spec-023's group-tag table** — spec-023 (`ready`) tags the two now-deleted constraints into
  `{soft, soft_optimisation}` (its §1 table). Removing them from that table is owned by whichever
  of {spec-023, spec-024} merges **second**: if spec-024 lands first, the spec-023 implementer
  omits the two dead names when tagging (their registry entries no longer exist); if spec-023
  lands first, Unit A here additionally removes their `groups=` tags. This is a merge-ordering
  coordination item, not deferred work — recorded in Dependencies.

## Dependencies

- **depends_on spec-021** — this spec edits the `ClubGameSpread` hard+soft that spec-021 lands.
  Must rebase on spec-021 before starting; building against the pre-spec-021 ClubGameSpread would
  fight the heavy IntVar encoding spec-021 removes.
- **Shares files with spec-022/023** (`registry.py`, `severity.py`, `unified.py`, `defaults.py`,
  `tester.py`). Rebase and re-run `validate_solver_stages` before merge. Coordinate the spec-023
  group-table edit per Out-of-scope.
- **Internal:** Unit B depends on Unit A (registry count + shared `unified.py`/`registry.py`).

## Risks & blast radius

- **Removing soft pressure changes solver objective landscape.** Deleting the two club-balance
  penalties shifts the objective; a re-solve may move some games. This is intended (the convenor
  is replacing the rule), but flag it: the next full 2026 solve will differ from the last one in
  NIHC slot/field distribution. Awareness item — no rollback shim.
- **Per-field grouping multiplies group count** (a club's day games split across up to 3 NIHC
  fields → up to 3 groups instead of 1). Variable count is bounded (DoD 9 measures it) but watch
  presolve time on the first full build.
- **`max_clubs_per_field` config key** — grep confirms its live readers are: (1) `tester.py:2291`
  (deleted in Unit A with the tester method), and (2) `tests/test_constraint_defaults_merge.py:17`
  (asserts the key exists in `CONSTRAINT_DEFAULTS` — must update this test when the key is
  removed). `scripts/` has no references. Delete the key from `config/defaults.py:135`,
  `config/season_2026.py:1022`, and `config/season_template.py:220`, and update the test. No
  scripts or reports outside tests/ read it.

  > (review fix — H3 follow-up: the original risk framing was "may be read by something" with a
  > conditional. After grepping, the answer is deterministic: the only live readers are the tester
  > method (deleted) and one test that asserts the key's presence. Both must be cleaned.)
