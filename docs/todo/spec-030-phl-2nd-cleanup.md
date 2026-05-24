<!-- status: building -->
<!-- reviewed: adversarial Sonnet review 2026-05-24 — fixes applied inline -->
<!-- severity: S2 -->
<!-- open_questions: 0 -->
<!-- depends_on: none -->
<!-- owner: session=opus-aiupd-030 claimed=2026-05-24 -->

# spec-030 — PHL/2nd cleanup: 2.5h cross-venue gap, drop the redundant concurrency atom, locked-week parity

**Spec source:** convenor request (this session) — (1) the PHL/2nd cross-venue start-time gap should be **2.5 hours, not 3**; (2) audit the three PHL/2nd constraints for double-enforcement; (3) make the kept PHL concurrency atom **skip locked weeks like its predecessor**.

## Why

Three small, related corrections to the PHL/2nd constraint cluster, all verified against `final-form`:

1. **The cross-venue gap is 30 minutes too large.** `phl_2nd_cross_venue_min_minutes` is `180` (3 h) in `config/defaults.py:144`. The convenor wants **150** (2.5 h): game length + warm-down + travel + warm-up needs 2.5 h start-to-start, not 3. This is a single config constant read by `PHLAnd2ndAdjacency` (`constraints/atoms/phl_2nd_adjacency.py:56`) and its regen-soft sibling (`constraints/atoms/phl_2nd_adjacency_regen_soft.py:73`), each with a hard-coded `180` fallback that must move in lockstep.

2. **`PHLAnd2ndConcurrencyAtBroadmeadow` is redundant.** Verified by reading all three atoms: `PHLAnd2ndAdjacency`'s same-venue branch only *allows* a club's PHL and 2nd games when they are on the **same field in adjacent day_slots** (`p_field == q_field and abs(p_slot - q_slot) == 1`, `phl_2nd_adjacency.py:99-100`). Therefore same-slot (`abs == 0`) is **always forbidden** by Adjacency at any venue, Broadmeadow included. `PHLAnd2ndConcurrencyAtBroadmeadow` (`constraints/atoms/phl_2nd_concurrency.py`) forbids exactly that same-club same-Broadmeadow-slot case via `sum(phl) + sum(2nd) <= 1` — a strict **subset** of what Adjacency already enforces. It is dead weight in a fresh build. (`PHLConcurrencyAtBroadmeadow`, the cross-club PHL↔PHL "one PHL game on the clock at NIHC" rule, is **not** redundant — it is stronger than `NoDoubleBookingFields` and is kept.)

3. **The kept PHL concurrency atom ignores locked weeks; its predecessor didn't.** `PHLConcurrencyAtBroadmeadow` was split from the legacy `PHLAndSecondGradeTimes`, which skipped locked weeks (the "bastardised constraints" locked-week workaround). The atom (`constraints/atoms/phl_concurrency.py`) has **no** `locked_weeks` guard, unlike its sibling `PHLAnd2ndAdjacency` (`phl_2nd_adjacency.py:74`). In a locked-week / regen run it can therefore fire on frozen weeks the convenor has already fixed, which is exactly the infeasibility the predecessor's skip avoided. Add the same `locked_weeks` skip for parity.

Cost of not doing this: the 7 pm-adjacent timing is needlessly strict (clubs lose a viable cross-venue slot), the registry carries a redundant atom that confuses the inventory and the regen core-hard set, and locked-week regen runs can wrongly go INFEASIBLE on PHL concurrency.

## Definition of Done

1. `config/defaults.py` `CONSTRAINT_DEFAULTS['phl_2nd_cross_venue_min_minutes']` is `150`; the `.get(..., 180)` fallbacks in `phl_2nd_adjacency.py:56-58` and `phl_2nd_adjacency_regen_soft.py:73` are `150`; both atom docstrings say "150 = 2.5 h" (currently "180 = 3 h").
2. `PHLAnd2ndConcurrencyAtBroadmeadow` is deleted everywhere: the atom file `constraints/atoms/phl_2nd_concurrency.py` removed; its three references in `constraints/atoms/__init__.py` (import line 14, list entry line 108, `__all__` entry line 157) removed; its `ConstraintInfo` entry in `constraints/registry.py` (lines 148-156) removed. (review fix — C1/C2: two additional code deletion sites were missing from DoD and the file list — see DoD 2a and 2b below.)
   - DoD 2a: `constraints/unified.py` — import at line 29 (`PHLAnd2ndConcurrencyAtBroadmeadow` in the `from constraints.atoms import (...)` block) and dispatch entry at line 482 (inside `_PHL_HARD_ATOMS` tuple) removed; after deletion the `_PHL_HARD_ATOMS` tuple becomes a single-element tuple `(PHLConcurrencyAtBroadmeadow,)`.
   - DoD 2b: `config/defaults.py` — line 177: the `DEFAULT_STAGES` `critical_feasibility` atoms list contains `'PHLAnd2ndConcurrencyAtBroadmeadow'` as a string key; remove it (leaving only `'PHLConcurrencyAtBroadmeadow'` from that pair). This is a runtime stage config: missing this causes a KeyError/registry failure when loading stages.
3. `len(CONSTRAINT_REGISTRY) == 50` (was 51); `tests/test_constraint_registry.py:102` updated from `== 51` to `== 50`.
4. `PHLConcurrencyAtBroadmeadow` (`constraints/atoms/phl_concurrency.py`) skips locked weeks: reads `locked_weeks = set(data.get('locked_weeks', set()))` and `continue`s when `locked_weeks and week in locked_weeks` (mirroring `phl_2nd_adjacency.py:74`), where `week = key[6]`.
5. Every surviving reference to `PHLAnd2ndConcurrencyAtBroadmeadow` in live tests is removed or re-pointed: `tests/atoms/test_phl_atoms.py`, `tests/test_constraint_groups.py`, `tests/test_groups_cli_wiring.py`, `tests/test_regen_group.py`, `tests/test_run_cli.py`. In particular `test_regen_group.py`'s expected `core_hard` set no longer contains it, and any `resolve_groups(['regen'])` / `core` membership assertion is updated.
6. A no-mock CP-SAT test in `tests/atoms/test_phl_atoms.py` (inside or alongside `TestPHLConcurrencyAtBroadmeadow`) proves the locked-week skip: Given a fixture with two same-(week,day,slot) PHL games at Broadmeadow on a **locked** week, When `PHLConcurrencyAtBroadmeadow.apply` runs with `data['locked_weeks'] = {that_week}`, Then **0** constraints are added (hand oracle: the only candidate group is on a skipped week, so `groups` is empty after filtering); and Given the same fixture on a **non-locked** week (or empty `locked_weeks`), Then exactly **1** `sum <= 1` constraint is added. This test lives in the same file as the existing `TestPHLConcurrencyAtBroadmeadow` class (lines 39-70) — not a new file.
7. A no-mock CP-SAT test proves the 2.5h boundary in `PHLAnd2ndAdjacency`: Given a club fielding PHL at 11:00 and 2nd at 13:30 (150 min apart) on **different** venues, the pair is allowed (no `p+q<=1` added); Given PHL 11:00 and 2nd 13:20 (140 min, < 150), the pair is forbidden (one constraint added). (Hand oracle: `|660−810| = 150 ≥ 150` allowed; `|660−800| = 140 < 150` forbidden.)
8. The regen behaviour change is documented (Risks below + `REGEN_CONSTRAINTS.md`): with `PHLAnd2ndConcurrencyAtBroadmeadow` gone from `core_hard`, a scoped regen's same-club PHL/2nd same-Broadmeadow-slot case is governed by the **soft** `PHLAnd2ndAdjacencyRegenSoft` analogue (which mirrors the same-venue adjacency rule, so the same-slot case still incurs a penalty) rather than a hard clause. No code change to the regen-soft atom is required — confirm by reading `phl_2nd_adjacency_regen_soft.py` that its same-venue branch penalises non-adjacent (incl. same-slot) pairs.
9. Docs updated per the doc registry. Type-check clean; changed-file lint clean; AST sweep clean (no dangling import of the deleted atom, no unlogged dark path); full suite green.

## Implementation units

Single unit. The three corrections are one cohesive touch of the PHL/2nd cluster (~30 LOC + deletions + tests + docs), all within `constraints/atoms/` + `registry.py` + the PHL test module + docs. They share the registry edit and the PHL test file, so splitting would create internal file collisions for no parallelism gain. Graded S2 (deletes a public symbol across modules + a behavioural config change). One worktree.

### Unit A — 2.5h gap + delete redundant atom + locked-week skip + tests + docs

- **Files touched:**
  - `config/defaults.py` — two edits: (1) line 144: `'phl_2nd_cross_venue_min_minutes': 180` → `150`; (2) line 177: in `DEFAULT_STAGES` `critical_feasibility` atoms list, remove the string `'PHLAnd2ndConcurrencyAtBroadmeadow'` (leaving only `'PHLConcurrencyAtBroadmeadow'` from that pair). (review fix — C2: this second edit was missing from the original file list.)
  - `constraints/atoms/phl_2nd_adjacency.py` — `.get(..., 180)` → `150` (line 57); docstring lines 9-12 "180 = 3 h" → "150 = 2.5 h".
  - `constraints/atoms/phl_2nd_adjacency_regen_soft.py` — `.get(..., 180)` → `150` (line 73); docstring line 12 "180 = 3 h" → "150 = 2.5 h".
  - `constraints/atoms/phl_concurrency.py` — add the `locked_weeks` read + per-key skip (DoD 4).
  - `constraints/atoms/phl_2nd_concurrency.py` — **DELETE**.
  - `constraints/atoms/__init__.py` — remove the 3 `PHLAnd2ndConcurrencyAtBroadmeadow` references (import line 14, `PHL_TIMES_ATOMS` list entry line 108, `__all__` entry line 157). After deletion `PHL_TIMES_ATOMS` becomes a single-element list `[PHLConcurrencyAtBroadmeadow]`; no rename needed.
  - `constraints/registry.py` — delete the `PHLAnd2ndConcurrencyAtBroadmeadow` entry (lines 148-156).
  - `constraints/unified.py` — remove the import of `PHLAnd2ndConcurrencyAtBroadmeadow` at line 29 (from the `from constraints.atoms import (...)` block) and remove it from the `_PHL_HARD_ATOMS` tuple at line 482; after deletion the tuple becomes `_PHL_HARD_ATOMS = (PHLConcurrencyAtBroadmeadow,)`. (review fix — C1: this file was entirely missing from the original file list; both references here cause an `ImportError`/`NameError` at runtime if not cleaned up.)
  - `tests/atoms/test_phl_atoms.py` — drop the `TestPHLAnd2ndConcurrencyAtBroadmeadow` test class (lines 78-102) and its import (line 20); add the DoD-6 locked-week-skip test (under `TestPHLConcurrencyAtBroadmeadow`) and DoD-7 2.5h-boundary test. The locked-week-skip test belongs in this file because `TestPHLConcurrencyAtBroadmeadow` already lives here (lines 39-70).
  - `tests/test_constraint_registry.py` — line 102 count `51` → `50`.
  - `tests/test_constraint_groups.py`, `tests/test_groups_cli_wiring.py`, `tests/test_regen_group.py`, `tests/test_run_cli.py` — remove/repoint every `PHLAnd2ndConcurrencyAtBroadmeadow` reference (grep each; the `core_hard`/`regen` expected sets shrink by one).
  - `docs/system/CONSTRAINT_INVENTORY.md` — delete ALL four occurrences of `PHLAnd2ndConcurrencyAtBroadmeadow`: lines 34 (quick-reference table), 97 (legacy cluster description), 144 (cluster atom count — update "2 —" to "1 —" and drop the name), 171 (per-atom table row). Also update the `phl_2nd_cross_venue_min_minutes` mentions at lines 197 and 241 from 180/3h to 150/2.5h. (review fix — H2: the original wording "row(s)" underspecified — there are 4 distinct occurrences across different table sections.)
  - `docs/system/REGEN_CONSTRAINTS.md` — remove `PHLAnd2ndConcurrencyAtBroadmeadow` from the `core_hard` table (line 45); add a one-line note that the same-club same-Broadmeadow-slot case is now covered by the soft `PHLAnd2ndAdjacencyRegenSoft` analogue in regen; update the soft-analogue table row at line 67 ("≥180-min cross-venue rule" → "≥150-min cross-venue rule"). (review fix — M2: the line-67 soft-analogue 180→150 update was not listed.)
  - `docs/system/HARNESS.md` — line 153: update the `PHL_HARD_ATOMS`/`_phl_times_atoms_hard` row to show only `PHLConcurrencyAtBroadmeadow` remains (drop `PHLAnd2ndConcurrencyAtBroadmeadow` from the list in that cell).
  - `docs/system/FORCED_GAMES_AS_COUNT_RULES.md` — line 190: remove `PHLAnd2ndConcurrencyAtBroadmeadow` from the historical atom count note (the cluster count drops from 4 to 3 live atoms: `PHLConcurrencyAtBroadmeadow`, `PHLRoundOnePlay`, `PreferredDates` — both already deleted — so the note can simply reflect that `PHLAnd2ndConcurrencyAtBroadmeadow` is also now deleted).
  - `docs/operator-human/RULES.md` — line 72: "3-hour start-to-start gap … default 180" → "2.5-hour … default 150".
  - `docs/operator-ai/GAME_TIME_DICTIONARIES.md` — lines 177-179: update the cross-venue-gap value from 180/3h to 150/2.5h.
  - `CLAUDE.md` (repo root) — update all "180-minute"/"180, real minutes"/"≥180-min" cross-venue adjacency mentions to "150-minute"/"150, real minutes"/"≥150-min": lines 280, 399 (two occurrences in the atom description and pitfalls/constraint-details tables). (Soft-contention with spec-029, which also edits CLAUDE.md in a different section — see Dependencies.)
- **Change summary:** one config constant + two atom fallbacks/docstrings (2.5h); one atom deletion with all call-sites (redundant concurrency); one guard added (locked-week parity); tests + docs.
- **Depends on:** none.
- **Executor model:** Opus (the redundancy claim and the regen-soft interaction are subtle; the deletion must be complete across registry + atoms + `unified.py` + `defaults.py:177` + 5 test files). (review fix — C1/C2: scope updated to include the two previously missing code cleanup sites.)
- **No-mock test outline (Given/When/Then, hand-computed oracles):** as DoD 6 and 7. Build tiny `X` dicts of real 11-tuple keys + `model.NewBoolVar`s (the existing `test_phl_atoms.py` fixtures show the pattern); assert on `len(model.Proto().constraints)` deltas, not on solver output.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — remove all 4 occurrences of the deleted atom (lines 34, 97, 144, 171); correct all 180/3h cross-venue values (lines 197, 241) to 150/2.5h. (review fix — H2: original "row(s)" underspecified 4 distinct locations.)
- `docs/system/REGEN_CONSTRAINTS.md` — drop the atom from `core_hard` (line 45); note the soft-analogue now covers the same-slot case; correct 180→150 in the soft-analogue table row (line 67). (review fix — M2: line-67 update was not previously listed.)
- `docs/system/HARNESS.md` — line 153: update `_phl_times_atoms_hard` / `_PHL_HARD_ATOMS` row to list only `PHLConcurrencyAtBroadmeadow`.
- `docs/system/FORCED_GAMES_AS_COUNT_RULES.md` — line 190: remove `PHLAnd2ndConcurrencyAtBroadmeadow` from the historical cluster atom list.
- `docs/operator-human/RULES.md` (line 72) + `docs/operator-ai/GAME_TIME_DICTIONARIES.md` (lines 177-179) — 3h/180 → 2.5h/150.
- `CLAUDE.md` — all "180-minute"/"180, real minutes"/"≥180-min" cross-venue adjacency mentions (lines 280, 399) → 150-minute/150/≥150-min.
- `docs/todo/GOALS.md` — spec-030 row already present (confirmed at line 166); update status from `review_pending` to `ready` after review stamps.
- `docs/todo/00-dependency-tree.md` — spec-030 node already present (confirmed at line 17); no action needed until done.

## Out of scope

- **Touching `PHLConcurrencyAtBroadmeadow`'s core logic** beyond the locked-week skip — it stays a hard cross-club rule.
- **Re-tuning any penalty weight** — the regen-soft adjacency weight is unchanged; only its threshold (2.5h via shared config) shifts.
- **The group restructure** (`symmetry_breakers`, `core` minus spacing) — that is **spec-032**; this spec leaves all group tags as-is except by deleting one atom's entry.
- **Removing `ClubFieldConcentration`** — that is **spec-031**.
- **Archiving the legacy `PHLAndSecondGradeTimes` engine path** — still referenced by `stages.py` engine keys; not this spec's concern.

## Dependencies

- **Other plans:** `depends_on: none`. Soft-contention only: spec-029 (`building`) also edits `CLAUDE.md` (a different section) and the docs tree. CLAUDE.md edits here are section-local (the adjacency note) and will merge cleanly; if spec-029 is still in-flight at merge time, resolve the CLAUDE.md hunk by keeping both sections. No shared *code* file with spec-029.
- **Within this plan:** single unit; no internal dependencies.
- **Downstream:** spec-031 and spec-032 both edit `constraints/registry.py` and `docs/system/CONSTRAINT_INVENTORY.md`; they declare `depends_on: spec-030` so the registry/inventory edits serialise (no concurrent-edit conflict). This spec must land first.

## Risks & blast radius

- **Regen weakening (intended, documented).** Deleting the `core_hard` `PHLAnd2ndConcurrencyAtBroadmeadow` means a scoped regen no longer *hard*-forbids a club's PHL+2nd sharing a Broadmeadow slot; the soft `PHLAnd2ndAdjacencyRegenSoft` analogue penalises it instead. This is consistent with spec-027's choice to soften the adjacency rule in regen (the same-slot case is a watchability preference, not a physical impossibility — the two teams have different players and can physically play at once). DoD 8 confirms the soft analogue covers it; the regen witness test in `test_regen_group.py` must still pass.
- **Incomplete deletion.** A missed `PHLAnd2ndConcurrencyAtBroadmeadow` reference (registry, `__init__`, **`constraints/unified.py`**, **`config/defaults.py:177`**, or any of the 5 test files / multiple doc files) breaks import or a count/membership/stage-load assertion at runtime. (review fix — C1/C2: `unified.py` and `defaults.py:177` were missing from the original scope; both cause runtime failures — `unified.py` breaks import, `defaults.py:177` causes a registry KeyError at stage load.) Mitigation: grep the symbol repo-wide after the edit; the type-check + full suite are the backstop.
- **Threshold off-by-units.** `phl_2nd_cross_venue_min_minutes` is in **minutes**; 2.5 h = 150. The boundary is `>=` (`abs(p_min - q_min) >= cross_venue_min`), so exactly 150 min is allowed. DoD-7's 150-vs-140 oracle pins this.

## Open Questions

None.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Autonomous: run end-to-end without waiting for the user, except where this hits `blocked`. -->
1. Status must be `ready` (carries a `reviewed:` stamp from /adversarial Mode A). If `review_pending`/`under_review`, let review finish — do not implement. If `blocked`, STOP.
2. Stamp `building`, claim `owner`. You are the orchestrator (Opus).
3. Single unit (Unit A): own worktree+branch; implement on Opus. Run the S2 gates: type-check, changed-file lint, AST dead-code/dark-path sweep, no-mock Given/When/Then tests (DoD 6/7 with hand oracles, ≥85% on changed atom code), docs updated.
4. After implementing, launch `/adversarial` Mode B to verify the diff against this plan's DoD (especially: deletion completeness via repo-wide grep; the regen-soft same-slot coverage claim; the 150-min boundary). Route fixes, re-verify. NEVER merge unverified.
5. Merge → push origin → post-merge verify (`load_season_data(2026)` loads; a quick `--list-groups`/registry import succeeds; count == 50) → remove worktree. Tick the checkbox.
6. Stamp the plan `done`, move it to `docs/todo/done/`, update `docs/todo/00-dependency-tree.md` (drop spec-030; note spec-031 now unblocked).
