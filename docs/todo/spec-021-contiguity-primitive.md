<!-- status: ready -->
<!-- owner: session=none claimed=none -->
<!-- depends_on: spec-016 (NIHC fill → soft) and spec-017 (spacing→hard, the soft_only-trap fix pattern). Shares config/defaults.py DEFAULT_STAGES + constraints/unified.py + constraints/registry.py with spec-018; rebase + re-run validate_solver_stages before merge. -->

# spec-021 — Shared contiguity pattern: anchored monotone-fill (venue) + floating no-gap (club); drop the heavy IntVar encodings

## Why

Three constraints implement "don't leave holes between used timeslots" at different scopes,
in three different (and two of them expensive) ways:

- **`EnsureBestTimeslotChoices`** (engine `_best_timeslot_choices_hard`, per **venue**):
  no-gap + slot-number bounding (`nts = ceil(games/fields)+1`, with a `BROADMEADOW_MAX_SLOTS`
  `AddDivisionEquality` special-case) + cross-field stacking. This is an **anchored** rule:
  games pack into the *earliest* slots (using slot s requires the earlier slots be used).
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
   drops every `AddDivisionEquality`/`ceil`/min/max IntVar (the costly part).
3. **The WF soft rule and 7pm penalty are obsolete.** `_best_timeslot_choices_soft` only does
   the "last-slot single field should be WF" penalty (`BestTimeslotWF`), which duplicates
   `NIHCFillWFBeforeEF` (spec-003/016) → **drop it**. The legacy 7pm penalty was lost in the
   atomization, but with anchored fill enforced as HARD, **packing into the earliest slots
   structurally avoids 7 pm** — so 7pm needs no separate rule. **Do not re-add it.**
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
distinct semantics (no shared atom, just shared building blocks):

```
slot_used_indicators(registry, vars_by_slot, kind, *key_prefix) -> dict[int, BoolVar]
    # one channeled `slot_used` BoolVar per slot (OR/max of that slot's vars), via the
    # pool registry keyed (kind, *key_prefix, slot) so callers dedupe. (The existing
    # ClubDayContiguousSlots / EnsureBestTimeslotChoices indicator-building, factored out.)

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
   IntVars and no `BROADMEADOW_MAX_SLOTS` special-case. Registered (severity 5→ but HARD;
   see DoD 6), wired into a **hard** stage.
3. The soft `BestTimeslotWF` rule is **deleted** (`_best_timeslot_choices_soft`, the
   `BestTimeslotWF` penalty bucket, weight key). WF preference is owned solely by
   `NIHCFillWFBeforeEF` (spec-016). No 7pm penalty is added.
4. `_best_timeslot_choices_hard`/`_soft` engine methods removed; `EnsureBestTimeslotChoices`
   engine-key dispatch + skip-checks removed; registry entry replaced by `VenueEarliestSlotFill`
   (or renamed). `BROADMEADOW_MAX_SLOTS` removed.
5. `ClubDayContiguousSlots` refactored to call `slot_used_indicators` + `enforce_no_gaps`
   (behaviour-identical; it's the reference impl). Its `tests/atoms/test_club_day_atoms*.py`
   stay green unchanged.
6. `ClubGameSpread` refactored: the UPPER contiguity uses the shared pattern (per Open
   decision: strict `enforce_no_gaps` OR a retained light bounded-gap); the heavy
   min/max/range/gap IntVars are removed where the shared primitive replaces them. The LOWER
   no-double-up bound is **extracted** to a concurrency atom (per Open decision) — not in the
   spread rule. Soft spread penalty either kept (lightened) or dropped per Open decision.
7. Staging fixed: `VenueEarliestSlotFill` runs in a **hard** stage (so earliest-packing is
   actually enforced — the whole point). The `ClubGameSpread` hard portion likewise runs in a
   hard stage if kept hard (no longer dead in `soft_only`). `validate_solver_stages == []`.
8. Behaviour checks (GWT, no mocks, hand oracle):
   - venue with 2 games / 2 fields offered slots {1,2,3}: a solution placing a game in slot 3
     while slot 1 empty is **INFEASIBLE**; placing in slot 1 is FEASIBLE (earliest-pack).
   - venue with games in slots {1,3} but not 2 → **INFEASIBLE** (no gap).
   - club with games in slots {1,2} feasible; {1,4} infeasible under the club rule (per chosen
     gap semantics, with the hand-computed threshold).
9. Variable-count check: on the 2026 fixture, the rewritten atoms add **strictly fewer**
   CP-SAT IntVars than the old engine methods (assert before/after `model.Proto().variables`
   for the venue + club groupings) — the efficiency win is measured, not assumed.
10. Full suite green; inventory + severity + stages docs updated; grep-clean for
    `_best_timeslot_choices`, `BROADMEADOW_MAX_SLOTS`, `BestTimeslotWF`, `AddDivisionEquality`
    (in this area).
11. **Production-wiring proof (not just unit fixtures).** An integration test loads the real
    2026 config, builds the model via the actual `DEFAULT_STAGES` dispatch, and asserts each
    rewritten rule **actually emits hard constraints in production** — i.e. it is reachable
    through `apply_stage_1_hard()`, not stranded in a `soft_only` stage. Concretely: assert the
    `VenueEarliestSlotFill` (and the kept-hard club rules) appear in a non-`soft_only` stage in
    `DEFAULT_STAGES`, and that a full-config build adds > 0 of their hard constraints. This is
    the explicit guard against the trap that currently leaves `EnsureBestTimeslotChoices` /
    `ClubGameSpread` hard parts dead.
12. **Systemic guard.** Add a test that fails if ANY constraint with a hard dispatch path is
    only reachable from `soft_only` stages in `DEFAULT_STAGES` (walk each engine hard-key /
    hard atom; verify at least one of its stages is not `soft_only`). This catches the
    soft_only trap for the whole family, not just these three, so future re-staging can't
    silently disable a hard rule. (If it surfaces others beyond spacing/spec-017, list them in
    the report and spawn a follow-up plan — do not widen this spec.)

## Open decisions (recommendations baked in)

- **A. Club gap: strict no-gap vs bounded (≤2)?** `ClubGameSpread` currently allows up to
  `max_gap=2` holes; strict `enforce_no_gaps` allows 0. A bounded-gap ≥1 genuinely needs a
  hole *count* (some arithmetic), so it can't use the pure cheap chain.
  **Recommendation:** make the club HARD rule **strict no-gap** (`enforce_no_gaps`, shared,
  cheap) and keep a **soft** penalty for any residual spread as the tuning lever — i.e. hard
  "no holes," soft "prefer tight." This drops the IntVars and matches `ClubDayContiguousSlots`.
  If the convenor truly needs "up to 2 holes allowed as hard," we keep a minimal hole-count
  (still dropping min/max via the indicators) — more code, so only if required.
- **B. Lower no-double-up: where does it go?** Recommendation: extract to a small concurrency
  atom **`ClubNoConcurrentSlot`** (per (club, week, day_slot): at most one of the club's games,
  across all its grades/teams) and put it in a **hard** stage. It complements
  `SameGradeSameClubNoConcurrency` (same-grade) by covering the cross-grade club case. Confirm
  the convenor wants this hard (a club's two teams in different grades can't share a slot) —
  it's currently dead (soft_only) so flipping it on is a real behaviour change.
- **C. Venue fill severity/level.** Recommendation: HARD (the earliest-packing guarantee is
  what lets us drop the 7pm rule). If the convenor prefers it soft, the 7pm-avoidance argument
  weakens and we'd revisit a soft late-slot penalty — so default HARD.

## Implementation units

> Sequence: A → (B, D in parallel-ish but both touch unified.py/registry, so serialize) → C.
> One worktree, commit per unit; rebase on spec-016/017/018 before merge.

### Unit A — Shared `_contiguity` helper + tests
- Files: `constraints/atoms/_contiguity.py` (new), `tests/atoms/test_contiguity_primitive.py`.
- No production wiring yet. Test each function on tiny CP-SAT fixtures with hand oracles.

### Unit B — `VenueEarliestSlotFill` atom (replace EnsureBestTimeslotChoices)
- Files: `constraints/atoms/venue_earliest_slot_fill.py` (new), `constraints/atoms/__init__.py`,
  `constraints/registry.py` (replace `EnsureBestTimeslotChoices` entry), `constraints/severity.py`,
  `constraints/unified.py` (remove `_best_timeslot_choices_hard/_soft`, dispatch, skip-keys,
  `BROADMEADOW_MAX_SLOTS`), `config/defaults.py` (DEFAULT_STAGES: move to a HARD stage; remove
  `BestTimeslotWF` weight; the `worst_timeslot_time` default can be removed), `analytics/tester.py`
  (replace the best-timeslot check with an earliest-fill check).
- Depends on Unit A. Test per DoD 2, 3, 8 (venue cases), 9.

### Unit C — `ClubGameSpread` refactor + extract `ClubNoConcurrentSlot`
- Files: `constraints/unified.py` (`_club_game_spread_hard/_soft`), `constraints/atoms/club_no_concurrent_slot.py`
  (new, per Open decision B), `constraints/registry.py`, `constraints/severity.py`,
  `config/defaults.py` (stages + `club_game_spread_*` defaults), `analytics/tester.py`.
- Depends on Unit A. Test per DoD 6, 8 (club cases), 9; concurrency test for the extracted atom.

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
- `docs/PERENNIAL_RULES.md` + `CLAUDE.md` Draw-Review-Checklist — the "last game on WF" and
  "avoid 7pm" review notes: WF is owned by `NIHCFillWFBeforeEF`; 7pm avoidance is now
  structural via earliest-fill. Update the checklist wording.
- `docs/DRAW_RULES.md` — describe earliest-slot packing (venue) vs game clustering (club).
- `docs/todo/GOALS.md` — add spec-021 row; record the "share the pattern, keep atoms separate"
  decision as a worked example of the §2 "extract a helper, don't merge" rule.

## Out of scope

- Re-adding any 7pm penalty (earliest-fill makes it moot — confirmed with the convenor).
- The per-field cross-field "balance fields within a slot" nicety (owned by
  `MaximiseClubsPerTimeslotBroadmeadow` / `MinimiseClubsOnAFieldBroadmeadow`) — untouched.
- A general "bounded gap ≤ N as a cheap primitive" — only built if Open decision A requires it.
- Atomising the *other* monolithic engine methods (`_equal_games_balanced_matchups`, etc.) —
  no bloat benefit (shared `_cache`), separate concern.
- The systemic "hard parts parked in soft_only" audit beyond these two atoms — if more are
  found, spawn a dedicated audit plan rather than widening this one.
