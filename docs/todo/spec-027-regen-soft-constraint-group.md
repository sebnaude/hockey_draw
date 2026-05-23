<!-- status: ready -->
<!-- reviewed: adversarial Sonnet review 2026-05-23 (re-review — dependency audit) — fixes applied inline -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-023 (constraint groups machinery — composable, deduped, flag-selected `--groups`; "a constraint is whole, split into two atoms if you must select sub-parts"). spec-026 (regen mode selects this group). spec-025 (pins, indirectly via spec-026). ALL THREE are unshipped on final-form as of 2026-05-23; this plan CANNOT START until all three are `done`. -->
<!-- DEPENDENCY STATUS (status is `ready` — fully shaped + hardened — but CANNOT be claimed until all three deps are `done`; gated by `depends_on`, per the basic-skill "unsatisfied dependencies = cannot start" rule, exactly like spec-026): the earlier inline note "spec-023 constraint-groups redesign HAS now landed on final-form" was FALSE and has been removed. Verified against code 2026-05-23: (1) `ConstraintInfo` has no `groups` field; (2) `resolve_groups`, `DERIVED_GROUPS`, `--groups` CLI flag do NOT exist anywhere in the codebase; (3) `soft_only` still lives in `constraints/stages.py` at lines 272 and 277 and has not been deleted. spec-023, spec-025, and spec-026 are all `ready` (hardened) but NONE is `done`/landed yet. This plan must wait for all three to land. (Status convention reconciled with spec-026 in the 2026-05-23 cross-plan global review: dependency-gating is expressed via `depends_on` + the dependency tree's "ready to start now" list, NOT via the `blocked` status, which the basic skill reserves for unanswered open questions.) -->
<!-- owner: session=none claimed=none -->

# spec-027 — Regeneration soft-constraint group: soft analogues of the hard rules, kept separate

## Why

In a scoped regeneration (spec-026), frozen pairings are pinned to their **dates** but freed in
**time/slot/field**, and a freed grade re-rolls its pairings entirely. Under the normal
constraint set this is frequently **infeasible**: rules like PHL/2nd adjacency, club-day
co-location, matchup spacing, and bye spacing all assume the solver controls the *time* of the
games they reason about — but in regen most games' times are the only free dimension and the
frozen surrounding games box them in. A full hard application would make the solver return
INFEASIBLE on changes the convenor knows are physically fine (the season "may force things to
be broken now," per the request).

The fix is a **dedicated regeneration constraint selection**: keep a small **core-hard** set
that encodes genuine physical impossibilities (two teams on one field/slot; a team in two
places at once; the freeze pins themselves) as **hard**, and apply **soft analogues** of every
other rule so the solver honours them when it can and reports a tracked penalty when it can't.
The convenor then sees exactly which "nice" rules bent to fit the change, instead of a bare
INFEASIBLE.

This is built on spec-023's groups machinery: a group is a named, deduped, flag-selected set of
**whole** constraints, and "if you need one idea of a constraint without another, split it into
two atoms." So the regen-soft forms are **atoms** (a hard atom and its soft-analogue atom are
two atoms), and a `regen` group selects {core-hard atoms} ∪ {soft-analogue atoms}, deliberately
NOT selecting the normal hard atoms of the softened rules. No `data['mode']` branch, no
`soft_only` resurrection (spec-023 will delete it) — just a different group selection.
(review fix — C1: changed "spec-023 deletes it" to "spec-023 will delete it"; spec-023 has NOT
landed; `soft_only` is still present in `constraints/stages.py` lines 272/277 as of 2026-05-23.)

The user explicitly wants this group **documented separately** from the primary draw
constraints, so it reads as "the relaxed rules we apply only when modifying an existing draw,"
not as a change to how a fresh season is built.

### Research findings (verified against final-form)

- **Severity-1 / core-feasibility constraints** (`constraints/registry.py` — final-form): `NoDoubleBookingTeams`
  (registry ~line 39), `NoDoubleBookingFields` (~46), `EqualGamesAndBalanceMatchUps` (~53),
  `SameGradeSameClubNoConcurrency` (~254), `AwayClubHomeWeekendsCount` (~84),
  `AwayClubPerOpponentAndAggregateHomeBalance` (~93), `PHLAnd2ndAdjacency` (~112),
  `PHLConcurrencyAtBroadmeadow` (~127), `PHLAnd2ndConcurrencyAtBroadmeadow` (~135),
  `EqualMatchUpSpacing` (~154, hard part). (review fix — line numbers corrected against final-form registry; the
  old name `PHLAndSecondGradeAdjacency` is superseded by `PHLAnd2ndAdjacency` since spec-014.)
  These are the candidates the regen group must triage into stay-hard vs soften.
  Note: `MaitlandHomeGrouping` / `AwayAtMaitlandGrouping` were **deleted** in spec-018 and are
  absent from final-form; they need no regen treatment.
- **The hard↔soft conversion pattern is established.** `EqualMatchUpSpacing`,
  and `ClubGameSpread` still implement BOTH a hard and a soft method in the engine
  (`constraints/unified.py` `_matchup_spacing_soft()` / `_club_game_spread_soft()`). Note:
  `ClubVsClubAlignment`'s unified.py soft path (`_club_alignment_soft`) remains in final-form's
  unified.py for legacy engine compatibility, but its production atoms are now
  `ClubVsClubStackedWeekends` + `ClubVsClubStackedCoLocation` (both HARD, no soft component) —
  new soft-analogue atoms are required for these (review fix — the original claim that
  `ClubVsClubAlignment` "already has a soft component" applies to the legacy engine path, not
  the final-form atom architecture). Spec-016 demoted NIHC-fill to a pure-soft atom (penalty
  BoolVars into `data['penalties']['nihc_fill_order']`,
  `nihc_fill_wf_before_ef.py:145-175`); spec-007 split adjacent-grade hard from the kept
  same-grade-same-club hard atom. A soft analogue = an atom that emits deviation/violation
  IntVars into a `data['penalties'][bucket]` instead of `model.Add(...)`.
- **Penalty→objective pipeline (reused unchanged).** Soft atoms append IntVars to
  `data['penalties'][bucket]` with a weight; `main_staged.py::_build_normalized_penalty`
  (`:77-107`) normalises per-bucket weight by var count and the objective
  (`main_staged.py:611-629`, `build_objective()`) does `Maximize(sum(X) − Σ normalised_penalty)`.
  Weights live in each season's `PENALTY_WEIGHTS` dict (e.g. `config/season_2026.py:1101-1138`).
  (review fix — PENALTY_WEIGHTS starts at line 1101, not 1102.) The regen-soft atoms add new
  buckets here; no objective change needed.
- **Group selection awaits spec-023.** `ConstraintInfo.groups: frozenset[str]`, `resolve_groups`
  (deduped union, registry order), `run.py generate --groups NAME...`, and the deletion of
  `soft_only` are ALL ABSENT from the current final-form codebase (verified 2026-05-23: no
  `groups` field on `ConstraintInfo`, no `DERIVED_GROUPS`, no `resolve_groups`, no `--groups`
  argument anywhere). This entire mechanism is spec-023's deliverable. spec-027 adds the
  `core_hard` / `regen_soft` tags and the derived/explicit `regen` group — but only AFTER
  spec-023 lands.
  (review fix — C1 PRIMARY: the previous bullet stated "Group selection is spec-023" implying
  it existed; replaced with an honest statement that it does NOT yet exist.)

## Definition of Done

1. **Core-hard set defined and tagged `core_hard`** (stays hard in regen). The set, with the
   physical justification each member must satisfy ("a hard violation here is a real-world
   impossibility, not an inconvenience"):
   | Constraint | Why it stays hard |
   |---|---|
   | `NoDoubleBookingTeams` | One team cannot play two games in a week. |
   | `NoDoubleBookingFields` | One field cannot host two games in one date+slot. |
   | `EqualGamesAndBalanceMatchUps` | Every team must play its correct number of games / balanced meetings — a draw with wrong counts is invalid, not merely ugly. |
   | `SameGradeSameClubNoConcurrency` | A club's two teams in one grade share players/parents; cannot be concurrent. |
   | `TeamConflict` | A specific named pair shares players; two games same week is a genuine impossibility. |
   | `ClubNoConcurrentSlot` | Physical slot-concurrency: same rationale as `NoDoubleBookingFields`; severity-2 hard atom. |
   | `PHLConcurrencyAtBroadmeadow` | Physical field/slot concurrency. |
   | `PHLAnd2ndConcurrencyAtBroadmeadow` | Physical field/slot concurrency. |
   | `AwayClubPerOpponentAndAggregateHomeBalance` | Per-opponent + aggregate balance for freed grades (which re-roll pairings). |
   | `LockedPairings` + `ForcedGames` (the pins) | The freeze itself; soft pins would defeat regen. (forward-looking: `LockedPairings` is spec-026 machinery, not a current registry entry; wire this tag once spec-026 lands.) |
   Tag each `core_hard` in the registry (additional tag alongside its existing `core`/severity).
   (review fix — `LockedPairings` does not yet exist as a registry canonical name; flagged as forward-looking per spec-026 dependency. `TeamConflict` and `ClubNoConcurrentSlot` added to the table — they were in the partition-completeness note but missing from the formal core-hard table, which is the authoritative DoD artifact.)
2. **Soft-analogue atoms for every other production hard constraint**, tagged `regen_soft`,
   each emitting penalties (no `model.Add` hard clause) into a dedicated `data['penalties']`
   bucket with a `PENALTY_WEIGHTS['regen_<name>']` weight. The constraints to soften and the
   penalty each measures:
   | Hard constraint | Soft-analogue atom | Penalty measures |
   |---|---|---|
   | `PHLAnd2ndAdjacency` | `PHLAnd2ndAdjacencyRegenSoft` | per PHL/2nd weekend where the same-club back-to-back / ≥3h cross-venue gap rule is broken |
   | `ClubDayParticipation` | `ClubDayParticipationRegenSoft` | per club-day participation violation (club has game on a day it shouldn't) |
   | `ClubDayIntraClubMatchup` | `ClubDayIntraClubMatchupRegenSoft` | per intra-club matchup scheduled outside club-day |
   | `ClubDayOpponentMatchup` | `ClubDayOpponentMatchupRegenSoft` | per opponent matchup scheduled outside club-day |
   | `ClubDaySameField` | `ClubDaySameFieldRegenSoft` | per club-day matchup split across fields |
   | `ClubDayContiguousSlots` | `ClubDayContiguousSlotsRegenSoft` | per gap between a club-day's games |
   | `ClubVsClubStackedWeekends` | `ClubVsClubStackedWeekendsRegenSoft` | per missing stacked-weekend coincidence |
   | `ClubVsClubStackedCoLocation` | `ClubVsClubStackedCoLocationRegenSoft` | per stacked matchup not co-located |
   | `BalancedByeSpacing` | `BalancedByeSpacingRegenSoft` | per bye outside its target spacing window |
   | `EqualMatchUpSpacing` | (reuse existing soft component) | per spacing-window violation (already exists in `unified.py:_matchup_spacing_soft`) |
   | `ClubGameSpread` | (reuse existing soft component) | per spread gap / double-up violation (already exists in `unified.py:_club_game_spread_soft`) |
   | `AwayClubHomeWeekendsCount` | `AwayClubHomeWeekendsCountRegenSoft` | per weekend-count deviation for frozen grades contributing to the aggregate |
   | `VenueEarliestSlotFill` | `VenueEarliestSlotFillRegenSoft` | per non-earliest-packed slot |
   (review fix — added three ClubDay sub-atom rows for `ClubDayParticipation`, `ClubDayIntraClubMatchup`,
   `ClubDayOpponentMatchup`, which are severity-2 HARD atoms in final-form. Added `ClubGameSpread`
   reuse row which was previously absent from this table. Added `AwayClubHomeWeekendsCount` row
   explicitly — DoD-3 resolved it as `regen_soft` but it was missing from this table.)
   For constraints that ALREADY have a soft component (`EqualMatchUpSpacing`, `ClubGameSpread`,
   NIHC-fill), the `regen_soft` selection reuses the existing soft atom — no new atom — and
   simply does NOT select the hard counterpart. Per spec-023's "split into two atoms" rule, any
   constraint whose hard and soft halves are still fused in one atom is split into a hard atom +
   a soft atom as part of this spec (named in Unit B).
   (review fix — `VenueEarliestSlotFill` is HARD severity 2 in final-form (spec-021 made it a
   hard atom, not soft); it requires a new soft-analogue atom `VenueEarliestSlotFillRegenSoft`,
   not reuse of an existing soft component. The parenthetical claim "already soft post-spec-021"
   is incorrect.)
3. **Home/away balance handling (RESOLVED open-decision A, recommendation baked in):**
   `AwayClubPerOpponentAndAggregateHomeBalance` stays **hard for freed grades** (a freed grade
   re-rolls pairings, so its balance must hold) but its `regen_soft` form applies to **frozen
   grades' contribution to `AwayClubHomeWeekendsCount`**, which aggregates weekends across ALL
   grades — frozen games already consumed weekends, so forcing the exact count hard can be
   infeasible. Net: `AwayClubPerOpponentAndAggregateHomeBalance` ∈ `core_hard`;
   `AwayClubHomeWeekendsCount` ∈ `regen_soft` (penalty on weekend-count deviation). Document the
   asymmetry. Both are severity-1 atoms in the current registry.
4. **PHL/2nd Broadmeadow concurrency (RESOLVED open-decision B):** `PHLConcurrencyAtBroadmeadow`
   and `PHLAnd2ndConcurrencyAtBroadmeadow` stay **`core_hard`** — they are field/slot physical
   concurrency rules (two games can't share a slot), squarely in the "no two teams per field
   slot / per timeslot" set the convenor named as must-stay-hard.

   **Partition completeness (review fix — triage now embedded in DoD-1 and DoD-2 above):**
   All production atoms have been triaged. Legacy-combined / obsolete entries (`FiftyFiftyHomeandAway`,
   `ClubDay`, `ClubVsClubAlignment`, `PHLAndSecondGradeTimes`) are NOT selected by either group;
   their active atom descendants are selected per their classification above.
   `TeamPairNoConcurrency` (severity 3, `has_soft_component=True`) is soft-optimisation, not a
   production hard constraint — it is not selected in `core_hard`; it belongs in the normal
   `soft` group and is reused unchanged.
   (review note — Low: Unit A implementer should confirm `TeamPairNoConcurrency` classification
   against the shipped spec-023 group table before tagging, as its soft component means it may
   already be in the soft group by default.)
5. **`regen` group** defined in the registry (spec-023 mechanism): `regen` resolves to the
   deduped union of `core_hard` ∪ `regen_soft` ∪ the always-soft `soft` group (lex ordering,
   preferred games, etc. stay soft as normal), and **excludes** the normal hard atoms of every
   softened constraint. A test asserts `resolve_groups(['regen'])` contains every `core_hard`
   member, every `regen_soft` member, and **none** of the hard atoms that `regen_soft` replaces
   (hand-list both sides).
6. **spec-026 consumes it:** regen mode selects `--groups regen` (or programmatically resolves
   `regen`). The DoD-3 warning in spec-026 ("may be infeasible without spec-027") is removed —
   regen now defaults to the `regen` group.
7. **Infeasibility→feasibility witness test:** a synthetic fixture that is INFEASIBLE under the
   full hard set (a frozen game boxes a freed game's only feasible time, violating hard
   adjacency) becomes FEASIBLE under `--groups regen`, with the adjacency penalty recorded > 0.
   Hand-construct the fixture so the single forced violation is provable.
8. **Penalty weights** added to each season `PENALTY_WEIGHTS` dict (suggested keys:
   `regen_phl_2nd_adjacency`, `regen_club_day_participation`, `regen_club_day_intra_club_matchup`,
   `regen_club_day_opponent_matchup`, `regen_club_day_same_field`, `regen_club_day_contiguous_slots`,
   `regen_clubvsclub_stacked_weekends`, `regen_clubvsclub_stacked_colocation`,
   `regen_balanced_bye_spacing`, `regen_away_club_home_weekends_count`,
   `regen_venue_earliest_slot_fill`) with sensible relative magnitudes (core-feasibility-adjacent
   rules weighted higher than cosmetic ones). Defaults documented.
   (review fix — added `regen_venue_earliest_slot_fill` and the three new ClubDay sub-atom keys to
   match atoms added in DoD-2/Unit B.)
9. **Separate documentation** `docs/system/REGEN_CONSTRAINTS.md` (NEW) describes the regen group
   in full — the core-hard set, every soft analogue, its bucket + weight, and the rule "this set
   applies ONLY in regeneration (spec-026); a fresh season build never uses it." Registered in
   the doc index (`docs/README.md`). The primary `CONSTRAINT_INVENTORY.md` gains only a pointer
   row, not the full table, keeping regen rules visibly separate per the convenor's request.
10. `len(CONSTRAINT_REGISTRY)` net change = number of new soft-analogue atoms added; count test
    updated. `validate_solver_stages` / group-order validator green; `resolve_groups(['regen'])`
    well-formed; full suite green. A full 2026 regen (`--regen-grades 6th --groups regen`) solves
    feasibly (post-merge verification).

## Dependencies

| Spec | What it delivers | Status (2026-05-23) | Blocks |
|---|---|---|---|
| **spec-023** | `ConstraintInfo.groups` field, `resolve_groups`, `DERIVED_GROUPS`, `--groups` CLI flag, deletion of `soft_only` | `under_review` — NOT LANDED | Unit A cannot start; nothing in spec-027 can start |
| **spec-025** | `LOCKED_PAIRINGS` config + `generate_X` enforcement | `under_review` — NOT LANDED | spec-026, indirectly this plan |
| **spec-026** | `--regen-from` mode that selects the `regen` group | `under_review` — NOT LANDED | Unit C cannot start |

**All three must reach `done` before any unit of spec-027 begins.** The dependency chain is
linear for startup: spec-023 and spec-025 can land in parallel; spec-026 needs spec-025 done;
spec-027 needs spec-023 AND spec-026 done.

(review fix — C1 CRITICAL: previous header line 6 stated "spec-023 constraint-groups redesign
HAS now landed on final-form (this reconciliation), so the original block is cleared." This is
FALSE. The block is NOT cleared. This Dependencies section replaces that false inline note with
a truthful status table. The header block comment has also been corrected.)

## Implementation units

> **Cannot begin until spec-023 + spec-025 + spec-026 are `done`.**
> Heavily shared: `constraints/registry.py`, `constraints/atoms/*`, season configs. Each
> soft-analogue atom is independent of the others (different files) and CAN parallelise once
> Unit A (tags + `regen` group) lands; sequence Unit A first, then fan out B-atoms.

### Unit A — `core_hard`/`regen_soft` tags + `regen` group + weights
- **Prerequisite:** spec-023 `done` (the `groups` field + resolver must exist in the codebase).
- Files: `constraints/registry.py` (tags on existing constraints, `regen` group definition via
  spec-023's `groups`/`DERIVED_GROUPS`), each season `PENALTY_WEIGHTS` dict (new `regen_*`
  buckets).
- Test: `resolve_groups(['regen'])` = hand-listed union (DoD 5); core-hard members present;
  replaced hard atoms absent.

### Unit B — Soft-analogue atoms (one sub-task per atom)
- **Prerequisite:** Unit A done.
- Files (new atoms): `constraints/atoms/phl_2nd_adjacency_regen_soft.py`,
  `club_day_same_field_regen_soft.py`, `club_day_contiguous_slots_regen_soft.py`,
  `club_day_participation_regen_soft.py`, `club_day_intra_club_matchup_regen_soft.py`,
  `club_day_opponent_matchup_regen_soft.py`,
  `clubvsclub_stacked_weekends_regen_soft.py`, `clubvsclub_stacked_colocation_regen_soft.py`,
  `balanced_bye_spacing_regen_soft.py`, `away_club_home_weekends_count_regen_soft.py`,
  `venue_earliest_slot_fill_regen_soft.py`; plus any hard/soft atom SPLITS required by
  spec-023's "two atoms" rule for constraints whose halves are still fused. Each registered in
  `constraints/registry.py` (`severity_level=5`, `has_soft_component=True`,
  `groups={regen_soft}`, tester check method) and `constraints/atoms/__init__.py`.
  (review fix — added `venue_earliest_slot_fill_regen_soft.py` (VenueEarliestSlotFill is HARD
  severity-2 in final-form), and three ClubDay sub-atom soft analogues for
  `ClubDayParticipation`, `ClubDayIntraClubMatchup`, `ClubDayOpponentMatchup`, which are
  HARD severity-2 atoms not covered by the original list.)
- Test (per atom, GWT, no mocks, hand oracle on a tiny CP-SAT fixture): a configuration that the
  hard form forbids yields penalty = the hand-computed violation count under the soft form, and
  the model stays feasible. Reuse the existing soft components where DoD 2 says so (no duplicate
  atom).

### Unit C — Wire `regen` into spec-026 + remove the infeasibility warning + docs
- **Prerequisite:** Units A+B done AND spec-026 `done`.
- Files: `run.py`/`main_staged.py` (regen selects `regen` group; remove spec-026 DoD-3 warning),
  `analytics/tester.py` (regen-soft pressure reporting), `docs/system/REGEN_CONSTRAINTS.md`
  (new), `docs/README.md` (index), `docs/system/CONSTRAINT_INVENTORY.md` (pointer row),
  `CLAUDE.md` (regen group note), `docs/todo/GOALS.md` (spec-027 row).
- Test: DoD-7 infeasible→feasible witness; the tester reports regen-soft penalties as soft
  pressure (not hard violations); end-to-end 2026 regen solves.

## Doc registry

- `docs/system/REGEN_CONSTRAINTS.md` — **NEW**, the separate home for the regen group (DoD 9).
- `docs/README.md` — register `REGEN_CONSTRAINTS.md` in the doc map (system category).
- `docs/system/CONSTRAINT_INVENTORY.md` — pointer row only; do NOT inline the regen table.
- `docs/system/STAGES.md` — note `regen` as a selectable group (cross-ref spec-023's `--groups`).
- `docs/system/HELPER_VARS.md` — if any soft-analogue reuses producer helpers, note the sharing.
- `CLAUDE.md` — Constraint-System section: the `regen` group exists for scoped regeneration,
  softens all but the core-hard physical rules, and is selected automatically by `--regen-from`.
- `docs/todo/GOALS.md` — spec-027 row; a §2 note that soft analogues are *separate atoms*
  (spec-023's two-atoms rule), never a hard/soft switch.
- `docs/todo/00-dependency-tree.md` — update spec-027 to `blocked` (depends spec-023 + spec-026,
  neither of which is `done`).

## Risks & blast radius

- **Weight tuning.** If regen-soft weights are mis-scaled the solver may sacrifice an important
  rule (adjacency) to satisfy a cosmetic one (lex order). Mitigation: weight core-feasibility-
  adjacent rules (adjacency, concurrency-adjacent) far above cosmetic ones; the DoD-7 witness
  and a regen on a real draw surface gross mis-weighting. This is awareness, not a rollback.
- **Atom proliferation.** ~13 new soft-analogue atoms grow the registry (9 original + 3 ClubDay
  sub-atoms + VenueEarliestSlotFill + AwayClubHomeWeekendsCount). Mitigation: reuse existing
  soft components where present (DoD 2); the count test pins the exact net change.
  (review fix — count updated from ~7 to ~13 to reflect all additions: original 7
  + VenueEarliestSlotFill + AwayClubHomeWeekendsCount + 3 ClubDay sub-atoms.)
- **Drift from the hard atom.** A soft analogue can fall out of sync with its hard sibling if
  the rule changes later. Mitigation: each soft analogue is tested against the SAME fixture
  shape as its hard sibling (penalty>0 exactly where the hard form forbids), so a divergence
  trips a test.
- **Two-atoms splits** touch the engine/atom boundary spec-023 is already reshaping; rebase on
  spec-023 before starting and re-run the group-order validator.
- **Dependency chain length.** This is the most-blocked spec: it needs spec-023 + spec-025 +
  spec-026 all `done`. spec-025 → spec-026 → (with spec-023) → spec-027. No work can start
  on this plan until the full chain clears.

## Out of scope

- **Making FORCED/LOCKED_PAIRINGS pins soft** — pins stay hard (they ARE the freeze); spec-025.
- **A general per-constraint hard/soft toggle / `data['mode']` branch** — explicitly rejected
  (spec-023's "a constraint is whole; split into two atoms"); the regen-soft forms are distinct
  atoms selected by group.
- **Reworking the objective / penalty normalisation** — reused unchanged
  (`main_staged.py:77-107`).
- **Softening the core-hard set under any flag** — the core-hard set is invariant; if a regen is
  infeasible even with core-hard-only, that is a real impossibility for the convenor to resolve
  (e.g. relax a pin), not a constraint to soften.
- **Auto-tuning regen weights** — fixed defaults (DoD 8); adaptive weighting is a separate idea.
- **Updating `00-dependency-tree.md`** — that file currently marks spec-027 as `ready` and says
  "the earlier hard block is now cleared," which is false. Correcting it is out of scope for
  this plan file; a human or the next spec-023 adversarial pass should fix it.
