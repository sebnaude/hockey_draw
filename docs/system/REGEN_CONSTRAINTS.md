# Regeneration Constraint Group (`regen`)

Reference for the `regen` constraint group introduced in spec-027. This document
is the single authoritative home for everything about softening constraints during
scoped draw regeneration.

---

## What regeneration is and why softening is needed

Regeneration (`--regen-from <draw> --regen-grades ... / --regen-weeks ...`,
spec-026) freezes every game outside the named scope to its date via
`LOCKED_PAIRINGS`, then re-solves the freed scope inside those pins.

The problem: hard rules written for a fresh season build assume the solver has full
freedom over all game times. When a substantial portion of the draw is already
pinned, the solver is handed a frozen-but-retimed draw where:

- adjacency windows may already be violated by the pinned positions,
- spacing targets may be locally unsatisfiable within the freed window,
- club-day clustering may be impossible given the fixed positions around it.

Running the full `default` / `core` constraint set in this context can make the
regen model INFEASIBLE even when the intended human outcome (update a few grades,
keep everything else sensible) is perfectly achievable. The `regen` group exists
to trade hard infeasibility for soft penalties: the rules that assumed solver
freedom become penalty terms so the solver finds a best-effort feasible solution
instead of failing.

---

## Core-hard set — constraints that stay HARD in regen

These 12 entries express genuine physical impossibilities or structural invariants.
Softening them would produce an invalid draw. They are tagged `core_hard` in
`constraints/registry.py` and are always included by the `regen` group.

| Constraint | Why it stays hard |
|---|---|
| `NoDoubleBookingTeams` | A team cannot play twice in the same week — a hard physical limit. |
| `NoDoubleBookingFields` | A field cannot host two games in the same timeslot on the same date. |
| `EqualGamesAndBalanceMatchUps` | Every team must play its required number of games with correct opponent balance. The draw is invalid otherwise. |
| `AwayClubPerOpponentAndAggregateHomeBalance` | Per-pair and per-team aggregate home/away balance must hold across the full season; regen of a scope does not change the season totals requirement. |
| `PHLConcurrencyAtBroadmeadow` | Two PHL games cannot share the same timeslot at Broadmeadow — a hard field constraint. |
| `SameGradeSameClubNoConcurrency` | Two teams from the same club in the same grade cannot play at the same time (a parent can only be in one place). |
| `ClubNoConcurrentSlot` | A club's games per timeslot/venue are capped by field capacity — a physical bound. |
| `TeamConflict` | Named team-pair conflicts must never be concurrent (person-in-two-places issue). |
| `ForcedGames` | FORCED_GAMES entries are enforced at `generate_X` time (variable elimination). Cannot be softened by a constraint group. |
| `BlockedGames` | BLOCKED_GAMES entries are enforced at `generate_X` time (variable elimination). Cannot be softened. |
| `LockedPairings` | LOCKED_PAIRINGS pins are enforced at `generate_X` time. Cannot be softened — these are the freeze pins written by the regen mode itself (spec-025/026). |

> **spec-030:** `PHLAnd2ndConcurrencyAtBroadmeadow` was deleted (its same-club PHL/2nd
> same-Broadmeadow-slot rule is a strict subset of `PHLAnd2ndAdjacency`'s same-venue
> branch). In regen, that same-slot case is now governed by the **soft**
> `PHLAnd2ndAdjacencyRegenSoft` analogue (below), whose same-venue branch penalises any
> non-adjacent pairing — including same-slot — so the case still incurs a penalty rather
> than a hard clause.

---

## Regen-soft analogues — 13 new penalty atoms (spec-027)

These atoms are the soft-analogue counterparts of the hard rules that the `regen`
group does NOT select directly. Each emits a penalty into
`data['penalties'][<bucket>]` instead of a hard CP-SAT clause. Severity 5,
`has_soft_component = True`, `groups = {'regen_soft'}`.

They are NOT in `DEFAULT_STAGES` and a fresh season build (no `--regen-from` flag)
**never applies them**.

| Hard rule replaced | RegenSoft atom | Penalty bucket | Default weight | What 1 penalty unit measures |
|---|---|---|---|---|
| `PHLAnd2ndAdjacency` | `PHLAnd2ndAdjacencyRegenSoft` | `regen_phl_2nd_adjacency` | 100 000 | Per PHL/2nd same-club weekend that breaks the same-venue back-to-back or the ≥150-min cross-venue rule |
| `AwayClubHomeWeekendsCount` | `AwayClubHomeWeekendsCountRegenSoft` | `regen_away_club_home_weekends_count` | 90 000 | Per weekend the home-weekend count deviates from the target for an away-based club |
| `ClubVsClubStackedWeekends` | `ClubVsClubStackedWeekendsRegenSoft` | `regen_clubvsclub_stacked_weekends` | 80 000 | Per budget-deviation weekend + per missing stacked-weekend coincidence across grades |
| `ClubVsClubStackedCoLocation` | `ClubVsClubStackedCoLocationRegenSoft` | `regen_clubvsclub_stacked_colocation` | 70 000 | Per extra field beyond one + per internal slot gap on a stacked pair-Sunday |
| `EqualMatchUpSpacing` | `EqualMatchUpSpacingRegenSoft` | `regen_equal_matchup_spacing` | 60 000 | Per too-close repeat matchup violating the ideal spacing window |
| `BalancedByeSpacing` | `BalancedByeSpacingRegenSoft` | `regen_balanced_bye_spacing` | 50 000 | Per too-close bye pair (below the ideal bye-gap threshold) |
| `ClubDayParticipation` | `ClubDayParticipationRegenSoft` | `regen_club_day_participation` | 40 000 | Per club-day team that fails to play on its club-day |
| `ClubDayIntraClubMatchup` | `ClubDayIntraClubMatchupRegenSoft` | `regen_club_day_intra_club_matchup` | 35 000 | Per intra-club derby scheduled off the club-day |
| `ClubDayOpponentMatchup` | `ClubDayOpponentMatchupRegenSoft` | `regen_club_day_opponent_matchup` | 35 000 | Per opponent matchup scheduled off the club-day |
| `ClubDaySameField` | `ClubDaySameFieldRegenSoft` | `regen_club_day_same_field` | 30 000 | Per extra field a club-day matchup is split across (beyond one) |
| `ClubDayContiguousSlots` | `ClubDayContiguousSlotsRegenSoft` | `regen_club_day_contiguous_slots` | 25 000 | Per internal gap in a club-day's slot sequence |
| `ClubGameSpread` | `ClubGameSpreadRegenSoft` | `regen_club_game_spread` | 20 000 | Per per-field interior hole + per off-primary-field game |
| `VenueEarliestSlotFill` | `VenueEarliestSlotFillRegenSoft` | `regen_venue_earliest_slot_fill` | 10 000 | Per non-earliest-packed slot / fill hole at a venue |

Penalty weights are read from `PENALTY_WEIGHTS` in the season config (key =
penalty bucket name). Seasons without an explicit entry fall back to the default
weight listed above.

The objective function is unchanged: `Maximize(sum(X) − Σ normalised_penalty)`
via `main_staged._build_normalized_penalty + build_objective`. Weights are
normalised per-bucket by variable count.

---

## The `regen` group definition

```
regen = core_hard ∪ regen_soft ∪ soft
```

**Predicate (DERIVED_GROUPS in `constraints/registry.py`):** a constraint is in
`regen` iff its `ConstraintInfo.groups` set intersects
`{'core_hard', 'regen_soft', 'soft'}`.

This resolves to **32 constraints** currently:

- 12 `core_hard` entries (the physical-impossibility set above — stay hard).
- 13 `regen_soft` entries (the new penalty atoms — replace the softened hard rules).
- 7 `soft` entries (the normal soft-optimisation atoms — unchanged, always run in
  any sensible solve).

The hard rules that `regen` does NOT select (because the regen-soft analogue
replaces them):

- `PHLAnd2ndAdjacency`, `AwayClubHomeWeekendsCount`, `ClubVsClubStackedWeekends`,
  `ClubVsClubStackedCoLocation`, `EqualMatchUpSpacing`, `BalancedByeSpacing`,
  `ClubDayParticipation`, `ClubDayIntraClubMatchup`, `ClubDayOpponentMatchup`,
  `ClubDaySameField`, `ClubDayContiguousSlots`, `ClubGameSpread`,
  `VenueEarliestSlotFill`.

---

## How the `regen` group is selected

**Auto-selected by `--regen-from`:** `run.py::_select_regen_group()` →
`resolve_groups(['regen'])` is called whenever `--regen-from` is passed. The
caller never needs to pass `--groups regen` explicitly.

**CLI (manual):** `run.py generate --groups regen --year 2026`. Selects the
deduped union of the regen set (same as above) without requiring `--regen-from`.

**Dispatch path:** a regen run is always routed through the STAGED constraint
dispatcher (`apply_constraint_set`) with a single synthetic `'regen'` stage
equal to the resolved regen set. The `--simple` engine-only path cannot dispatch
the non-engine `RegenSoft` atoms, so **`--simple` is ignored for regen runs**.

**Normal (non-regen) runs:** the `regen` group is never selected by `default`,
`all`, `production`, `core`, `soft`, or any legacy stage name — a fresh season
build never applies any `regen_soft` atom.

---

## Engine-key design note (EqualMatchUpSpacing / ClubGameSpread)

`EqualMatchUpSpacing` and `ClubGameSpread` are ENGINE keys: the unified engine
runs their hard and soft parts together and there is no soft-only dispatch path
(spec-023 deleted `soft_only`). Reusing their engine key with a "soft-only" flag
is therefore impossible.

`EqualMatchUpSpacingRegenSoft` and `ClubGameSpreadRegenSoft` are standalone atoms
that re-implement only the penalty side of each rule. This is intentional and not
a deficiency — it follows the same principle as every other regen-soft atom in
this set (split into two atoms when you need to apply one idea without the other,
per the atomization model in `docs/todo/GOALS.md`).

---

## Invariant — only for regeneration

> **The `regen_soft` atoms are NEVER applied in a fresh season build.**

Their groups set is `{'regen_soft'}` only. They have no `core`, `club_day`,
`club_alignment`, `critical_feasibility`, or `soft_optimisation` tag. A run
without `--regen-from` (or without an explicit `--groups regen`) will never
select them, regardless of any `--slack` or `--relax` flag.

---

## See also

- `docs/system/STAGES.md` — `--groups` mechanism; how `regen` is registered as a
  `DERIVED_GROUPS` predicate.
- `docs/system/CONSTRAINT_INVENTORY.md` — per-atom engineering detail (rows for
  each `regen_soft` atom are there with forced-games / locked-week / dummy-key
  handling).
- `docs/todo/GOALS.md` — `spec-027` section for the original design rationale.
- `spec-027-regen-soft-constraint-group.md` — implementation plan (in
  `docs/todo/` or `docs/todo/done/`).
- `spec-026-regeneration-mode.md` — the `--regen-from` CLI and freeze-pin logic
  that the `regen` group was built to serve.
