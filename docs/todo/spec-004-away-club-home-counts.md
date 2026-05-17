<!-- status: ready -->
<!-- owner: unassigned -->
<!-- depends_on: none -->

# spec-004 — Away-club home/away atoms aligned (FORCED-Friday aware)

**Spec source:** [`docs/todo/GOALS.md` → spec-004](GOALS.md#spec-004--away-club-homeaway-atoms-aligned-with-forced-friday-math)

> **Coordination:** Shares helper `_phl_forced_friday_helper.py` with spec-005. **Either spec-004 or spec-005 implements the helper; the other depends on it.** Convention: whichever plan is picked up first implements the helper, flips its own status to in_progress, and updates the OTHER plan's `depends_on` line to point to its commit hash. Re-read the sibling plan before starting.

## Why

Today, `NonDefaultHomeGrouping` and `FiftyFiftyHomeandAway` together approximate per-club home/away balance, but they don't account for FORCED PHL Friday games. Naive math counts PHL Fridays as Sunday weekend slots, leading to over-counted home weekends and sparse Sunday draws. The convenor manually fudges. Codifying as registry atoms with shared adjuster math fixes it once for all away clubs (current: Maitland, Gosford; future: any).

## Definition of Done

1. New helper module `constraints/atoms/_phl_forced_friday_helper.py` exporting `phl_forced_friday_count(data, club)` and `away_club_required_sundays(data, club)` — both FORCED-aware, both handle the duplicate-counting case (one variable matching multiple FORCED scopes counts as ONE Friday game).
2. New atom `AwayClubHomeWeekendsCount` — forces `sum(home_weekend_indicator) == away_club_required_sundays(data, club) + forced_friday_count` for each away-based club. A Friday game counts as that weekend.
3. New atom `AwayClubPerOpponentAndAggregateHomeBalance` — for each team in each away-based club, for each opponent in its grade: `home_games_against_opponent` is an IntVar bounded `[floor(total/2), ceil(total/2)]`; AND aggregate `home_games_total` is an IntVar bounded `[floor(total_games/2), ceil(total_games/2)]`. Both bounds applied as `model.Add(...)`.
4. Both atoms registered with severity 1 (CRITICAL).
5. The old `FiftyFiftyHomeandAway` legacy class is marked obsolete in the registry — kept callable for parity but not in any `DEFAULT_STAGES` entry once these atoms ship.
6. `NonDefaultHomeGrouping` is reviewed: if its semantics overlap with `AwayClubHomeWeekendsCount`, consolidate; if it covers a distinct dimension (consecutive-home spacing), keep but document the split.
7. Unit tests with FORCED entries:
   - Given Maitland with 2 FORCED Friday games + 8 PHL required + 6 max other grade, When solving, Then Maitland home weekends == 8 (the 2 Fridays + 6 Sundays covering other grades; PHL satisfied by 2F + 6S).
   - Given one FORCED entry of `count==2 sum of Maitland Fridays` AND another `count==1 Maitland-vs-Tigers Friday`, When counting, Then `phl_forced_friday_count('Maitland') == 2` (not 3 — the per-pair entry is one of the two summed).
   - Given each pair plays 3 times, When solving, Then each pair lands 2H/1A or 1H/2A (within ±1) — no pair stuck at 3H/0A.
   - Given total team games=18, When solving, Then aggregate home games for that team ∈ {9}.
8. `docs/system/CONSTRAINT_INVENTORY.md` gets two new rows + obsoletes the `FiftyFiftyHomeandAway` row appropriately.
9. `docs/system/COUNT_ADJUSTERS.md` documents the FORCED-Friday adjuster formula.
10. `docs/operator-human/RULES.md` describes the home/away expectations in plain English.

## Implementation units

### Unit 1 — Helper module (foundational)

- **Files touched:** `constraints/atoms/_phl_forced_friday_helper.py` (new), `tests/atoms/test_phl_forced_friday_helper.py` (new).
- Pure function, no model side-effects. Hand-computed oracles in tests.
- **Critical:** must handle multi-scope FORCED entries without double counting. The correct way is to count *variables* (X keys) that satisfy ANY of the relevant FORCED scopes for the club's Fridays, not to sum FORCED entry counts.

### Unit 2 — AwayClubHomeWeekendsCount atom

- **Files touched:** `constraints/atoms/away_club_home_weekends_count.py` (new), `constraints/registry.py`, `config/defaults.py::DEFAULT_STAGES`.
- **Depends on:** Unit 1.

### Unit 3 — AwayClubPerOpponentAndAggregateHomeBalance atom

- **Files touched:** `constraints/atoms/away_club_home_balance.py` (new), `constraints/registry.py`, `config/defaults.py::DEFAULT_STAGES`.
- **Depends on:** Unit 1.
- Mark `FiftyFiftyHomeandAway` legacy entry as obsolete.

### Unit 4 — Docs

- **Files touched:** `docs/system/CONSTRAINT_INVENTORY.md`, `docs/system/COUNT_ADJUSTERS.md`, `docs/operator-human/RULES.md`.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — 2 new rows, 1 obsolescence note
- `docs/system/COUNT_ADJUSTERS.md` — FORCED-Friday formula section
- `docs/operator-human/RULES.md` — home/away plain English
- `docs/todo/GOALS.md` — flip spec-004 status to "done"

## Out of scope

- Generalising to clubs other than Maitland / Gosford — atoms iterate `home_field_map`, so future away clubs work automatically; no further code needed.
- Reworking `NonDefaultHomeGrouping` consecutive-home-week logic — separate plan if overlap turns out to be problematic.
