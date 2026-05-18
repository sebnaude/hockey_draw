# FORCED_GAMES supersedes per-venue Friday count atoms

> **STATUS: DONE — 2026-04-28.** Migration shipped on `final-form` in
> commits `e9bf5a7` (FORCED `'club'` filter + Phase 21 subset-consistency
> validation) and `5cfae6c` (atom retirement). Three Friday-count atoms
> (`BroadmeadowFridayCount`, `GosfordFridayCount`, `MaitlandFridayCount`)
> are gone; their budgets live in `season_2026.py:FORCED_GAMES` now.
> `GosfordFridayRoundsForced` was RETAINED — its per-round
> enforcement isn't yet covered by per-round FORCED entries.
> Test bar: 1272 passed, 1 skipped (was 1268).
>
> The convention going forward is documented in:
> - `CLAUDE.md` §3-§4 (FORCED `club=` filter + per-venue counts use FORCED)
> - `docs/PERENNIAL_RULES.md` Rule 3 (count budgets pattern for new seasons)
>
> The remainder of this document is preserved as the historical
> design rationale and migration plan.

---

## End-to-end verification (spec-009 — 2026-05-18)

All three behavioural claims in this document have been verified end-to-end
by new test files on the `spec-009-forced-count-verification` branch:

| Check | Description | Test file | Verdict |
|---|---|---|---|
| 1 | FORCED total + per-pair share one variable (multi-scope); total doesn't inflate | `tests/test_forced_total_plus_per_pair.py` | **PASS** |
| 2 | PHL Friday FORCED entries exempt from `ClubVsClubCoincidence` Sunday count via Adjuster #5 | `tests/atoms/test_cvc_coincidence_phl_friday_adjuster.py` | **PASS** |
| 3 | Same-grade-same-club double-ups: concurrent play blocked; coincidence atom sees distinct matchups | `tests/atoms/test_double_up_handling.py` | **PASS** |

No bugs found. The shipped code matches the spec exactly. Test bar after adding
15 new tests: **1438 passed** (was 1423).

## TL;DR

The user's design decision: **per-venue / per-day game-count limits are NOT hardcoded atoms — they are `FORCED_GAMES` entries in the season config.** `FORCED_GAMES` already supports partial-key matching plus `constraint` + `count` overrides, so an entry like

```python
{'club': 'Maitland', 'grade': 'PHL', 'day': 'Friday', 'count': 2, 'constraint': 'equal'}
```

means "exactly 2 PHL Friday games involving any Maitland team." Combine that with

```python
{'teams': ['Norths', 'Maitland'], 'grade': 'PHL', 'day': 'Friday', 'count': 1, 'constraint': 'equal'}
```

and the multi-scope FORCED registration (commit `cd8a338`) ensures the Norths-vs-Maitland Friday game counts toward **both** rules — exactly 2 Maitland Friday games, one of which is vs Norths.

Once that pattern is verified end-to-end and pre-solver consistency checks are in place, the three Friday-count atoms (`BroadmeadowFridayCount`, `GosfordFridayCount`, `MaitlandFridayCount`) — committed as part of Phase 3a in `1956608` — are **redundant and must be removed** in favor of explicit FORCED entries in `config/season_2026.py`.

This task is to be executed **before** the rest of Phase 3 (atomizing `ClubDayConstraint` and `ClubVsClubAlignment`).

---

## Background

`utils.py::_build_forced_game_rules` (line 500) groups FORCED entries by their *scope* (the non-team fields: `grade, day, day_slot, time, week, date, round_no, field_name, field_location`). Per scope it tracks:

- a list of team matchers (`('pair', t1, t2)`, `('any', team)`, or `('all',)`);
- a constraint type (`'equal' | 'lesse' | 'greater' | 'greatere' | 'less'`, default `'equal'`);
- an integer `count` (default `1`).

The solver then enforces, per scope, `sum(matching X-vars) <op> count`. Variable elimination (any var matching the scope but no team matcher gets dropped) plus the count constraint together implement "force N games of this kind."

Bug fix `cd8a338` ensures a single var that matches multiple scopes is registered against **every** matching scope — so one Norths-vs-Maitland Friday game counts toward both the "2 Maitland Friday" scope and the "1 Norths-vs-Maitland Friday" scope simultaneously.

### Today's state

- **Scope handling** is in `utils.py:540-568` — works for any subset of `_SCOPE_FIELDS`.
- **Constraint type + count** parsed at `utils.py:571-577`.
- **Multi-scope match** in `_get_matching_forced_scopes` (line 629) — returns ALL matching scopes, not just first (the `cd8a338` fix).
- **'teams' / 'team1' / 'team2' team filters** are supported (lines 581-609).
- **`'club'` team filter is NOT supported in FORCED_GAMES** — only in BLOCKED_GAMES (line 771-774). **This is the gap to close first.**
- **Pre-solver validation** of forced-rule consistency exists partially (`utils.py:1700-1800` checks for double-booking on `equal` entries) but does not yet validate that overlapping count-rules are mathematically consistent (e.g. forcing `equal 2` for one scope and `equal 0` for an overlapping scope is contradictory).

---

## Why remove the Friday-count atoms

The atoms `BroadmeadowFridayCount`, `GosfordFridayCount`, `MaitlandFridayCount` (committed in `1956608`) hardcode constraints of the shape "sum(PHL vars matching Friday at venue X) <op> N" using config keys in `CONSTRAINT_DEFAULTS`. That is *exactly* what a single `FORCED_GAMES` entry expresses, and FORCED entries are:

1. **Discoverable** — they appear in season config alongside the other forced/blocked rules, in the same format reviewers already audit.
2. **Reportable** — they show up in `forced_game_outcomes` in draw metadata and in the pre-season report.
3. **Composable** — they combine with finer-grained FORCED rules (e.g. "1 of those games is Norths-vs-Maitland") via the multi-scope match.
4. **Validatable** — pre-solver checks can detect contradictions between overlapping rules.
5. **Per-season** — different seasons can change the count without code edits.

Hardcoded count atoms are none of those things. Keeping both creates two parallel mechanisms with the potential to disagree.

---

## What you need to do

### 1. Add `'club'` team-filter support to `_build_forced_game_rules`

In `utils.py:581-612`, after the `'teams'` / `'team1'+team2'` branches, add a `'club'` branch that mirrors BLOCKED_GAMES handling at line 771-774:

```python
elif 'club' in entry:
    club_val = entry['club']
    resolved = _resolve_team_name(club_val, effective_grade, team_names_set, team_lookup, teams)
    for rt in resolved:
        scope_groups[scope_key].append(('any', rt))
```

`_resolve_team_name` already handles club-name → team-name expansion (it returns all PHL teams of club X if `effective_grade='PHL'`).

### 2. Pre-solver consistency validation

Extend `validate_game_config` (in `utils.py`) with a phase that, for each pair of overlapping FORCED scopes, asserts the per-scope count constraints are **mathematically satisfiable together**.

The check is essentially: if scope A is `{grade='PHL', day='Friday'}` with `count=2 equal` and scope B is `{grade='PHL', day='Friday', club='Maitland'}` with `count=2 equal`, then B's vars are a subset of A's vars, so `count_B <= count_A` must hold (else infeasible). Generalizes to:

- **Subset relation**: if scope B's fields are a strict superset of scope A's fields (B is more specific), B's matching vars are a subset of A's. Apply `count_B <op_B> threshold` against `count_A <op_A> threshold` for compatibility.
- **Disjoint scopes**: no validation needed — independent.
- **Overlapping but neither subset**: harder; for first cut, warn (don't fail) and let the solver detect infeasibility.

Output should be a phase report in the same shape as the existing `validate_game_config` phases.

### 3. Add tests proving the multi-scope FORCED works for the Maitland Friday case

Create `tests/test_forced_games_multi_scope.py` (or extend an existing test file). Build a small fixture (4-team PHL grade, 2 weeks with Friday slots at Maitland Park), populate `FORCED_GAMES` with:

```python
[
    {'club': 'Maitland', 'grade': 'PHL', 'day': 'Friday', 'field_location': 'Maitland Park',
     'count': 2, 'constraint': 'equal'},
    {'teams': ['Norths', 'Maitland'], 'grade': 'PHL', 'day': 'Friday', 'field_location': 'Maitland Park',
     'count': 1, 'constraint': 'equal'},
]
```

Solve and assert:
- Exactly 2 Friday games at Maitland Park exist.
- Exactly 1 of them is Norths-vs-Maitland.
- The other is a different Maitland-involving pair.
- No other Friday games at Maitland Park.

Use the real solver path through `generate_X` + a CP-SAT model — no mocks. (Pattern: `tests/atoms/conftest.py:build_model_X` shows how to build a small real fixture.)

### 4. Migrate `config/season_2026.py` to FORCED entries

Add to `FORCED_GAMES`:

```python
# === Friday-night per-venue PHL counts (replaces hardcoded atoms) ===
{'grade': 'PHL', 'day': 'Friday', 'field_location': 'Newcastle International Hockey Centre',
 'count': 3, 'constraint': 'lesse',
 'description': 'Max 3 PHL Friday games at Broadmeadow per season'},
{'grade': 'PHL', 'day': 'Friday', 'field_location': 'Central Coast Hockey Park',
 'count': 8, 'constraint': 'equal',
 'description': 'Exactly 8 PHL Friday games at Gosford per season (AGM 2026)'},
{'grade': 'PHL', 'day': 'Friday', 'field_location': 'Maitland Park',
 'count': 2, 'constraint': 'equal',
 'description': 'Exactly 2 PHL Friday games at Maitland Park per season (Gosford-vs-Maitland only)'},
```

The Gosford-Friday-rounds-{2,4,5,9,10} rule from `GosfordFridayRoundsForced` already maps to FORCED entries in season_2026.py (lines 555-603 — `Gosford Friday Night - Apr 17` etc., one per required round). Verify each required round has exactly one FORCED entry; if so, the `GosfordFridayRoundsForced` atom is also redundant and should be removed.

### 5. Remove the redundant atoms

Once steps 1-4 are green, delete:

- `constraints/atoms/broadmeadow_friday_count.py`
- `constraints/atoms/gosford_friday_count.py`
- `constraints/atoms/gosford_friday_rounds.py` (only if its function is fully covered by individual per-round FORCED entries — verify first)
- `constraints/atoms/maitland_friday_count.py`

Update:
- `constraints/atoms/__init__.py` — remove imports + drop from `PHL_TIMES_ATOMS`
- `constraints/unified.py:_PHL_HARD_ATOMS` — remove the deleted atoms
- `constraints/registry.py` — remove their `ConstraintInfo` entries
- `tests/atoms/test_phl_atoms.py` — remove their `Test*` classes
- `tests/atoms/test_phl_atoms_parity.py` — drop the MaitlandFridayCount adjustment (the parity should now be **exact**: atom_count == legacy_count, because both atoms and legacy now use FORCED for Friday counts)
- `tests/test_constraint_registry.py` — bump entry-count assertion down (29 minus number of removed atoms)
- `config/defaults.py:CONSTRAINT_DEFAULTS` — remove `max_friday_broadmeadow`, `gosford_friday_games`, `maitland_friday_games`, `gosford_friday_rounds` (no longer used)

### 6. Run the full quick test suite

The test bar is in `docs/ATOMIZATION_HANDOFF.md`. Must remain at ≥1268 passing (the post-Phase-3a baseline). The replacement should be a near-zero net change in test count: -8 atom tests for the 4 removed atoms, +N tests for the new FORCED multi-scope coverage.

### 7. Document the change

Update the following so the convention is clear to future AI sessions:

- **`CLAUDE.md`** — add a section "Per-venue / per-day game counts use FORCED_GAMES, not constraints." Briefly: "If you find yourself adding a hardcoded count constraint, stop. Add a FORCED entry to season config instead. The constraint mechanism is reserved for *structural* rules (no-double-booking, adjacency, balance) — not for *count budgets*, which the FORCED dictionary handles natively."
- **`docs/CONSTRAINT_INVENTORY.md`** — drop the deleted atoms from the table; note that `BroadmeadowFridayCount` etc. are now expressed as FORCED entries.
- **`docs/ATOMIZATION_PLAN.md`** — mark Phase 3a as DONE (with the count-atom retraction noted), update the cluster's atom count from 8 down to 4 (`PHLConcurrencyAtBroadmeadow`, `PHLAnd2ndConcurrencyAtBroadmeadow`, `PHLRoundOnePlay`, `PreferredDates`).
- **`docs/ATOMIZATION_HANDOFF.md`** — append a "Phase 3a retraction + FORCED migration" entry to the commit table.
- **`config/season_2026.py`** — add a comment header on the new FORCED entries explaining the pattern (so editors don't try to re-introduce the atoms).
- **`docs/PERENNIAL_RULES.md`** (if it exists, otherwise create) — describe the perennial Friday-count rule as a season-level FORCED entry pattern, not a constraint.

---

## Open question

If a future season DOES want a hardcoded floor (e.g. "broadmeadow must have at least 1 PHL game per Friday it's open"), is that still a FORCED rule (`constraint='greatere', count=1`) per Friday, or a constraint? Keep it as FORCED — same mechanism — and only add a new constraint class if the rule cannot be expressed as a per-scope count.

---

## Suggested commit shape

Two commits:

1. **`feat(forced-games): support 'club' filter + add multi-scope consistency validation`**
   — utils.py + tests/test_forced_games_multi_scope.py + tests/test_config_validation.py extension.
2. **`refactor(constraints): retire Friday-count atoms in favor of FORCED_GAMES entries`**
   — delete the four atom files, update unified.py / registry.py / atom tests / season_2026.py / docs.

Keep them separate so the FORCED enhancement is reviewable on its own and doesn't get tangled with the atom removal.

---

## After this task — what's left in the atomization plan

The full pickup state, in priority order:

| Phase | Scope | Notes |
|---|---|---|
| **3a (retraction)** | This document | Remove Friday-count atoms, migrate to FORCED |
| **3b** | Atomize `ClubDayConstraint` (5 atoms) | Pickup notes in `docs/ATOMIZATION_HANDOFF.md` Phase 3 section |
| **3c** | Atomize `ClubVsClubAlignment` (3 atoms + `PHLAnd2ndBackToBackSameField`) | Same |
| **4** | FORCED/BLOCKED count adjusters for the **remaining** count-sensitive constraints — `ClubVsClubCoincidence`, `EqualMatchUpSpacing`, `MaitlandHomeGrouping`, `AwayAtMaitlandGrouping`, `EqualGames`. Friday counts are **out of scope** now (handled by FORCED entries directly). | Each adjuster's formula must be reported to the user before commit |
| **6** | Generic home-ground rename (`MaitlandHomeGrouping` → `NonDefaultHomeGrouping`, etc.) | `AWAY_VENUE_RULES` skeleton already in `config/defaults.py` |
| **7a** | Tests on real sampled data + per-club / per-type / per-severity violation breakdown | Patterns in `docs/ATOMIZATION_PLAN.md` 7a-bis |
| **7b** | Configurable `SOLVER_STAGES` in season config + CLI flags `--stages-config`, `--stage-only`, `--skip-stage`, `--list-stages` | |
| **7c** | Move `constraints/{original,ai}.py` and `constraints/archived_equalspacing_original.py` to `constraints/archived/`. Add `tests/test_no_legacy_imports.py` to forbid prod imports. Remove `--ai` CLI flag. | |
| **7d** | Final docs sweep — `CLAUDE.md` rewrite, `docs/HARNESS.md` + `docs/STAGES.md` new, `docs/CONSTRAINT_INVENTORY.md` becomes single source for constraint table | |

The hand-off doc `docs/ATOMIZATION_HANDOFF.md` remains the canonical pickup point for the rest. **Update it at the end of this task** to reflect that Friday-count atoms are gone.

---

## Reporting template (for the user when you finish)

```
## FORCED-as-count-rules migration — DONE

**Commits:**
- <hash> feat(forced-games): support 'club' filter + multi-scope consistency validation
- <hash> refactor(constraints): retire Friday-count atoms in favor of FORCED_GAMES entries

**Files removed:** N atom files + their registry/unified entries / tests
**Files added:** N FORCED entries in season_2026.py + N test cases

**Validation phases added to validate_game_config:**
- Phase X: <name> — <one-line behavior>

**Tests:** <N passing / M total> (was 1268)

**Regressions / parity gaps:**
- <none> OR
- <named gap, why intentional>

**Docs updated:**
- CLAUDE.md — added "Per-venue counts use FORCED, not constraints" section
- docs/CONSTRAINT_INVENTORY.md — Phase 3a cluster reduced from 8 atoms to 4
- ...

Ready for Phase 3b (ClubDayConstraint atomization)?
```
