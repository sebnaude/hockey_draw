# Atomization Plan — Unified Constraints, Helper-Var Registry, Generic Home/Away

**Branch:** `final-form` (worktree at `C:/Users/c3205/Documents/Code/python/draw-final-form`)
**Status:** APPROVED — partially implemented.

## Phase status (as of commit `48f5222`)

| Phase | Status | Commit |
|---|---|---|
| 0 — Constraint inventory | ✅ DONE | `6e16d14` |
| 1 — Helper-Var Registry | ✅ DONE | `244f8cd` |
| 2 — ConstraintInfo extension | ✅ DONE | `c64c1d4` |
| 3a — Atomize PHLAndSecondGradeTimes | ✅ DONE — `1956608` shipped 8 atoms; per-venue Friday count atoms (Broadmeadow / Gosford / Maitland) retired in favor of `FORCED_GAMES` entries (see `docs/FORCED_GAMES_AS_COUNT_RULES.md`). Cluster now has 5 atoms: `PHLConcurrencyAtBroadmeadow`, `PHLAnd2ndConcurrencyAtBroadmeadow`, `GosfordFridayRoundsForced`, `PHLRoundOnePlay`, `PreferredDates`. | `1956608` + retraction |
| 3b — Atomize ClubDayConstraint | ⬜ NOT STARTED | — |
| 3c — Atomize ClubVsClubAlignment | ⬜ NOT STARTED | — |
| 4 — FORCED/BLOCKED count adjusters | ⬜ NOT STARTED (depends on 3) | — |
| 5 — Constants migration | ✅ DONE | `535cac3` |
| 6 — Generic home-ground rename | 🟡 PREP DONE — `AWAY_VENUE_RULES` skeleton committed (`48f5222`); rename + per-club iteration still TODO | partial |
| 7a — Tests on real sampled data | ⬜ NOT STARTED | — |
| 7b — Configurable stages | ⬜ NOT STARTED | — |
| 7c — Move legacy to `constraints/archived/` | ⬜ NOT STARTED (depends on 3) | — |
| 7d — Documentation update | 🟡 partial — `docs/CONSTRAINT_INVENTORY.md` (Phase 0) and `docs/HELPER_VARS.md` (Phase 1) shipped | partial |

Test baseline at this point: **1246 passed, 1 skipped** (started at 1216).

The hand-off doc `docs/ATOMIZATION_HANDOFF.md` is the canonical pickup point for the next session.
**Goal recap:** one idea per constraint; constraint+helper-var registry; zero hardcoded constants in constraints; FORCED/BLOCKED-aware count adjustments; generic home-ground concept; tests on real sampled data; configurable stage assignment; per-club / per-type violation breakdowns.

This plan is in 7 phases, each one independently shippable. Each phase has a concrete deliverable, a test bar, and a list of files touched.

---

## Decisions (locked in)

| # | Question | Decision |
|---|---|---|
| 1 | Atom naming | Keep `PHLConcurrencyAtBroadmeadow`, `ClubDayParticipation`, etc. — descriptive, no `Atom` suffix. |
| 2 | Legacy classes (`original.py`, `ai.py`) | **Reference-only.** Move into `constraints/archived/` (NOT `_legacy/` — that path is gitignored). The pipeline must NOT be able to import them in prod. Existing `constraints/archived_equalspacing_original.py` also moves under this directory. |
| 3 | Same as #2 — legacy code moves to `constraints/archived/`, prod imports forbidden. |
| 4 | CLUB_DAYS opponent format | Atom `ClubDayOpponentMatchup` matches `original.py` behavior — supports `{'date': datetime, 'opponent': 'OppClub'}`. `ai.py`'s simpler form (date-only) was a regression and is dropped. |
| 5 | Config key for per-club home-venue rules | `AWAY_VENUE_RULES`. |
| 6 | Tests | Both static fixtures AND programmatic per-test (see Phase 7a below). Plus per-club + per-type breakdowns in tester output. |
| 7 | Staged execution path | **Configurable** — user picks how many stages and which atoms go in each. See "Configurable Stages" section below. |

## Context (carried over from working session)

Captured here so anyone picking up the work has a self-contained briefing.

**Repo state (commit `cd8a338` on `final-form`):**
- `final-form` is the unified-constraints branch. Recent commits: `6eb2f78` (initial merge from `feat/ai-updates`), `2f15ee1` (additive restore of dropped content), `3e9d4a5` (drop conflicting season_2026 entries), `cd8a338` (cherry-picked forced-games multi-scope fix).
- `feat/ai-updates` is at `8031354` (forced-games multi-scope fix). All other WIP from feat/ai-updates is now on final-form.
- 4 commits unpushed across both branches.
- Test suite: **1211 passed, 1 skipped** out of 1212 (excluding two slow integration tests).

**Architectural state of `final-form`:**
- `constraints/unified.py` (~1404 lines) — 2-stage hard/soft engine, `SharedVariablePool` (lazy helper-var cache), 27+ pre-computed groupings, lookup caches (team_club, club_teams_map, club_home_field, etc.). This is the architectural achievement of `final-form`. The atomization replaces its inline constraint application with atom-based dispatch.
- `constraints/registry.py` — currently maps canonical name ↔ solver class ↔ tester method. 21 entries (19 solver constraints + ForcedGames + BlockedGames as `tester_only=True`). Phase 2 extends with `atom_group`, `required_helpers`, `forced_blocked_adjuster`.
- `constraints/original.py` (1733 lines) — legacy classes incl. PHL locked-week HACKs that must be redesigned into atoms (the HACK markers in PHLAndSecondGradeTimes lines ~242-301 are explicit user-flagged tech debt).
- `constraints/ai.py` (2040 lines) — parallel "AI-enhanced" implementations.
- `utils.py` (3571 lines) — `generate_X` returns `(X, conflicts)` (NOT `(X, Y, conflicts)` — final-form handles dummies via SharedVariablePool, not Y dict). Includes the 20-phase `validate_game_config` harness, `_build_team_lookups`, `_resolve_team_name`, `_build_forced_game_rules` (returns 3-tuple `(scope_groups, constraint_types, constraint_counts)`), `_get_matching_forced_scopes` (multi-scope match — bug fix `cd8a338`), `validate_draw_keys`, `repair_locked_keys`.
- `analytics/tester.py` (2495 lines) — `DrawTester` with slack-aware checks. Includes `_check_forced_games` and `_check_blocked_games`. Uses `(date, slot)` for adjacency grouping (not `(date, slot, field)`). `_check_club_game_spread` uses dynamic per-club overlap bound `T // 2 - 1` and Broadmeadow-only filter.
- `config/defaults.py` — perennial config (FIELDS, DAY_TIME_MAP, HOME_FIELD_MAP, GRADE_ORDER, PERENNIAL_BLOCKED_GAMES).
- `config/season_2026.py` — 261 FORCED_GAMES, 69 BLOCKED_GAMES (incl `*PERENNIAL_BLOCKED_GAMES` spread), 2 CLUB_DAYS.
- `tests/fixtures/draw_2026_first6weeks.json` — 6-week fixture used by integration tests.

**Key facts that drove design choices:**
- Variable key = 11-tuple `(team1, team2, grade, day, day_slot, time, week, date, round_no, field_name, field_location)` where `team1` is alphabetically first. Home/away is determined by `field_location`, NOT by team1/team2 position.
- A "dummy" key is a 4-tuple `(team1, team2, grade, index)` — short keys are skipped via `len(key) < 11` or `not key[3]`.
- `home_field_map` example: `{'Maitland': 'Maitland Park', 'Gosford': 'Central Coast Hockey Park'}`. All other clubs default to Newcastle International Hockey Centre (Broadmeadow).
- A FORCED entry can match multiple scopes simultaneously — bug fix `cd8a338` ensures the variable registers against EVERY matching scope (date-scope + team-scope) so each scope's `sum == N` constraint sees it as a candidate.
- The user's example for FORCED-aware adjusters: `ClubVsClubAlignment` counts how many times a club-pair meets per grade. If FORCED forces 2 PHL games of that pair onto Friday Gosford (not Sunday), the standalone Sunday "alignment day" loses 2 PHL games for that pair — which can leave a 2nd-grade game with no PHL counterpart. Atoms must adjust expected counts based on what's been forced/blocked.
- Aggregate per-team home/away constraint was intentionally removed (CLAUDE.md: "by design"). Per-pair balance only.
- Multi-idea constraints to atomize (3 confirmed):
  - `PHLAndSecondGradeTimes` (133 lines) → 7 atoms
  - `ClubDayConstraint` (121 lines) → 5 atoms
  - `ClubVsClubAlignment` (127 lines) → 3 atoms + extracted `PHLAnd2ndBackToBackSameField`

---

## Phase 0 — Pre-flight & freeze the inventory  *(no code changes)*  — ✅ DONE (`6e16d14`)
**Deliverable shipped:** `docs/CONSTRAINT_INVENTORY.md` — 21-row table of every registered constraint, atom-target name(s), and 9 findings flagged for sign-off.


Before atomizing, lock down what we have:
- Snapshot the current 21 registered constraints (registry.py) and the 19 solver classes in `original.py`/`ai.py` into a markdown table.
- For each constraint write a one-line "what it actually does" derived from the code, not the docstring.
- For each multi-idea constraint, list the atoms we'll split it into (this PR has the candidates already; we'll lock the names).

**Deliverable:** `docs/CONSTRAINT_INVENTORY.md` — one table.
**Test bar:** none (doc only).
**Files:** `docs/CONSTRAINT_INVENTORY.md` (new).

---

## Phase 1 — Helper-Variable Registry — ✅ DONE (`244f8cd`)
**Deliverable shipped:** `constraints/helper_vars.py` with `HelperVarRegistry` (declare/freeze/get_declared + pool-style legacy API). `unified.py` wired (`self.registry` + `self.pool` alias). 15 unit tests. `SharedVariablePool` retained as alias for back-compat. Documented in `docs/HELPER_VARS.md`.

Note for Phase 3 atoms: the registry's pool-style `.get(key)` returns the cached pool var (or None). Use `registry.get_declared(kind, key)` for the new declarative path.


**The problem today:** `unified.py` has a `SharedVariablePool` that lazy-creates BoolVars/IntVars and caches them. It works but it's an in-line cache, not a declarative registry. Constraints can't *declare* what they need; they reach in and ask. There's no compile-time check that a constraint built its own helper without going through the pool. There's no way to introspect "what helper-vars does the model use" for debugging.

**Design:**

```python
# constraints/helper_vars.py (new)

class HelperVar:
    """Declarative spec for a helper var. Registered once per (kind, key)."""
    def __init__(self, kind: str, key: tuple, builder: Callable, description: str):
        self.kind = kind          # e.g. 'is_slot_used', 'is_phl_friday'
        self.key = key            # tuple uniquely identifying the instance
        self.builder = builder    # called once: builder(model, X, data) -> Var
        self.description = description

class HelperVarRegistry:
    def declare(self, kind: str, key: tuple, builder, description: str):
        """Constraints call this in their declare() phase to register intent."""
    def get(self, kind: str, key: tuple) -> Var:
        """Get-or-create the var. Builds via the registered builder."""
    def freeze(self):
        """Called once after all constraints declare. Builds every declared var.
        Constraints can no longer declare after freeze."""
```

**Catalog of helper vars** (from current unified.py audit + what atomization needs):

| Kind | Key shape | Built by | Used by |
|---|---|---|---|
| `is_slot_used` | (week, day, location, day_slot) | sums vars at that slot, ≥1 ⇒ 1 | EnsureBestTimeslotChoices, ClubGameSpread |
| `is_field_used` | (week, day, location, field, day_slot) | sums vars at that field+slot, ≥1 ⇒ 1 | EnsureBestTimeslotChoices, MinimiseClubsOnAField |
| `weekend_used` | (week, location) | sums all venue vars in week | EnsureBestTimeslotChoices |
| `team_plays_in_week` | (team, week) | sum of team's vars in that week | NoDoubleBookingTeams (built-in), Adjacency atoms |
| `pair_plays_in_week` | (team1, team2, week, grade) | OR over all timeslot vars | EqualMatchUpSpacing, ClubVsClubAlignment |
| `pair_plays_on_date` | (team1, team2, date, grade) | OR over date's timeslot vars | new ClubVsClubCoincidence atom |
| `club_plays_on_date_at_field` | (club, date, field) | OR over teams from club at that field | ClubDay-SameField atom |
| `club_grade_in_slot` | (club, grade, week, day, day_slot) | OR over teams of (club,grade) in slot | ClubGradeAdjacency |
| `home_game` | (team, week, opponent) | indicator: team played at home | FiftyFifty, MaitlandHomeGrouping |
| `is_phl_friday` | (week,) | indicator: PHL plays Friday this week | PHL Friday count atoms |

**Constraint contract:**
```python
class Constraint:
    def declare_helpers(self, registry: HelperVarRegistry, data): ...
    def apply(self, model, X, data, registry: HelperVarRegistry): ...
```

The `UnifiedConstraintEngine` runs:
1. Every constraint's `declare_helpers()` → registry collects intents.
2. `registry.freeze()` → builds every distinct helper var **once**.
3. Every constraint's `apply()` → uses `registry.get(...)` to fetch.

**Deliverable:** `constraints/helper_vars.py`, integration in `UnifiedConstraintEngine`.
**Test bar:**
- Unit tests on registry: declare/get returns same var; double-declare with same key is OK; declare after freeze raises; get for undeclared raises.
- Integration: take 3 existing constraints (NoDoubleBookingFields, EnsureBestTimeslotChoices, ClubGameSpread), port them to declare via registry, prove same model output as before.
**Files:** `constraints/helper_vars.py` (new), `constraints/unified.py` (engine changes), `tests/test_helper_var_registry.py` (new).
**No new behavior** — purely an architectural shift. Behavior tests must show identical solutions on a sampled fixture.

---

## Phase 2 — Constraint Registry (extend the existing one) — ✅ DONE (`c64c1d4`)
**Deliverable shipped:** `ConstraintInfo` gained `atom_group`, `required_helpers`, `forced_blocked_adjuster`. `HELPER_VAR_CATALOG` set lists allowed helper-var kinds. New helpers `get_atoms_in_group`, `get_adjuster`, `validate_required_helpers`. 7 new tests.


The existing `constraints/registry.py` is a name-mapping registry (canonical ↔ solver class ↔ tester method). It should also know:
- Severity, slack key, has-soft-component (already there).
- **Required helper-var kinds** — what helpers each atom declares.
- **Atom group** — which atoms came from splitting one bigger constraint (so reports can group them).
- **FORCED/BLOCKED count-adjuster hook** — optional callable that, given FORCED_GAMES + BLOCKED_GAMES, returns adjustment metadata for this constraint's count expectations (this is the mechanism for the user's example).

**Augmented `ConstraintInfo`:**
```python
@dataclass
class ConstraintInfo:
    canonical_name: str
    solver_class_names: List[str]
    tester_check_methods: List[str]
    tester_violation_names: List[str]
    severity_level: int
    slack_key: Optional[str] = None
    has_soft_component: bool = False
    tester_only: bool = False
    # NEW
    atom_group: Optional[str] = None              # e.g. 'PHLAndSecondGradeTimes'
    required_helpers: List[str] = field(default_factory=list)   # kinds it uses
    forced_blocked_adjuster: Optional[Callable] = None
```

The atomized constraints get registered with `atom_group='PHLAndSecondGradeTimes'` so reports can roll them up.

**Deliverable:** registry schema + registration of atoms (added in later phases).
**Test bar:** registry-completeness tests already exist; extend to assert every atom's required_helpers exist in the helper-var catalog.
**Files:** `constraints/registry.py`, `tests/test_constraint_registry.py`.

---

## Phase 3 — Atomize the 3 multi-idea constraints

For each, split into atoms. **Each atom** = 1 file in `constraints/atoms/`, 1 class, 1 idea, registers via the registry, declares its helpers.

### 3a. `PHLAndSecondGradeTimes` → 4 atoms (post-FORCED-migration)
| Atom | Idea |
|---|---|
| `PHLConcurrencyAtBroadmeadow` | At Broadmeadow, no two PHL games in the same `(slot, location)`. |
| `PHLAnd2ndConcurrencyAtBroadmeadow` | At Broadmeadow, PHL and same-club 2nd grade can't share a slot. |
| `PHLRoundOnePlay` | Every PHL team plays in round 1. |
| `PreferredDates` | Preferred dates carry positive weight. |

**Removed from this list (now expressed as `FORCED_GAMES` entries in season config):**
- `BroadmeadowFridayCount` (max 3 PHL Fridays at NIHC) — `{grade='PHL', day='Friday', field_location=NIHC, count=3, constraint='lesse'}`
- `GosfordFridayCount` (exactly 8 per season) — `{grade='PHL', day='Friday', field_location=Gosford, count=8, constraint='equal'}`
- `MaitlandFridayCount` (exactly 2 per season) — `{grade='PHL', day='Friday', field_location=Maitland, count=2, constraint='equal'}`
- `GosfordFridayRoundsForced` ({2,4,5,9,10}) — already covered by per-round FORCED entries in `season_2026.py:555-603`

See `docs/FORCED_GAMES_AS_COUNT_RULES.md` for the design decision and migration steps. The user's rule: **count budgets use `FORCED_GAMES`, constraints reserve for structural rules** (no-double-booking, adjacency, balance, spacing).

### 3b. `ClubDayConstraint` → 4 atoms
| Atom | Idea |
|---|---|
| `ClubDayParticipation` | All teams of that club play on that date. |
| `ClubDayIntraClubMatchup` | Same-grade duplicates within the club must matchup. |
| `ClubDayOpponentMatchup` | If `opponent` set, force cross-club games (today only `original.py` has this — atom unifies). |
| `ClubDaySameField` | All games on same field. |
| `ClubDayContiguousSlots` | All games in contiguous slots. |

### 3c. `ClubVsClubAlignment` → 3 atoms
| Atom | Idea |
|---|---|
| `ClubVsClubCoincidence` | Club-pair meetings should coincide on the same date across grades (≥ N grades on that date). |
| `ClubVsClubFieldLimit` | When coinciding, ≤ 2 fields. |
| `ClubVsClubDeficitPenalty` | Soft penalty for missing coincidences. |

The PHL/2nd back-to-back same-field algorithm that was hacked onto `ClubVsClubAlignment` becomes a separate atom: `PHLAnd2ndBackToBackSameField`.

**Per-atom test bar:**
1. Sampled-data unit test that triggers a violation by construction → atom adds an infeasibility / penalty.
2. Sampled-data unit test that satisfies the constraint → atom adds nothing.
3. Equivalence test: applying all atoms together produces same solution as the legacy combined constraint on the 2026 fixture.

**Files:**
- `constraints/atoms/phl_concurrency.py`, `phl_2nd_concurrency.py`, `broadmeadow_friday_count.py`, `gosford_friday_count.py`, `maitland_friday_count.py`, `phl_round_one_play.py`, `preferred_dates.py`
- `constraints/atoms/club_day_*.py` (4 files)
- `constraints/atoms/club_vs_club_*.py` (3 files)
- `constraints/atoms/phl_2nd_back_to_back.py`
- `tests/atoms/test_*.py` for each
- `constraints/unified.py` updated to apply atoms instead of legacy combined classes (for the unified path only; legacy `original.py`/`ai.py` keep their combined classes for the staged path until phase 7 retires them).

**Important — DO NOT delete `original.py`/`ai.py` versions yet.** Keep them callable from `main_staged.py` so we can A/B compare. They get retired in Phase 7 once atoms prove out.

---

## Phase 4 — FORCED/BLOCKED-aware count adjustments  *(the user's example mechanism)*

**The user's case:** ClubVsClubAlignment counts how many times each club-pair meets per grade. If FORCED_GAMES forces 2 PHL games of that pair onto Friday nights at Gosford (i.e. NOT Sunday), then on the standalone Sunday "alignment day" only N-2 games of PHL exist for that pair, which can leave a 2nd-grade game with no PHL counterpart.

**Mechanism:** every count-sensitive constraint declares a `forced_blocked_adjuster` callable in its registry entry. Signature:

```python
def adjuster(constraint_data: dict, forced_games: list, blocked_games: list) -> dict:
    """Returns a dict of count overrides keyed by whatever the constraint cares about.

    For ClubVsClubCoincidence:
      input — forced/blocked partial keys
      output — { (club_pair, grade): expected_sunday_meetings }
                where expected = total_meetings - forced_off_sunday - blocked_on_sunday
    """
```

The `UnifiedConstraintEngine` runs every adjuster after FORCED/BLOCKED are parsed, before constraints `apply()`. The adjuster output is stuffed into `data['count_adjustments'][canonical_name]` and the atom reads it.

**Catalog of constraints that need adjusters:**
| Constraint | Why |
|---|---|
| `ClubVsClubCoincidence` | Forcing pair off Sunday lowers expected coincidences. (User's example.) |
| ~~`BroadmeadowFridayCount` / `GosfordFridayCount` / `MaitlandFridayCount`~~ | **Out of scope — retired.** Per-venue Friday counts are FORCED_GAMES entries now (see `docs/FORCED_GAMES_AS_COUNT_RULES.md`). The locked-week HACK in `original.py` becomes obsolete when those legacy classes archive in Phase 7c. |
| `EqualMatchUpSpacing` | Forcing pair-meetings onto specific weeks reduces flexibility for spacing. |
| `MaitlandHomeGrouping` | Forcing a home weekend changes the consecutive-window calculation. |
| `AwayAtMaitlandGrouping` | Forcing an away match at Maitland changes the away-clubs-per-week count. |
| `EqualGames` | If a team is FORCED into N games already, the per-team total budget is reduced. |

For each of the above I'll list the adjustment formula in the `ATOMIZATION_PLAN.md` as I implement, **and report each one to you** so you can sign off on the formula.

**Generalization:** the registry doesn't care WHAT the adjuster does — it just calls it. So when a future constraint cares about FORCED/BLOCKED in some new way, it just declares its own adjuster. No solver-engine surgery needed.

**Test bar:**
- Adjuster unit tests with synthesized FORCED entries → assert correct count override.
- Integration: solve a 4-week mini-fixture with a FORCED Friday game; assert ClubVsClubCoincidence doesn't penalize the missing Sunday game.

**Files:** each atom file gets a `def forced_blocked_adjuster(...)` registered in registry; `tests/atoms/test_*_adjuster.py`.

---

## Phase 5 — Constants migration to config — ✅ DONE (`535cac3`)
**Deliverable shipped:** Perennial `CONSTRAINT_DEFAULTS` in `config/defaults.py` (every constraint param has a default; seasons override only what they want changed). `_merge_constraint_defaults()` in `utils.py` merges season overrides into perennials. New keys `phl_adjacency_window_minutes`, `gosford_friday_rounds`, `worst_timeslot_time`. `unified.py` reads `PHL_ADJACENCY_MINUTES` and Gosford-Friday rounds from config; 4 dead class constants removed. 8 new tests.

`original.py` and `ai.py` left untouched (reference-only, archived in Phase 7c).


Audit found these hardcoded constants in `constraints/unified.py` (and similar spots in original/ai):

| Current location | Constant | New config home |
|---|---|---|
| unified.py | `PHL_ADJACENCY_MINUTES = 180` | `CONSTRAINT_DEFAULTS['phl_adjacency_window_minutes']` |
| unified.py | `BROADMEADOW_MAX_SLOTS = 6` | derived from `DAY_TIME_MAP[broadmeadow]['Sunday']` length — no constant needed |
| unified.py | `MAITLAND_AWAY_HARD_LIMIT = 3` | already exists as `away_maitland_max_clubs` — wire it |
| unified.py | `BROADMEADOW`, `GOSFORD`, `MAITLAND` venue strings | from `home_field_map` values |
| unified.py | `GRADE_ORDER` list | already exists as `grade_order` in SEASON_CONFIG — wire it |
| unified.py | round set `[2, 4, 5, 9, 10]` for Gosford Friday | `CONSTRAINT_DEFAULTS['gosford_friday_rounds']` |
| original.py / ai.py | `'Newcastle International Hockey Centre'` literal | use `home_field_map` reverse-lookup default |
| original.py / ai.py | `'Maitland Park'`, `'Central Coast Hockey Park'` | use `home_field_map` |
| original.py / ai.py | `'PHL'`, `'2nd'`, `'Friday'`, `'Sunday'` | grade strings — use `grade_order`; day strings — pass via `data['playing_days']` |

**Pattern:** every constant becomes either (a) a key in `CONSTRAINT_DEFAULTS`, or (b) derived from existing data structures.

**Test bar:** parametrize a couple of atom tests with the constant value coming from config; flip the value, observe atom behavior changes.

**Files:** `config/defaults.py` (extend `CONSTRAINT_DEFAULTS`), every atom that referenced the constant.

---

## Phase 6 — Generic Home-Ground — 🟡 PREP DONE (`48f5222`); rename TODO

**Skeleton shipped:** `AWAY_VENUE_RULES` dict in `config/defaults.py` keyed by club name with per-club `max_consecutive_home`/`friday_games`/`max_away_clubs`. No constraint reads from it yet.

**Still TODO:** rename + per-club iteration. The constraint logic still hardcodes `'Maitland'`/`'Gosford'` strings; the rename touches ~20 files / ~100 references and needs registry aliases preserved for severity/slack lookups. See pickup notes in `docs/ATOMIZATION_HANDOFF.md`.


**Today:** `home_field_map` exists (`{'Maitland': 'Maitland Park', 'Gosford': 'Central Coast Hockey Park'}`) but constraints hard-code "Maitland" / "Gosford" everywhere.

**Target model:**
- A team is "non-default home" iff `home_field_map.get(team.club.name)` is set AND it's not Broadmeadow.
- Every "Maitland-specific" or "Gosford-specific" constraint becomes: scope = "all non-default-home clubs", parameterized per club.

**Refactors:**
| Today | Generic |
|---|---|
| `MaitlandHomeGrouping` (sliding window of consecutive home weeks) | `NonDefaultHomeGrouping` — applied per non-default-home club |
| `MaxMaitlandHomeWeekends` | folded into the above (was redundant) |
| `AwayAtMaitlandGrouping` | `AwayAtNonDefaultGrouping` per club |
| `FiftyFiftyHomeandAway` | already iterates over `home_field_map.keys()` — verify it's clean, no string literals |
| `MaitlandHomeGrouping` | `NonDefaultHomeGrouping` |

**Plus:** add a config knob per club for the per-club specifics:
```python
AWAY_VENUE_RULES = {
    'Maitland': {'max_consecutive_home': 1, 'friday_games': 2, ...},
    'Gosford':  {'max_consecutive_home': 2, 'friday_games': 8, ...},
}
```

**Adding/removing a club** is then: add/remove an entry in `home_field_map` + (optionally) `AWAY_VENUE_RULES`. No constraint changes.

**Test bar:**
- Add a fictional 3rd non-default-home club to a fixture and assert constraints scope to it correctly.
- Remove `'Gosford'` from `home_field_map` and verify Gosford-specific constraints go silent (rather than crash).

**Files:** rename `MaitlandHomeGrouping` → `NonDefaultHomeGrouping` etc. (keep old class names as aliases for back-compat in registry); `config/defaults.py` adds `AWAY_VENUE_RULES`.

---

## Phase 7 — Test suite + harness improvements + retirement

### 7a. Tests on real sampled data, no mocks

**Already partially done** in `tests/fixtures/draw_2026_first6weeks.json` and `tests/test_constraints_realdata.py`. Extend to:

**Per-atom test pattern (mandatory for every atom):**
1. **Solo-clean test** — apply atom to a clean fixture → assert no infeasibility, no penalty.
2. **Solo-violation test** — apply atom to a fixture with a deliberately-baked violation → assert the atom flags the expected violation/penalty.
3. **Programmatic variant** — same shape as #2 but the violation is constructed in-test from the clean fixture, not a separate file. Confirms tests can build their own scenarios.

**Test fixture layout:**
- `tests/fixtures/draw_2026_first6weeks.json` — clean baseline (already exists).
- `tests/fixtures/violations/` (new directory) — one fixture per category of violation:
  - `club_game_spread_overlap.json` — clubs with overlapping games beyond limit
  - `club_grade_adjacency.json` — same-club adjacent grades in same slot
  - `phl_friday_overcount.json` — too many PHL Friday games at Broadmeadow
  - `home_away_imbalance.json` — non-default-home club skewed
  - `club_day_split_field.json` — club day games on two fields
  - ...one per atom that benefits from a static fixture.
- Each fixture file has metadata header `"_violations": ["AtomName1", "AtomName2"]` so tests can self-discover what they should be flagging.

**No `unittest.mock`, no `monkeypatch` on solver / data dicts.** Stub out only filesystem I/O.

### 7a-bis. Per-club / per-type violation breakdown in DrawTester

**Today:** `DrawTester.run_violation_check()` returns a flat list of `Violation` objects per constraint.

**Required:** add structured aggregation, especially for SOFT violations, so analysing a draw shows *which* clubs are at the limit / over-limit / breaking soft penalties, not just a flat count.

**API extension to `ViolationReport`:**
```python
@dataclass
class ViolationBreakdown:
    by_club: Dict[str, List[Violation]]      # club_name -> violations involving that club
    by_type: Dict[str, List[Violation]]      # constraint_canonical_name -> violations
    by_severity: Dict[str, List[Violation]]  # 'CRITICAL'/'HIGH'/.../'VERY LOW' -> violations
    soft_pressure: Dict[str, Dict[str, float]]
    # ^ canonical_name -> { 'at_limit': N_clubs, 'over_limit': N_clubs,
    #                       'total_penalty': X, 'worst_club': 'ClubName',
    #                       'worst_value': X }

@dataclass
class ViolationReport:
    ...existing fields...
    breakdown: ViolationBreakdown
```

**Example use case (the user's example):**
For `ClubGameSpread` with rule "max 1 overlap game per weekend per club":
- A club has 4 timeslot-numbers in a weekend, only 2 may share a slot, the other 2 must be different.
- If a club violates this, the breakdown shows: which weekend, which 4 slots, which 2 are duplicated, and which other clubs are at-limit (1 overlap = at limit) vs over-limit (≥2 overlaps).
- For SOFT violations specifically, `soft_pressure` summarises: how much penalty came from each club, who's worst, who's at risk.

**Each atom must populate `Violation.affected_clubs: List[str]` and `Violation.metric_value: Optional[float]`** so the breakdown can aggregate without re-deriving.

### 7b. Configurable Stages (the user's request)

**Today:** `main_staged.py` has a hardcoded `STAGES` dict mapping stage names to lists of constraint classes (legacy). The `--staged` CLI flag activates it.

**Target:** the user picks how many stages and which atoms go in each via config.

**Design:**
```python
# config/defaults.py — sensible default
DEFAULT_STAGES = [
    {
        'name': 'critical_feasibility',
        'description': 'Hard feasibility — every constraint that must hold for a valid draw',
        'atoms': [
            'NoDoubleBookingTeams', 'NoDoubleBookingFields',
            'EqualGames', 'BalancedMatchups',
            'PHLConcurrencyAtBroadmeadow', 'PHLAnd2ndConcurrencyAtBroadmeadow',
            'BroadmeadowFridayCount', 'GosfordFridayCount', 'MaitlandFridayCount',
            'PHLRoundOnePlay',
        ],
    },
    {
        'name': 'home_away_balance',
        'description': 'Per-pair home/away + non-default-home grouping',
        'atoms': [
            'FiftyFiftyHomeandAway',
            'NonDefaultHomeGrouping', 'AwayAtNonDefaultGrouping',
        ],
    },
    {
        'name': 'club_alignment',
        'description': 'Cross-grade coincidence + field limits',
        'atoms': [
            'ClubGradeAdjacency',
            'ClubVsClubCoincidence', 'ClubVsClubFieldLimit',
            'PHLAnd2ndBackToBackSameField',
        ],
    },
    {
        'name': 'club_day',
        'atoms': [
            'ClubDayParticipation', 'ClubDayIntraClubMatchup',
            'ClubDayOpponentMatchup', 'ClubDaySameField', 'ClubDayContiguousSlots',
        ],
    },
    {
        'name': 'soft_optimisation',
        'atoms': [
            'EqualMatchUpSpacing', 'ClubGameSpread',
            'ClubVsClubDeficitPenalty', 'PreferredDates',
            'EnsureBestTimeslotChoices', 'PreferredTimes',
            'MaximiseClubsPerTimeslotBroadmeadow', 'MinimiseClubsOnAFieldBroadmeadow',
        ],
    },
]

# Override per-season
# config/season_2026.py
SOLVER_STAGES = DEFAULT_STAGES  # or a custom list
```

**Per-stage options (optional fields on each stage dict):**
- `'name'` (required), `'description'` (optional)
- `'atoms'` (required) — list of canonical atom names from the registry
- `'time_limit_seconds'` (optional, default `SEASON_CONFIG['max_time_per_stage']`)
- `'use_prior_solution_as_hint'` (optional, default True)
- `'soft_only'` (optional, default False) — if True, this stage only applies penalties; no hard constraints
- `'requires_complete_solution'` (optional, default True) — if False, stage runs even on partial solutions

**Validation:**
- Every atom listed in any stage must exist in the constraint registry.
- Every registry-listed atom must appear in at least one stage (warn, don't fail) OR appear in an `'unstaged'` list explicitly excluded.
- No atom may appear in two stages (would re-add identical constraints).
- Validation runs at config load via a new `_validate_stages` phase in `validate_game_config`.

**CLI:**
- `--stages-config path/to/stages.json` — override the in-config stages with an external file (useful for diagnostics, not normal prod).
- `--stage-only NAME` — run only one named stage (debugging).
- `--skip-stage NAME` — skip a stage.
- `--list-stages` — print the configured stages and exit.

**Pipeline rewire:**
- `main_staged.py::main_staged()` reads `data['solver_stages']` (loaded from config), validates, and iterates stages.
- For each stage, instantiates the listed atoms via `registry.get_atom(name)`, calls `declare_helpers()` then `apply()` through the unified engine.
- Old hardcoded `STAGES` and `STAGES_AI` dicts are deleted. The `--ai` flag is deprecated (atoms are the implementation; there's no original/AI distinction anymore).

**Test bar:**
- Round-trip: load season config → 5 stages get registered → solver runs each stage → all 1212 tests pass.
- Edit `SOLVER_STAGES` to put `EqualGames` in stage 2 instead of stage 1 → solver still produces valid draw (just slower path).
- Validation tests: duplicate atom across stages raises; unknown atom name raises; missing required atom for a constraint raises with helpful suggestion.

**Files:** `main_staged.py`, `config/defaults.py` (add `DEFAULT_STAGES`), `utils.py` (`_validate_stages` phase), `tests/test_configurable_stages.py` (new), `run.py` (new CLI flags).

### 7b. Harness improvements (FORCED/BLOCKED handling)
This is the broader item from your prompt. Concrete improvements I'd ship:

1. **Typed key-matchers.** Today scope dicts use string keys (`'date'`, `'grade'`, `'field_location'`). Move to a `Scope` dataclass:
   ```python
   @dataclass(frozen=True)
   class Scope:
       grade: Optional[str|list] = None
       date: Optional[str] = None
       day: Optional[str] = None
       week: Optional[int] = None
       field_name: Optional[str] = None
       field_location: Optional[str] = None
       round_no: Optional[int] = None
       time: Optional[str] = None
       club: Optional[str] = None
       teams: Optional[list] = None
   ```
   Validates at config-load time. Catches typos like `'feild_location'`.

2. **Declarative FORCED entry effects.** Each FORCED/BLOCKED entry can declare which constraint adjusters it triggers. E.g.:
   ```python
   {'date': '2026-04-17', 'grade': 'PHL', 'count': 1,
    'reason': 'Norths-Gosford Friday',
    'affects_alignment': True,   # triggers ClubVsClubCoincidence adjuster
    'affects_friday_count': True}
   ```
   Optional metadata; the adjusters can also infer from scope. Useful for documentation / reports.

3. **Validator output as report.** The 20-phase `validate_game_config` already exists. Re-pipe its output into `analytics/preseason_report.py` so config conflicts show up in the pre-season report instead of just stderr.

4. **`generate_X` returns a typed `XBuild` object** (not just a dict + int) with: `vars`, `forced_groups`, `blocked_eliminations`, `home_filter_eliminations`, `phl_filter_eliminations`. Used by debugging and reports.

5. **Recently-shipped bug fix from `8031354`** (multi-scope forced match) becomes the *test* for "harness handles overlapping scopes correctly" — protects the regression.

### 7c. Move legacy classes to `constraints/archived/`

**Decision (locked):** legacy combined classes in `original.py` and `ai.py` are reference-only. The pipeline must NOT be able to use them in prod.

**Move:**
- `constraints/original.py` → `constraints/archived/original.py`
- `constraints/ai.py` → `constraints/archived/ai.py`
- `constraints/archived_equalspacing_original.py` → `constraints/archived/equalspacing_original.py` (formalize the existing convention)

**Pipeline lockdown:**
- `constraints/__init__.py` exports ONLY atoms + `UnifiedConstraintEngine` + `registry`. No re-export of archived modules.
- Add a `tests/test_no_legacy_imports.py` that greps the prod codebase (everything outside `constraints/archived/` and `tests/`) for `from constraints.original` / `from constraints.ai` / `from constraints.archived` and fails the test if any prod module imports them.
- Update `main_staged.py`, `analytics/*`, `run.py`, every script in `scripts/` to import only atom names via the registry.
- The `--ai` CLI flag is removed (no original/AI distinction in atoms).

**Retain for reference:**
- `constraints/archived/README.md` explaining: "These are pre-atomization implementations kept for historical reference. Do NOT import in production code; the pipeline is locked against this. Use the atoms in `constraints/atoms/` instead."

**Test bar:** the no-legacy-imports test passes; full test suite passes; solver produces equivalent draw to pre-atomization run on the 2026 fixture.

### 7d. Documentation update (task #9)
Rewrite CLAUDE.md sections:
- Constraint architecture → atoms + registry
- Helper-var registry → API + catalog
- FORCED/BLOCKED → adjusters
- Generic home-ground → non-default-home-club model
- Move constraint-list table from CLAUDE.md to `docs/CONSTRAINT_INVENTORY.md` (single source).

**Files:** `CLAUDE.md`, `docs/CONSTRAINT_INVENTORY.md`, `docs/HELPER_VARS.md`, `docs/HARNESS.md`.

---

## Order & dependencies

```
0 (inventory) ──┐
                ├─→ 1 (helper-var registry) ─→ 3 (atomize) ─→ 7c (retire legacy)
2 (constraint registry extend) ─┘                      │
                                                        ↓
                            4 (FORCED/BLOCKED adjusters)
                                                        ↓
5 (constants migration) ────────────────────────────────┤
6 (generic home-ground) ────────────────────────────────┤
                                                        ↓
7a (tests) ─ 7b (harness) ─ 7d (docs)
```

**Critical path:** Phase 1 (helper-var registry) blocks Phase 3 (atomization). Phase 3 + 4 should ship together for each atomized cluster. Phases 5/6 can run in parallel with 3/4 once 1 lands. Phase 7 closes the loop.

---

## What I'll do per phase

Each phase ships as 1+ commits with green tests. After each phase I'll report:
- What changed, with file:line refs
- Any FORCED/BLOCKED count-adjustment formulas added (so you can sanity-check)
- Any constants that resisted migration (with reason)
- Any atom that doesn't have full parity with its legacy combined version (with the gap explained)

I will *not* invent new behavior or new constraint ideas in this work — only re-shape existing logic into atoms. Behavioral changes get raised separately.

---

## Definition of done (whole plan)

- All atoms registered, all helper-vars in registry catalog.
- `constraints/original.py` and `constraints/ai.py` moved to `constraints/archived/`; no prod module imports them.
- `unified.py` dispatches via atoms only; `SharedVariablePool` replaced by `HelperVarRegistry`.
- `main_staged.py` reads `SOLVER_STAGES` from config; supports `--stages-config`, `--stage-only`, `--skip-stage`, `--list-stages`.
- Every constant from the punch list is in `CONSTRAINT_DEFAULTS`/`AWAY_VENUE_RULES` or derived from existing data.
- `MaitlandHomeGrouping`/`AwayAtMaitlandGrouping`/etc. renamed to `NonDefaultHomeGrouping`/`AwayAtNonDefaultGrouping`/etc., scope by `home_field_map` keys.
- FORCED/BLOCKED count adjusters registered for at least: `ClubVsClubCoincidence`, `BroadmeadowFridayCount`, `GosfordFridayCount`, `MaitlandFridayCount`, `EqualMatchUpSpacing`, `NonDefaultHomeGrouping`, `AwayAtNonDefaultGrouping`, `EqualGames`. Each with a documented formula.
- Test suite: ≥1212 passing (current baseline), every atom has solo-clean + solo-violation tests, plus `tests/fixtures/violations/` has at least one fixture per atom needing a static one.
- `DrawTester.run_violation_check()` returns `ViolationReport.breakdown` with by_club, by_type, by_severity, soft_pressure aggregations.
- `tests/test_no_legacy_imports.py` passes.
- CLAUDE.md updated; new docs `CONSTRAINT_INVENTORY.md`, `HELPER_VARS.md`, `HARNESS.md`, `STAGES.md` added.
