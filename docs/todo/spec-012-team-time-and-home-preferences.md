<!-- status: ready -->
<!-- owner: unassigned -->
<!-- depends_on: none -->

# spec-012 — Team time preferences + Maitland home/away weekend preference (verify + wire up)

**Spec source:** Convenor request 2026-05-18 (research session).

## Why

The convenor flagged two preferences as either "maybe already implemented" or "pretty sure it's implemented" — i.e. verification tasks. Make them first-class and wired in for 2026:

1. **Team time preference** — Maitland and Port Stephens both ask not to play at 08:30. The `PreferredTimes` constraint reads `PREFERENCE_NO_PLAY` (`config/season_2026.py:187` — currently empty) and supports `team_name` + `time` filters via `utils.py::normalize_preference_no_play`. The mechanism exists; the data does not.
2. **Maitland home-and-away weekend preference** — convenor expects pairs of consecutive (home, away) weekends rather than long runs of either. The current `NonDefaultHomeGrouping` atom limits *consecutive home weekends* (via `CONSTRAINT_DEFAULTS['maitland_max_consecutive_home']=1`) and `AwayAtNonDefaultGrouping` caps away-club density per weekend at Maitland. Together these approximate the H/A alternation, but they don't actively *prefer* alternation. The user is asking us to verify the existing combo is enough, OR add a soft constraint that does.

## Definition of Done

### Part A — team time preferences

1. `config/season_2026.py::PREFERENCE_NO_PLAY` populated with two entries (verify exact team names against `config/season_2026.py::TEAMS`):
   ```python
   PREFERENCE_NO_PLAY = {
       'maitland_no_8_30am': {'club': 'Maitland', 'time': '08:30',
           'description': 'Maitland teams prefer not to play at 08:30'},
       'port_no_8_30am': {'club': 'Port Stephens', 'time': '08:30',
           'description': 'Port Stephens teams prefer not to play at 08:30'},
   }
   ```
   - Whether the mechanism currently supports `'club'` as a `PREFERENCE_NO_PLAY` key needs verification. If only `team_name` works, expand the normaliser to accept `club` (mirrors the spec-001 / FORCED-GAMES `'club'` filter pattern).
2. Test `tests/test_preference_no_play_time_only.py`:
   - Fixture with 2 Maitland teams + Sunday 08:30 slots.
   - Solve and assert: penalty for Maitland teams playing at 08:30 > 0, and the solver avoids it unless structurally needed.
3. The pre-season report surfaces the new preferences.

### Part B — Maitland home/away weekend alternation

1. **Audit first.** Read `NonDefaultHomeGrouping` (`unified.py::_maitland_grouping_hard` and the home_field_map iteration) plus `AwayAtNonDefaultGrouping`. Document in a comment block: "Together, these enforce X (consecutive cap) and Y (away density). They do NOT prefer alternation; they prevent long home runs."
2. **Decision point:** is "no >1 consecutive home weekend" sufficient for the convenor's H/A alternation intent? If yes → close Part B as VERIFIED, document in `docs/operator-human/RULES.md`. If no → ship a soft atom `MaitlandAlternateHomeAway`:
   - Per pair of consecutive playable weekends `(w, w+1)`, penalty when both are home OR both are away.
   - Soft, severity 4 (LOW), weight tunable via `PENALTY_WEIGHTS['away_club_alternate_home_away']`.
   - Reuses the `weekend_used` BoolVar already in the helper-var pool from `MaxMaitlandHomeWeekends`.
3. Tests:
   - Solve a fixture, count consecutive home weekends across the season, assert the count of "home-home" or "away-away" weekend pairs is minimised compared to a baseline run.
4. `docs/operator-human/RULES.md` — describe the H/A alternation expectation in one paragraph.

### Part C — Docs + report wiring

1. `docs/operator-ai/CONFIGURATION_REFERENCE.md` — document `PREFERENCE_NO_PLAY` entry shape including the `'club'` filter if added.
2. `docs/system/CONSTRAINT_INVENTORY.md` — `PreferredTimes` row clarified; new `MaitlandAlternateHomeAway` row if Part B Option 2 ships.
3. `analytics/preseason_report.py` — list `PREFERENCE_NO_PLAY` entries.

## Implementation units

### Unit 1 — Time preferences (Part A)

- **Files touched:** `config/season_2026.py`, possibly `utils.py::normalize_preference_no_play` (add `'club'` filter if missing), `tests/test_preference_no_play_time_only.py` (new).

### Unit 2 — Audit + decide (Part B)

- **Files touched:** none yet — produce a `docs/research/maitland_home_away_audit.md` (or comment in the existing convenor notes). Decision drives Unit 3.

### Unit 3 — Alternation atom (Part B Option 2, optional)

- **Files touched:** `constraints/atoms/maitland_alternate_home_away.py` (new), `constraints/registry.py`, `config/defaults.py::DEFAULT_STAGES` (add to `soft_optimisation`), `tests/atoms/test_maitland_alternate.py` (new).

### Unit 4 — Docs

- **Files touched:** `docs/operator-ai/CONFIGURATION_REFERENCE.md`, `docs/system/CONSTRAINT_INVENTORY.md`, `docs/operator-human/RULES.md`, `analytics/preseason_report.py`.

## Doc registry

- `docs/operator-ai/CONFIGURATION_REFERENCE.md` — PREFERENCE_NO_PLAY shape
- `docs/system/CONSTRAINT_INVENTORY.md` — PreferredTimes, possibly MaitlandAlternateHomeAway
- `docs/operator-human/RULES.md` — H/A alternation + time preferences
- `docs/todo/GOALS.md` — add spec-012 row, flip to "done"

## Out of scope

- Per-team time preferences beyond the two clubs flagged (add to season config in future, no code change needed).
- Generalising H/A alternation to all away-based clubs (Gosford has its own H/A semantics — separate plan if needed).
