# Perennial Rules

Rules that apply **every season**, not just a single year. These are standing decisions that should be carried forward when setting up a new season config.

---

## Venue Rules

### 1. All games at Broadmeadow for first two playing rounds

All games in rounds 1 and 2 must be scheduled at Newcastle International Hockey Centre (Broadmeadow). No games at Maitland Park or Central Coast Hockey Park in the opening two rounds.

**Enforcement:** BLOCKED_GAMES entries in the season config blocking Maitland Park and Central Coast Hockey Park for `round_no` 1 and 2. These are included in `config/defaults.py` and should be copied into each new season's BLOCKED_GAMES.

**Rationale:** Ensures all teams start the season at the central venue for visibility, consistency, and operational simplicity.

**Overridable by FORCED_GAMES (spec-001).** Each `PERENNIAL_BLOCKED_GAMES` entry carries `'perennial': True`, which marks it as a *default* the convenor can override. A variable matched by a perennial BLOCKED scope is **kept** if any `FORCED_GAMES` entry also matches it — so a deliberate opening-round Maitland-vs-Norths game at Maitland Park can be forced without commenting out the perennial rule. Example:

```python
# season_2026.py
FORCED_GAMES = [
    {'teams': ['Maitland', 'Norths'], 'grade': 'PHL',
     'date': '2026-03-22', 'field_location': 'Maitland Park',
     'count': 1, 'constraint': 'equal',
     'description': 'Opening round special — Maitland-vs-Norths at MP'},
]
# Perennial BLOCKED still removes every other round 1-2 Maitland Park var.
```

Season-specific (non-perennial) BLOCKED entries are NOT overridable — they always eliminate even when FORCED also matches.

**Override via FORCED_GAMES applies to ALL grades (spec-010).** The grade-agnostic implementation means a 2nd-grade (or any-grade) forced opening-round game at Maitland Park is overridden by the same mechanism as PHL. Example: forcing a 2nd-grade Maitland-vs-Norths game at Maitland Park in round 1 is expressed identically to the PHL case — set `grade: '2nd'` in the FORCED entry.

---

### 2. NIHC field-fill order: WF before EF before SF

At Newcastle International Hockey Centre (Broadmeadow), within any single (date, day_slot) bucket the fields must be filled in priority order: **West Field → East Field → South Field**. Concretely, for every (date, day_slot) where multiple NIHC fields are valid slot options:

- If East Field has a game, West Field must also have a game.
- If South Field has a game, East Field must also have a game.
- The two together transitively imply: if South Field has a game, West Field must also have a game.

This generalises the historical "last game of the day on West Field" preference — that rule was a special case of field-fill ordering. The general form gives correct behaviour for *every* slot of the day, not only the last one (e.g. a midday slot with only one game also lands on WF rather than EF).

**Enforcement:** Hard constraint — two CRITICAL atoms `NIHCFillWFBeforeEF` and `NIHCFillEFBeforeSF` in `constraints/atoms/` (spec-003). Both atoms are wired into the `critical_feasibility` stage in `DEFAULT_STAGES` so they apply on every solver run. A matching tester check (`_check_nihc_fill_wf_before_ef` + `_check_nihc_fill_ef_before_sf`) catches the same violations on already-published draws.

**Edge case:** if WF (or EF) isn't a valid slot for a given (date, day_slot) — e.g. a field unavailability that day — the atoms skip the bucket rather than assert an impossible implication. The detection key is "does any decision variable exist for that field at that slot?" — if not, the field wasn't an option and the rule doesn't fire.

**Rationale:** Operational preference for West Field over East over South; ground staff have a fixed walk-up routine; spectators learn where the headline games live. Generalising removes the historical manual-review footnote.

---

## Count-Budget Rules

### 3. Per-venue / per-day game-count budgets are FORCED_GAMES entries

Count budgets — "max N PHL Fridays at Broadmeadow per season," "exactly M Friday games at Gosford," "exactly K Friday Maitland games" — belong in season config as `FORCED_GAMES` entries, **not** as hardcoded constraint atoms.

**Pattern:**
```python
FORCED_GAMES = [
    {'grade': 'PHL', 'day': 'Friday',
     'field_location': 'Newcastle International Hockey Centre',
     'count': 3, 'constraint': 'lesse',
     'description': 'Max 3 PHL Friday games at Broadmeadow per season'},
    {'grade': 'PHL', 'day': 'Friday',
     'field_location': 'Central Coast Hockey Park',
     'count': 8, 'constraint': 'equal',
     'description': 'Exactly 8 PHL Friday games at Gosford per season'},
]
```

**Why:** FORCED entries are discoverable in season config, reportable in draw metadata, composable with finer-grained pair rules via the multi-scope match (commit `cd8a338`), validatable by pre-solver consistency checks, and trivially per-season tweakable. Hardcoded count atoms are none of those things.

**Enforcement:** `utils._build_forced_game_rules` + the per-scope sum constraint added in `generate_X`. See `docs/FORCED_GAMES_AS_COUNT_RULES.md` for the full rationale.

**When NOT to use:** Reserve constraint classes for *structural* rules (no-double-booking, adjacency, balance, spacing). Counts are budgets — different mechanism.

---

## How to apply

When creating a new season config:

1. Copy `config/season_template.py` to `config/season_YYYY.py`
2. Import perennial BLOCKED_GAMES from `config/defaults.py` and extend with season-specific entries
3. Add per-venue Friday count budgets as FORCED_GAMES entries (see Rule 3 above)
4. Review this document for any rules that need manual attention (e.g., constraint-enforced rules that should be checked during draw review)
