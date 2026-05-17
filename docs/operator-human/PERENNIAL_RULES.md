# Perennial Rules

Rules that apply **every season**, not just a single year. These are standing decisions that should be carried forward when setting up a new season config.

---

## Venue Rules

### 1. All games at Broadmeadow for first two playing rounds

All games in rounds 1 and 2 must be scheduled at Newcastle International Hockey Centre (Broadmeadow). No games at Maitland Park or Central Coast Hockey Park in the opening two rounds.

**Enforcement:** BLOCKED_GAMES entries in the season config blocking Maitland Park and Central Coast Hockey Park for `round_no` 1 and 2. These are included in `config/defaults.py` and should be copied into each new season's BLOCKED_GAMES.

**Rationale:** Ensures all teams start the season at the central venue for visibility, consistency, and operational simplicity.

---

### 2. Last game of day on West Field at Broadmeadow (single-field timeslot)

If only one field is being used for the last timeslot of a day at Broadmeadow (NIHC), that game should be on West Field (WF), not East Field (EF).

**Enforcement:** Constraint (to be implemented). Flag during draw review if not satisfied.

**Rationale:** Operational preference for West Field when only one game is running at end of day.

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
