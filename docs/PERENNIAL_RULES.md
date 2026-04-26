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

## How to apply

When creating a new season config:

1. Copy `config/season_template.py` to `config/season_YYYY.py`
2. Import perennial BLOCKED_GAMES from `config/defaults.py` and extend with season-specific entries
3. Review this document for any rules that need manual attention (e.g., constraint-enforced rules that should be checked during draw review)
