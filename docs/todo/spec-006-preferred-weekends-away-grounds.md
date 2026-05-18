<!-- status: in_progress -->
<!-- owner: session=spec-006-agent claimed=2026-05-18T00:00:00Z -->
<!-- depends_on: none -->

# spec-006 — Preferred / non-preferred weekends for away grounds

**Spec source:** [`docs/todo/GOALS.md` → spec-006](GOALS.md#spec-006--preferred--non-preferred-weekends-for-away-grounds)

## Why

NRL-Knights-at-Maitland weekends are tracked in `docs/seasonal/2026/operational_TODO.md` as "Need to verify where/how this is enforced." Answer: they aren't — convenor reviews manually. Some convenor preferences are positive (Knights' home weekend → prefer extra hockey games AT Maitland Park to ride the traffic) and some negative (NRL Knights home game → prefer NO hockey at Maitland Park that weekend). A first-class soft constraint covers both with the same data structure.

## Definition of Done

1. New config list `PREFERRED_WEEKENDS` in season config (or extension of existing FORCED/BLOCKED scaffold if reusable — confirm by reading the scaffold before deciding). Entry shape:
   ```python
   {
     'date': '2026-04-05',                  # or 'dates': [...]
     'field_location': 'Maitland Park',
     'field_name': 'Main',                  # optional — venue-level if omitted
     'mode': 'prefer' | 'avoid',
     'weight': 1000,                        # optional, default from PENALTY_WEIGHTS
     'description': 'NRL Knights vs Raiders at Maitland',
   }
   ```
2. New atom `PreferredWeekendsAwayGround` reading `PREFERRED_WEEKENDS`:
   - For each `prefer` entry: penalty = `weight × (target_game_count - actual_games_at_venue_on_date)` clamped at 0.
   - For each `avoid` entry: penalty = `weight × actual_games_at_venue_on_date`.
   - Soft, severity 5, never blocks feasibility.
3. Registered + wired into `soft_optimisation` stage.
4. Default `PENALTY_WEIGHTS['preferred_weekends_away_ground'] = 1000` — tunable.
5. Unit tests:
   - Given an `avoid` entry for Maitland Park on 2026-04-05, When solving, Then no games scheduled at Maitland Park on that date if feasibly avoidable.
   - Given a `prefer` entry, When solving, Then games at the venue on that date are preferred (more than equivalent dates without the entry, all else equal).
   - Given conflicting preferences (`prefer` and `avoid` on same date), When solving, Then no crash; penalties accumulate as written.
6. 2026 NRL Knights dates from `operational_TODO.md` migrated into `config/season_2026.py::PREFERRED_WEEKENDS` as `avoid` entries — and the corresponding line in `operational_TODO.md` marked done.
7. `docs/system/CONSTRAINT_INVENTORY.md` row.
8. `docs/operator-ai/CONFIGURATION_REFERENCE.md` documents the new config list shape.
9. `docs/operator-human/RULES.md` mentions preferred/avoided weekends per venue.

## Implementation units

### Unit 1 — Scaffold check + decision

- **Read first:** `config/defaults.py::FORCED_GAMES` and `BLOCKED_GAMES` shape, `utils.py` matchers. The user notes "scaffold should already exist." Confirm whether `FORCED_GAMES` already supports `soft: true` + `weight` semantics (look at `constraints/atoms/_adjusters.py` and the FORCED_GAMES iteration in `generate_X`).
- **Decide:** extend FORCED_GAMES with `soft: true` mode OR create a new `PREFERRED_WEEKENDS` list. Document the decision at the top of Unit 2's atom file.

### Unit 2 — Atom + config

- **Files touched:** `constraints/atoms/preferred_weekends_away_ground.py` (new), `constraints/registry.py`, `config/defaults.py` (PENALTY_WEIGHTS + maybe PREFERRED_WEEKENDS), `config/season_2026.py::PREFERRED_WEEKENDS` (migration of 2026 dates).

### Unit 3 — Docs + ops TODO

- **Files touched:** `docs/system/CONSTRAINT_INVENTORY.md`, `docs/operator-ai/CONFIGURATION_REFERENCE.md`, `docs/operator-human/RULES.md`, `docs/seasonal/2026/operational_TODO.md` (mark the Maitland no-play dates section as DONE).

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md`
- `docs/operator-ai/CONFIGURATION_REFERENCE.md`
- `docs/operator-human/RULES.md`
- `docs/seasonal/2026/operational_TODO.md` — mark resolved
- `docs/todo/GOALS.md` — flip spec-006 status to "done"

## Out of scope

- Per-team preferences (team X prefers to play on date Y) — that's `PreferredTimesConstraint` (existing).
- Hard "must not play at venue on date" — that's BLOCKED_GAMES (existing).
