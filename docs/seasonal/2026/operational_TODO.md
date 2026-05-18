# TODO

## HIGHLY IMPORTANT: Week 1 Draw Modified Externally

Week 1 draw was changed by John Mayers and is not currently reflected in this system.

**Status:** Need to obtain the updated Week 1 draw and import it into the system.

---

## Maitland No-Play Dates (Newcastle Knights NRL Home Games)

Maitland doesn't want to play on these dates due to Newcastle Knights home games at Maitland:

- Sunday, 5 April 2026 – Knights vs Raiders
- Sunday, 26 April 2026 – Knights vs Panthers
- Sunday, 3 May 2026 – Knights vs Rabbitohs
- Sunday, 28 June 2026 – Knights vs Wests Tigers
- Sunday, 5 July 2026 – Knights vs Dolphins
- Sunday, 16 August 2026 – Knights vs Titans

**Status:** [DONE: implemented in spec-006] — All 6 dates are now migrated to
`config/season_2026.py::PREFERRED_WEEKENDS` as `'mode': 'avoid'` entries for
`'field_location': 'Maitland Park'`. The `PreferredWeekendsAwayGround` soft atom
(severity 5, `soft_optimisation` stage) applies a penalty of
`PENALTY_WEIGHTS['preferred_weekends_away_ground']` (default 1000) for each game
scheduled at Maitland Park on these dates. This is a **soft** constraint — the
solver avoids these dates when feasibly possible but will schedule there if no
other feasible assignment exists. For a hard block, use `BLOCKED_GAMES` instead.

## Norths v Wests Weekend – 12-14 June 2026

Requirements from Norths:

- **Friday 12th June (Friday Night):** PHL – Norths v Wests
- **Saturday 13th June:** Balance of grades Norths v Wests play, with the last game finishing around 4:30pm
- Any Norths grades that can have byes that weekend should have byes
- Any Norths v Wests games not played by end of day Saturday are played in the afternoon of **Sunday 14th June**

**Status:** Not yet implemented.
