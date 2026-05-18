<!-- status: done -->
<!-- owner: session=spec-009-agent claimed=2026-05-18T00:00:00Z -->
<!-- completed: 2026-05-18 -->
<!-- depends_on: none -->

# spec-009 — FORCED count rules + adjusters — end-to-end verification

**Spec source:** Convenor request 2026-05-18 (research session).

## Why

Three independent count-rule behaviours have all been *shipped* on `final-form` (`docs/system/FORCED_GAMES_AS_COUNT_RULES.md`, `docs/system/COUNT_ADJUSTERS.md`), but none have been re-verified end-to-end since the surrounding atomisation. The convenor wants high confidence before next solver run that:

1. A FORCED entry expressing a *total count* (e.g. "exactly 8 Gosford Friday games per season") interacts correctly with per-pair FORCED entries that scope into it (e.g. "1 of those is Gosford-vs-Maitland on round 4"). Multi-scope registration (`utils.py::_get_matching_forced_scopes`) must register one variable against **all** matching scopes.
2. PHL Friday-night forced games are exempted from `ClubVsClubCoincidence`'s Sunday-meetings count via the `club_vs_club_coincidence_adjuster` (Adjuster #5). I.e. the Sunday-alignment cluster does not expect a Sunday meeting for a club-pair whose PHL meeting was forced onto Friday.
3. "Double-ups in grade" (a club fielding two teams in the same grade) are handled correctly across the count machinery — `ClubVsClubCoincidence` treats each matchup as a distinct contribution, and `ClubDay` / `ClubVsClubFieldLimit` don't silently collapse them.

These are **verification** outcomes, not implementation. If any check fails, that becomes the bug to fix — but the baseline expectation is "shipped, prove it."

## Definition of Done

### Check 1 — FORCED total + per-pair stacking

1. New test `tests/test_forced_total_plus_per_pair.py` (or extend `tests/test_forced_games_multi_scope.py`) that:
   - Builds a small fixture (2 PHL clubs each with 1 team, 10 weeks with Friday slots at Gosford).
   - Populates `FORCED_GAMES` with `{grade:'PHL', day:'Friday', field_location:'Central Coast Hockey Park', count:8, constraint:'equal'}` PLUS `{teams:['Gosford PHL','Maitland PHL'], grade:'PHL', day:'Friday', field_location:'Central Coast Hockey Park', count:1, constraint:'equal'}`.
   - Solves and asserts exactly 8 Gosford Friday games AND exactly 1 of them is Gosford-vs-Maitland.
   - Asserts a third FORCED entry `{teams:['Gosford PHL','Maitland PHL'], grade:'PHL', day:'Friday', count:1, constraint:'equal'}` (no field_location) does NOT inflate the Gosford total to 9 (the per-pair forced game is shared between both scopes).
2. If the assertion fails, file a bug commit immediately — do not edit the test to match the buggy behaviour.

### Check 2 — PHL Friday exempted from ClubVsClubCoincidence

1. New test `tests/atoms/test_cvc_coincidence_phl_friday_adjuster.py`:
   - Fixture: club pair (Maitland, Norths) with PHL meetings = 4, 3rd grade meetings = 3.
   - FORCED: 2 PHL meetings of that pair on Friday at Gosford.
   - Run `club_vs_club_coincidence_adjuster` directly → assert it returns `{'PHL': {('Maitland','Norths'): 2}}`.
   - Run the atom and inspect `data['count_adjustments']['ClubVsClubCoincidence']` → assert atom uses `expected = 2` not `expected = 4` for the PHL row of the alignment.
2. Mirror in the tester: `DrawTester` must not flag a PHL Sunday-coincidence miss when the missing meeting is accounted for in FORCED Fridays.

### Check 3 — Same-grade-same-club double-ups

1. New test `tests/atoms/test_double_up_handling.py`:
   - Fixture: club A fields 2 teams in grade `2nd`. Club B has 1 team.
   - `EqualGamesAndBalanceMatchUps` schedules `2 * meetings_per_pair` games of A-vs-B at the 2nd-grade level.
   - `SameGradeSameClubNoConcurrency` prevents the two A teams playing in the same slot.
   - `ClubDayParticipation` / `ClubVsClubCoincidence` count *distinct matchups* (each A team's meeting with B's team), not *clubs*.
2. Assertions: solve a small instance, then inspect:
   - The two A vs B games are in different slots.
   - The coincidence count for (A, B) in 2nd grade equals 2 per "weekend" they both play (one per matchup), not 1.

### Check 4 — Documentation closure

1. `docs/system/FORCED_GAMES_AS_COUNT_RULES.md` — verification status table appended with results, dated.
2. `docs/system/COUNT_ADJUSTERS.md` — Adjuster #5 row gets a "verified end-to-end (test ref)" note.
3. `CLAUDE.md` — if any bug surfaced and was fixed, the resulting change is noted in the "Common Pitfalls" or "Critical Rules" section.

## Implementation units

### Unit 1 — Check 1 test + bug fixes if any

- **Files touched:** `tests/test_forced_total_plus_per_pair.py` (new), possibly `utils.py::_get_matching_forced_scopes`.

### Unit 2 — Check 2 test

- **Files touched:** `tests/atoms/test_cvc_coincidence_phl_friday_adjuster.py` (new), possibly `constraints/atoms/club_vs_club_coincidence.py`.

### Unit 3 — Check 3 test

- **Files touched:** `tests/atoms/test_double_up_handling.py` (new). Reuse fixtures from `tests/atoms/conftest.py`.

### Unit 4 — Docs

- **Files touched:** `docs/system/FORCED_GAMES_AS_COUNT_RULES.md`, `docs/system/COUNT_ADJUSTERS.md`, `CLAUDE.md` (only if bugs found).

## Doc registry

- `docs/system/FORCED_GAMES_AS_COUNT_RULES.md` — verification status appended
- `docs/system/COUNT_ADJUSTERS.md` — Adjuster #5 verified-note
- `docs/todo/GOALS.md` — add spec-009 row, flip to "done"

## Out of scope

- Reworking the FORCED multi-scope mechanism — that's done. Plan is to verify, not redesign.
- Generalising adjuster #5 to non-Friday non-Sunday days — file a separate plan if needed.
