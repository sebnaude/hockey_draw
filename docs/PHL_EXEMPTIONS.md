# PHL Weekend & Block Exemptions — How They Work

How the grade **PHL** is allowed to play when other grades cannot. The work
happens in the harness that sets up the decision-variable dict (`X`), across two
passes in `utils.py`:

1. **`generate_timeslots()`** (`utils.py:259`) — builds the grade-agnostic list of
   `Timeslot`s from the season dates, `DAY_TIME_MAP`, and `FIELD_UNAVAILABILITIES`.
   This is where *whole weekends* get killed.
2. **`generate_X()`** (`utils.py:3509`) — loops over (matchup × timeslot), applies
   grade filters + `BLOCKED_GAMES`/`FORCED_GAMES`, and creates the `BoolVar`s.

There are two kinds of exemption: **structural** (real code branches PHL never
falls into) and the **weekend carve-out** (achieved by config technique, *not* by
PHL-specific code).

---

## 1. The whole-weekend block — and how PHL is exempted from it

`generate_timeslots` applies a per-field **whole-weekend hammer**:

```python
# utils.py:315-317  — inside the per-field loop
if any(current_date.date() in [(w - timedelta(days=1)).date(), w.date(), (w + timedelta(days=1)).date()]
    for w in field_unavailabilities.get(field_name, {}).get('weekends', [])):
    continue          # skips Fri, Sat AND Sun for that field
```

A date in a field's `'weekends'` list blocks **Friday + Saturday + Sunday** for
**every grade** — there is no grade dimension at this stage. Paired effect: if
NIHC's whole weekend is out, `round_no` is **not** incremented
(`utils.py:304-309`), so a fully-blocked weekend doesn't even count as a playable
round.

**PHL is not exempted by any code branch.** It is exempted by a deliberate config
technique, visible in the comments of `FIELD_UNAVAILABILITIES`
(`config/season_2026.py:118-174`). For the two state-championship weekends
(Masters SC May 15-17, U16 Girls SC Jun 19-21) the convenor **downgrades** the
block:

- The weekend is **removed from `'weekends'`** (the whole-weekend hammer) — see the
  commented-out lines `# May 15-17 ... moved to whole_days, Sunday open for PHL`
  (`:123, :141, :159`).
- Instead, only **Friday + Saturday** are added to `'whole_days'` (`:129-130`,
  `:147-148`), which blocks just those specific dates (`utils.py:326-327`).
- **The Sunday is left untouched** → `generate_timeslots` produces Sunday
  timeslots for that week, and `round_no` increments, so it is a real round.

That alone would open the Sunday for *all* grades. A **second layer** re-closes it
for everyone except PHL, in `BLOCKED_GAMES` (`config/season_2026.py:399-443`):

```python
# State Championship Sundays — PHL only. One entry per non-PHL grade, no team/club
# → blocks ALL vars matching (grade + date + location).
{'grade': '2nd', 'date': '2026-05-17', 'field_location': 'Maitland Park', ...},
{'grade': '3rd', 'date': '2026-05-17', ...},   # 4th, 5th, 6th too
# ...repeated for NIHC and for 2026-06-21
```

`generate_X` applies these as grade-scoped eliminations. **PHL has no such entry**,
so PHL's Sunday variables survive. Net effect: on May 17 and Jun 21, only PHL plays
at NIHC and Maitland Park.

> **The pattern:** "PHL is exempt from the blocked weekend" =
> (a) don't use the whole-weekend block — block only Fri/Sat as `whole_days`;
> (b) re-block the now-open Sunday for grades 2nd–6th via `BLOCKED_GAMES`.
> PHL falls through both.

**Gosford does it on the opposite day.** Central Coast keeps its Sundays blocked
(`whole_days` May 17 / Jun 21, `:167, :169`) but leaves the **Friday** open
(`# Jun 19 / May 15 Friday OPEN for forced PHL 8pm game`, `:170-171`) so a forced
PHL Friday game can land there.

---

## 2. Structural exemptions (real code branches in `generate_X`)

These let PHL play days/venues that lower grades are hard-filtered out of.

**PHL-only days (Friday) and venues (Gosford)** — `utils.py:3658-3661, 3710-3720`:

```python
phl_only_venues = {'Central Coast Hockey Park'}   # Gosford
phl_only_days   = {'Friday'}
...
else:   # only reached by grades that are NOT PHL and NOT 2nd
    if t.field.location in phl_only_venues:  continue   # lower grades skip Gosford
    if t.day in phl_only_days:               continue   # lower grades skip Friday
```

PHL (and 2nd) take the earlier `if is_phl:` / `elif is_second:` branches
(`:3688, :3704`) and **never reach** this exclusion block — that is the exemption.
The same `grade not in ('PHL','2nd')` guard is mirrored in the validators so they
stay consistent: `validate_game_config` (`utils.py:1131-1136`) and
`_check_forced_game_feasibility` (`utils.py:3041-3047`).

**PHL plays by whitelist, not by the global calendar.** PHL variables are kept
*only* if the slot is in `phl_valid_slots`, built from `PHL_GAME_TIMES`
(`utils.py:3566-3586, 3687-3702`). This is what lets PHL have Friday-night slots
that don't exist for anyone else — `PHL_GAME_TIMES` (`config/season_2026.py:52-78`)
defines EF Friday 7pm at NIHC, Gosford Friday 8pm, Maitland Friday 7pm.

---

## 3. The `FORCED_GAMES`-over-perennial override (spec-001)

Not PHL-specific, but it is the third "exemption from a block" in the same harness
and interacts with the above. A variable hit by an **exclusively perennial**
`BLOCKED` scope (e.g. the rounds-1-2 Broadmeadow-only rule, `'perennial': True`) is
**kept** if a `FORCED_GAMES` entry also matches it (`utils.py:3739-3778`). A
non-perennial (season-specific) block always wins. This is how the forced
Gosford/Maitland Friday games survive the surrounding Friday blocks. Pinned by
`tests/test_perennial_blocked_forced_exemption.py`.

---

## Summary

| What PHL is exempt from | How (mechanism) | Location |
|---|---|---|
| Whole-weekend `FIELD_UNAVAILABILITIES` block (championship weekends) | Config downgrades `weekends`→`whole_days` (Fri/Sat only), leaving Sunday slots generated | `config/season_2026.py:118-174` + `utils.py:315-327` |
| The now-open championship Sunday (vs other grades) | Per-grade `BLOCKED_GAMES` entries for 2nd–6th only; PHL has none | `config/season_2026.py:399-443` |
| Friday-night exclusion | `phl_only_days` check sits only on the non-PHL/2nd branch | `utils.py:3661, 3717` |
| Gosford (Central Coast) venue exclusion | `phl_only_venues` check sits only on the non-PHL/2nd branch | `utils.py:3660, 3714` |
| The global calendar generally | PHL kept by `PHL_GAME_TIMES` whitelist, not by `DAY_TIME_MAP` | `utils.py:3566-3586, 3687-3702` |
| Perennial (default) blocks | Matching `FORCED_GAMES` scope overrides perennial-only block | `utils.py:3739-3778` |

**Headline:** there is no single `if grade == 'PHL': allow` line for weekends. The
weekend exemption is an emergent two-step config pattern — *don't block the weekend
wholesale, then block it back for every grade except PHL* — while the
Friday/Gosford exemptions are genuine code branches that PHL simply never falls
into.
