# Seasonal — Draw Creation Documentation

**Audience:** The convenor (and any AI helping the convenor) during *this season's* draw build.

**Purpose:** Working state for a single season — convenor notes, club contact lists, in-flight operational TODOs, special-weekend requirements, hand-edit records.

**Tone:** Whatever works for the convenor. Bullet points, dated entries, scratchpad-style. These are *living* documents until the season is done.

**Structure:** Year-scoped subdirectories — `seasonal/{year}/`. After a season closes, the dir stays as historical reference but is no longer updated.

## Contents

```
seasonal/
  2026/
    convenor_notes.md       — running scratchpad of decisions, asks, follow-ups
    club_contacts.md        — phone/email per club, who handles what
    operational_TODO.md     — open items that block draw creation or publication
  README.md                 — this file
```

## What goes where

- **`convenor_notes.md`** — anything the convenor jots down while building the draw. Decisions, special requests from clubs, "remember to ask X". Dated entries best.
- **`club_contacts.md`** — names + emails + phone numbers per club. Update when contacts change.
- **`operational_TODO.md`** — actionable items that need to be addressed before / during draw publication. Mark with status. Items that block solver runs go here. Items that are pure system / engineering work go in `../todo/` instead.

## Adding a new season

1. `mkdir docs/seasonal/{year}`
2. Copy `convenor_notes.md`, `operational_TODO.md` from prior season as templates (strip old content)
3. Carry forward `club_contacts.md` (update where needed)
4. Add the year to the README contents list

## What does NOT live here

- Pre-season config or compliance *reports* → `../reports/{year}/`
- Engineering implementation plans / specifications → `../todo/`
- Plain-English rule docs that apply across seasons → `../operator-human/`
