# Reports

**Audience:** Convenor + downstream stakeholders (clubs, the league office).

**Purpose:** Per-season report documents, including:

- Input requirements gathered from emails / meetings before drafting the draw
- Pre-season configuration reports (output of `run.py preseason`)
- Compliance certificates (constraint pass/fail summaries after solver runs)
- Post-publication recap docs

**Structure:** Year-scoped subdirectories — `reports/{year}/`.

**Update cadence:** Once per draw publication cycle. Reports are *snapshots*, not living docs — once finalised, they're immutable.

## Contents (current)

| Path | Purpose |
|---|---|
| `README.md` | This file. |
| *(none yet — populate per season)* | |

## Suggested structure for `reports/{year}/`

- `inputs_collected.md` — what each club asked for, source email/meeting, date received
- `preseason_report.md` — output of the preseason skill, captured at config-lock time
- `compliance_v{X}.md` — per-version solver compliance report
- `convenor_signoff.md` — the convenor's "ready to publish" note

## What does NOT live here

- Working notes during draw creation → `../seasonal/{year}/convenor_notes.md`
- Operational TODOs for *this* season → `../seasonal/{year}/operational_TODO.md`
- System-level engineering reports → `../system/`
