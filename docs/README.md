# Hockey Draw Scheduler — Documentation

All project documentation, organised by **audience and lifecycle**. Pick the category that matches who you are and what you need.

## The six categories

| Category | Audience | When to read | When to update |
|---|---|---|---|
| **`operator-human/`** | Convenor — using the system | Onboarding, running the system, looking up a rule | When a behaviour visible to the operator changes, or the rules of the competition change |
| **`operator-ai/`** | AI agents operating the system | Every AI session (after `CLAUDE.md` + `todo/GOALS.md`) | When a CLI flag, config key, file path, or operational procedure changes |
| **`system/`** | Engineers extending or debugging | Before any non-trivial code change; when looking up atom / helper-var detail | **Every time engineering behaviour changes** — same commit |
| **`reports/`** | Convenor + stakeholders (clubs, league) | Reviewing what a given season looked like | Once per draw publication cycle. Reports are snapshots, not living docs |
| **`seasonal/`** | Convenor (+ helping AI) for *this* season | While building / publishing the season | While the season is live; freeze when published |
| **`todo/`** | Anyone about to implement system work | Before starting an engineering unit | Continuously — every plan moves `not_ready → ready → in_progress → done` |

Each category has its own `README.md` with the full contents list and conventions. Read that before adding files to a category.

## At-a-glance file tree

```
docs/
  README.md                         ← you are here
  operator-human/
    README.md
    USER_GUIDE.md                   how to install, configure, run
    CAPABILITIES.md                 what the system can do
    RULES.md                        the rules, plain English
    PERENNIAL_RULES.md              standing rules across seasons
  operator-ai/
    README.md
    AI_OPERATIONS_MANUAL.md         master AI-operator reference
    SYSTEM_OPERATION.md             running, monitoring, troubleshooting
    SEASON_SETUP.md                 spinning up a new season
    CONFIGURATION_REFERENCE.md      every config parameter
    CONSTRAINT_APPLICATION.md       FORCED / BLOCKED / AWAY_VENUE_RULES etc
    GAME_TIME_DICTIONARIES.md       PHL/2nd time-slot filtering
  system/
    README.md
    SYSTEM_OVERVIEW.md              architecture
    CONSTRAINT_INVENTORY.md         the atom registry (SSoT, per-atom detail)
    HARNESS.md                      end-to-end pipeline reference
    STAGES.md                       SOLVER_STAGES config + CLI flags
    HELPER_VARS.md                  HelperVarRegistry API
    COUNT_ADJUSTERS.md              FORCED/BLOCKED count adjuster formulas
    FORCED_GAMES_AS_COUNT_RULES.md  why per-venue Friday counts → FORCED
  reports/
    README.md
    {year}/                         per-season report bundles (snapshots)
  seasonal/
    README.md
    {year}/
      convenor_notes.md             working scratchpad
      club_contacts.md              who-to-call list
      operational_TODO.md           in-flight ops items for this season
  todo/
    README.md
    GOALS.md                        product + engineering goals + specifications
    spec-*.md                       individual implementation plans (status-tracked)
    done/                           completed plans (archival)
```

## Where do I add a new doc?

| Question | Answer |
|---|---|
| Is it about a *rule* convenors care about? | `operator-human/RULES.md` (high-level) or `system/CONSTRAINT_INVENTORY.md` (technical) |
| Is it a CLI / config reference? | `operator-ai/` |
| Is it engineering caveats / atom internals? | `system/` |
| Is it for *this* season only? | `seasonal/{year}/` |
| Is it a published report / snapshot? | `reports/{year}/` |
| Is it a plan for work to be done? | `todo/spec-*.md` |
| Is it the long-horizon goal? | `todo/GOALS.md` (single file) |

When unsure: ask which category's *audience* would read it first.

## See also

- Repo-root `CLAUDE.md` — mandatory first read for AI sessions, includes category-update rules
- Repo-root `README.md` — quick-start for new humans
