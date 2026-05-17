# Operator (AI) Documentation

**Audience:** An AI agent (Claude, sub-agent, automated runner) that needs to operate the scheduler — generate draws, test, diagnose, write config, apply hand-edits.

**Tone:** Procedural. Imperative. Assumes the reader is capable but has no project context.

**Update cadence:** When a CLI flag, config key, file path, or operational procedure changes.

## Contents

| File | Purpose |
|---|---|
| `AI_OPERATIONS_MANUAL.md` | Master reference for an AI driving the system end-to-end. |
| `SYSTEM_OPERATION.md` | Running the solver, monitoring it, troubleshooting. |
| `SEASON_SETUP.md` | Spinning up a new season's config. |
| `CONFIGURATION_REFERENCE.md` | Every config parameter, its default, where it lives. |
| `CONSTRAINT_APPLICATION.md` | How to apply restrictions (FORCED, BLOCKED, AWAY_VENUE_RULES, etc.). |
| `GAME_TIME_DICTIONARIES.md` | `PHL_GAME_TIMES` / `SECOND_GRADE_TIMES` filtering rules. |
| `README.md` | This file. |

## What does NOT live here

- The *why* behind a constraint, or its registry detail / caveats → `../system/`
- Plain-English rule descriptions → `../operator-human/RULES.md`
- This-season's working state (notes, contacts, ops TODOs) → `../seasonal/{year}/`

## Reading order for a fresh AI session

1. Root `CLAUDE.md` (mandatory)
2. `docs/todo/GOALS.md` (the *why*)
3. `AI_OPERATIONS_MANUAL.md` (this dir) — operational reference
4. Then drop into specifics as needed (CONFIGURATION_REFERENCE, CONSTRAINT_APPLICATION, etc.)

## Update imperative

When you learn something new about how the system works — a config requirement, a non-obvious rule, a workaround — add it to the relevant file in this directory so the next AI session has it. This is non-negotiable.
