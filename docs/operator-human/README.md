# Operator (Human) Documentation

**Audience:** The human running the scheduler — convenor, season setter-upper, anyone who needs to *use* the system but doesn't need to know how it works under the hood.

**Tone:** Concise. How-to. Non-technical. Rules in plain English.

**Update cadence:** When a behaviour visible to the operator changes, or when the rules of the competition change.

## Contents

| File | Purpose |
|---|---|
| `USER_GUIDE.md` | How to install, configure, and run the system. Quick start. |
| `CAPABILITIES.md` | What the system can and cannot do. High-level feature list. |
| `RULES.md` | The scheduling rules in plain English — hard rules, soft preferences, by category. *No code, no constraint class names.* |
| `PERENNIAL_RULES.md` | Standing rules that apply every season (rounds 1–2 at Broadmeadow, last-game-on-WF, etc.). |

## What does NOT live here

- Anything mentioning constraint class names, atom names, helper-var registries, or solver internals → `../system/`
- Anything an AI agent needs to *operate* the system programmatically → `../operator-ai/`
- Per-season convenor notes, club contacts, season TODOs → `../seasonal/{year}/`
