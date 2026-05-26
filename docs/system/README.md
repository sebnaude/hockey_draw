# System Documentation

**Audience:** Engineers (human or AI) extending, debugging, or designing the system. The reader is technical and wants the fine-grained truth — what each piece does, its quirks, what to know before touching it.

**Tone:** Dot-pointy, concise, caveat-rich. "This constraint can be toggled to not apply to forced games" — exactly the kind of detail that lives here.

**Update cadence:** Every time engineering behaviour changes. A new atom is registered → `CONSTRAINT_INVENTORY.md` gets a row. A helper var kind is added → `HELPER_VARS.md` documents it. A new solver stage → `STAGES.md`.

## Contents

| File | Purpose |
|---|---|
| `SYSTEM_OVERVIEW.md` | Architecture. CP-SAT, decision variables, pipeline shape. |
| `CONSTRAINT_INVENTORY.md` | **The atom registry** — single source of truth for every constraint, its severity, slack key, forced-games behaviour, helper-var consumption, atom mapping. *Engineering detail per atom lives here.* |
| `HARNESS.md` | End-to-end solver pipeline: `generate_X` → `validate_game_config` → unified engine → stages → output. |
| `STAGES.md` | `SOLVER_STAGES` config schema + validation + CLI flags. |
| `HELPER_VARS.md` | `HelperVarRegistry` API — how atoms declare and share helper BoolVars/IntVars. |
| `COUNT_ADJUSTERS.md` | FORCED/BLOCKED count adjuster formulas (Phase 4). |
| `FORCED_GAMES_AS_COUNT_RULES.md` | Why per-venue Friday count atoms were retired in favour of `FORCED_GAMES`. |
| `REGEN_CONSTRAINTS.md` | The `regen` constraint group (spec-027): core-hard set, 13 RegenSoft penalty atoms, group definition, dispatch wiring, and the engine-key design note for `EqualMatchUpSpacing`/`ClubGameSpread`. |
| `TESTING.md` | The green test suite (spec-034): batched-run + coverage runner (`scripts/run_green_suite.py`), the no-mock policy, the three real-data assurances (atoms enforce / tester detects / soft measured), the assurance→test mapping, and the honest coverage numbers. |

## The atom registry rule

When you add or change an atom, you update `CONSTRAINT_INVENTORY.md` *in the same commit*. The per-atom row must capture:

- Forced-games handling (excluded / included / n/a)
- Locked-week handling (yes / no / n/a)
- Dummy-key handling (yes / no / n/a)
- Slack key (if any)
- Helper vars consumed
- One-liner caveats

If a column doesn't apply, write `n/a` — never leave blank. A missing row is treated as a bug.

## What does NOT live here

- Plain-English rule descriptions for the convenor → `../operator-human/RULES.md`
- CLI cheat-sheets and how-tos → `../operator-ai/`
- Per-season tuning (which forced games, what locked weeks) → `../seasonal/{year}/`
