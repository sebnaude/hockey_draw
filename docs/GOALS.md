# Project Goals

The north star for the hockey draw scheduler. Read this before any non-trivial change so the work fits the shape we're building toward, not just the immediate ask.

This is the *why*. For the *how*, see the doc map at the bottom.

---

## 1. Product goal

**Generate a publishable hockey season draw that the convenor can hand to clubs with confidence, in one solver run, with every constraint either satisfied or visibly tracked as a known soft penalty.**

Concretely:
- ~47 teams across 6 grades (PHL, 2nd–6th), 10 clubs, multiple venues.
- 22 Sunday rounds + 2 Friday-only PHL weeks, across a 27-week calendar window.
- Output: a versioned draw (JSON + xlsx + per-club + Revo CSV) that the convenor publishes externally.
- Hand-edits after publish are first-class: every change goes through `scripts/test_scenario.py` before promotion, and bumps the MINOR version. Solver re-runs bump MAJOR.

Anything that does not directly serve "publishable draw + auditable changes" is a side quest.

---

## 2. Engineering goal — the atomization model

The codebase is moving from **monolithic "do-everything" constraint classes** to a **registry of atomic constraints, dispatched from a config-driven stage plan**. As of `final-form` this is **shipped**; the goal now is *protecting it* and *extending it the same way*.

### What "atomization" means here

Every rule in the schedule (e.g. "PHL and 2nd grade games at Broadmeadow run back-to-back on the same field") is decomposed into the smallest independently-meaningful unit. Each unit is:

- A **named atom** in `constraints/atoms/` — one file per atom, one purpose per atom.
- **Registered** in `constraints/registry.py` with metadata (severity, slack key, helper-var declarations, the legacy class it descended from).
- **Dispatched** from a **`SOLVER_STAGES`** list in `config/defaults.py` (or a season override / `--stages-config` JSON), via `constraints/stages.py::apply_solver_stage`.
- **Tested in isolation** in `tests/atoms/` against a small CP-SAT fixture, plus a parity test that pins behaviour against the pre-atomization implementation.
- **Visible in the violation report** with per-club / per-type / soft-pressure breakdowns when violated.

Helper variables (coincidence BoolVars, count-summing IntVars) are declared via `HelperVarRegistry` so two atoms asking for the same helper share one variable — no hand-rolled duplicates.

### Why this matters

The legacy `constraints/original.py` and `constraints/ai.py` (now archived) had ~1700 and ~2000 lines of constraints that did 3–5 things each. That made it impossible to:

- **Skip one piece of a rule** without skipping the whole rule.
- **Diagnose infeasibility** beyond "it's somewhere in this 200-line method."
- **Slack a single dimension** without writing a hacked-in conditional.
- **Test a rule** without booting a full season fixture.
- **Reuse helper variables** across rules — every constraint built its own.

Atomization made each of these one-line operations: skip an atom by name, drop it from a stage, slack via its registry key, test it standalone, share helpers via the registry. The whole point of the refactor.

### Status (as of 2026-04-29)

✅ All phases shipped — see `docs/ATOMIZATION_PLAN.md` for the phase table and `docs/ATOMIZATION_HANDOFF.md` for the commit-by-commit chain. Test bar: **1287 passed, 1 skipped**.

Three monolithic constraints have been atomized so far (14 atoms across 3 clusters):

| Cluster | Atoms | Source class (archived) |
|---|---|---|
| PHLAndSecondGradeTimes | 5 | `original.py::PHLAndSecondGradeTimes` |
| ClubDayConstraint | 5 | `original.py::ClubDayConstraint` |
| ClubVsClubAlignment | 4 | `original.py::ClubVsClubAlignment` |

Plus FORCED/BLOCKED count adjusters (Phase 4), generic non-default-home rules (Phase 6), and config-driven solver stages with CLI flags (Phase 7b).

### How we handle atomization going forward

**Default rule: every new behaviour is an atom.** No new code goes into `original.py` / `ai.py` — they're in `constraints/archived/` and a test (`tests/test_no_legacy_imports.py`) actively blocks production imports of them.

When you need to add or change a constraint:

1. **Identify the smallest unit.** If the rule has two clauses you could imagine slacking independently, that's two atoms.
2. **Write the atom** in `constraints/atoms/<descriptive_name>.py`. Subclass `Constraint` (from `constraints.atoms.base`). One `apply(model, X, data)` method. Skip dummy keys (`len(key) < 11`) and locked weeks (`key[6] <= data.get('current_week', 0)`).
3. **Declare helpers** via `data['helper_registry']` (a `HelperVarRegistry`), not by creating BoolVars/IntVars directly. See `docs/HELPER_VARS.md`.
4. **Register** in `constraints/registry.py` with: canonical name, severity, slack key (if applicable), helper-var catalog entries it consumes.
5. **Wire into a stage** in `config/defaults.py::DEFAULT_STAGES` (or whatever stage is appropriate). Use the canonical name.
6. **Test in isolation** in `tests/atoms/test_<name>.py` against a tiny CP-SAT fixture. Add a parity test if there's a legacy implementation to match.
7. **Update docs**: `CONSTRAINT_INVENTORY.md` (the SSoT table), `STAGES.md` if you added a stage, and this file if the goal shifts.

**Atomizing a remaining monolith** (anything in `constraints/archived/` that the registry still aliases to a single legacy class) follows the same pattern as Phases 3a/3b/3c: read the original code, list the independent behaviours, write one atom each, add a parity test against the original, wire into stages, then mark the legacy entry as "atomized" in `CONSTRAINT_INVENTORY.md`.

**Never re-monolithize.** If two atoms have shared logic, extract a helper (see `constraints/atoms/_club_day_shared.py` for the pattern), don't merge the atoms. The merge always feels right in the moment and is always regretted three months later.

---

## 3. Operational goals

The system is run by one convenor (you), with occasional second-opinion solver runs from AI agents. So:

- **Configs over code.** Anything a convenor might tune season-to-season (stages, slack, forced/blocked games, venue rules, game-time dicts) lives in `config/`, not hardcoded in solver classes. See `docs/CONFIGURATION_REFERENCE.md` and `docs/ai/CONFIGURATION_REFERENCE.md`.
- **Solver runs are observable.** Every run writes checkpoints (`checkpoints/run_XX/`), logs (`logs/`), and a versioned draw with full metadata (mode, stages, slack, forced/blocked outcomes, penalties). Never re-run blind.
- **Hand-edits are auditable.** `scripts/test_scenario.py` produces a BEFORE/AFTER violation diff for any proposed change before it touches `current.json`.
- **Tests are real.** No mocking of solver state. Real CP-SAT models with small sampled fixtures. The test bar is a tripwire — every phase must hold or lift it.

---

## 4. What success looks like

You can — without thinking too hard:

1. Add a new constraint by writing one file in `constraints/atoms/`, one registry entry, one test, one stage entry.
2. Skip one piece of a rule for one season by removing one atom name from one stage.
3. Loosen one dimension of a rule by passing `--slack N` and knowing exactly which atoms read that slack.
4. Diagnose an infeasibility down to a single cluster via `run.py diagnose --stage X`.
5. Apply a convenor hand-edit, see the violation diff, promote it as a MINOR version, all in one script.
6. Hand a new convenor (or AI agent) `CLAUDE.md` + this file + `docs/ATOMIZATION_PLAN.md` and have them productive in an hour.

If any of these starts to feel hard, that's the signal that something has drifted from the model. Fix the drift before adding more features.

---

## Doc map

| Doc | Role |
|---|---|
| **`CLAUDE.md`** (repo root) | Mandatory first read for any AI session. Quick-reference rules, file structure, pitfalls, all-the-commands cheat sheet. |
| **`docs/GOALS.md`** (this file) | The *why*. Product + engineering goals. Read alongside CLAUDE.md. |
| **`docs/ATOMIZATION_PLAN.md`** | Phase-by-phase plan + status table. All phases ✅ as of 2026-04-29. Read for refactor context. |
| **`docs/ATOMIZATION_HANDOFF.md`** | Commit-chain handoff doc with full per-phase notes. Read to understand a specific phase's history. |
| **`docs/CONSTRAINT_INVENTORY.md`** | Single source of truth: every registered constraint, its severity, slack key, atom-target mapping. 37 entries. |
| **`docs/HELPER_VARS.md`** | `HelperVarRegistry` API — how atoms declare and share helper BoolVars/IntVars. |
| **`docs/COUNT_ADJUSTERS.md`** | Phase-4 FORCED/BLOCKED count adjuster formulas + status. |
| **`docs/STAGES.md`** | `SOLVER_STAGES` config schema, validation, CLI flags (`--stages-config`, `--stage-only`, `--skip-stage`, `--list-stages`). |
| **`docs/HARNESS.md`** | End-to-end solver pipeline reference: `generate_X` → `validate_game_config` → unified engine → stages → output. |
| **`docs/FORCED_GAMES_AS_COUNT_RULES.md`** | Why per-venue Friday count atoms were retired in favour of `FORCED_GAMES`. Perennial rule. |
| **`docs/PERENNIAL_RULES.md`** | Standing rules that apply *every* season (rounds 1–2 at Broadmeadow, last-game-on-WF, FORCED-as-count-budget). |
| **`docs/DRAW_RULES.md`** | Convenor-facing rule documentation (hard vs soft constraints, by rule). |
| **`docs/CAPABILITIES.md`** | What the system can do. Pitch deck / external-facing. |
| **`docs/SYSTEM_OVERVIEW.md`** | Architecture overview. CP-SAT, decision variables, pipeline. |
| **`docs/README.md`** | Doc index + quick orientation. |
| **`docs/ai/AI_OPERATIONS_MANUAL.md`** | Deep technical reference for AI agents. |
| **`docs/ai/CONFIGURATION_REFERENCE.md`** | Every config parameter. Read before changing config. |
| **`docs/ai/CONSTRAINT_APPLICATION.md`** | How to apply restrictions (FORCED, BLOCKED, etc.). |
| **`docs/ai/GAME_TIME_DICTIONARIES.md`** | PHL_GAME_TIMES / SECOND_GRADE_TIMES filtering rules. |
| **`docs/ai/SEASON_SETUP.md`** | Spinning up a new season. |
| **`docs/ai/SYSTEM_OPERATION.md`** | Running the solver, monitoring, troubleshooting. |
| **`README.md`** (repo root) | Project quick-start. External-facing. |
| **`TODO.md`** (repo root) | In-flight items the convenor is tracking. Volatile. |
| **`convenor_notes.md`** (repo root) | Convenor's working scratchpad. Volatile. |
| **`club_contacts.md`** (repo root) | Club contact list. Reference data. |
