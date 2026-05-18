<!-- status: done -->
<!-- owner: session=spec-010-agent claimed=2026-05-18T00:00:00Z -->
<!-- completed: 2026-05-19 -->
<!-- depends_on: none -->

# spec-010 — Remove "every PHL team plays round 1"; verify R1/R2 carve-out for 2nd grade

**Spec source:** Convenor request 2026-05-18 (research session).

## Why

Two related round-1 issues:

1. The `PHLRoundOnePlay` atom (`constraints/atoms/phl_round_one_play.py`) currently requires every PHL team to play in round 1. With FORCED PHL games at non-Broadmeadow venues now legal via spec-001 (FORCED ⊕ PERENNIAL → FORCED), and with the convenor wanting flexibility to start some PHL teams in round 2, this atom is over-constraining. The constraint hides legitimate solutions.

2. spec-001 shipped "FORCED overrides PERENNIAL R1/R2-Broadmeadow-only" for PHL. The convenor needs the **same carve-out to apply to 2nd grade FORCED entries** — i.e. a forced 2nd-grade game in round 1 at Maitland Park should also defeat the perennial block. spec-001's implementation (`generate_X` in `utils.py` reading the `'perennial': True` flag) should already cover this since it's grade-agnostic, but it has not been explicitly verified for the 2nd-grade case.

## Definition of Done

### Part A — remove `PHLRoundOnePlay`

1. `constraints/atoms/phl_round_one_play.py` — deleted, OR atom kept but removed from `_PHL_HARD_ATOMS` in `constraints/unified.py` and from the `critical_feasibility` stage in `config/defaults.py::DEFAULT_STAGES`. Preferred: keep the file (parity reference) but remove from registry's stage list AND mark the registry entry's `solver_class_names` empty so accidental re-introduction via legacy CLI flags is impossible.
2. Tests referencing `PHLRoundOnePlay` updated: any test asserting the rule fires is removed; any test asserting that *not* enforcing it produces a valid schedule is added.
3. The compliance certificate / pre-season report no longer lists "every PHL team plays round 1" as a constraint.
4. Legacy `constraints/archived/original.py` and `archived/ai.py` references are NOT touched — they're archived for parity, not production.

### Part B — verify 2nd-grade FORCED-overrides-PERENNIAL carve-out

1. New test `tests/test_perennial_carveout_2nd_grade.py`:
   - Fixture has perennial R1/R2 Maitland Park / Central Coast block in `BLOCKED_GAMES`.
   - FORCED entry: `{teams:['Maitland 2nd','Norths 2nd'], grade:'2nd', date:'<round-1 date>', field_location:'Maitland Park'}`.
   - After `generate_X`, assert the matching variable exists (FORCED won).
   - Solve and assert the game lands at Maitland Park on the round-1 date.
2. If `generate_X` is grade-blind today, this test should pass on the first run — making it explicit also locks the behaviour.
3. `docs/operator-human/PERENNIAL_RULES.md` — Rule "R1/R2 at Broadmeadow only" gets a one-line note: "Override via FORCED_GAMES applies to ALL grades."

### Part C — Docs

1. `docs/system/CONSTRAINT_INVENTORY.md` — `PHLRoundOnePlay` row marked obsolete with rationale.
2. `docs/operator-human/RULES.md` — drop the "every PHL team plays in round 1" bullet if present.
3. `docs/operator-ai/CONSTRAINT_APPLICATION.md` — same.
4. `config/defaults.py::DEFAULT_STAGES` — remove `'PHLRoundOnePlay'` from `critical_feasibility`.

## Implementation units

### Unit 1 — Remove the atom

- **Files touched:** `constraints/unified.py` (`_PHL_HARD_ATOMS` tuple), `config/defaults.py::DEFAULT_STAGES`, `constraints/atoms/__init__.py`, `constraints/registry.py`, `tests/atoms/test_phl_atoms.py` (drop tests), `tests/test_constraint_registry.py` (adjust entry count).

### Unit 2 — Verify 2nd grade R1/R2 carve-out

- **Files touched:** `tests/test_perennial_carveout_2nd_grade.py` (new), `docs/operator-human/PERENNIAL_RULES.md`.

### Unit 3 — Docs

- **Files touched:** `docs/system/CONSTRAINT_INVENTORY.md`, `docs/operator-human/RULES.md`, `docs/operator-ai/CONSTRAINT_APPLICATION.md`.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — PHLRoundOnePlay → obsolete
- `docs/operator-human/RULES.md` — drop round-1 PHL line
- `docs/operator-human/PERENNIAL_RULES.md` — multi-grade override note
- `docs/operator-ai/CONSTRAINT_APPLICATION.md`
- `docs/todo/GOALS.md` — add spec-010 row, flip to "done"

## Out of scope

- Adding a *soft* preference that PHL teams play in round 1 (the convenor's stance is: not needed; FORCED entries express the intent directly).
- Touching any other atoms in the `PHLAndSecondGradeTimes` cluster.
