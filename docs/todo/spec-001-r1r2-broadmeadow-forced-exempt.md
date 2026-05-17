<!-- status: in_progress -->
<!-- owner: session=opus-4.7-spec001 claimed=2026-05-17T08:24:41Z -->
<!-- depends_on: none -->

# spec-001 — Rounds 1–2 Broadmeadow-only rule exempts FORCED games

**Spec source:** [`docs/todo/GOALS.md` → spec-001](GOALS.md#spec-001--rounds-12-broadmeadow-only-rule-must-exempt-forced-games)

## Why

`PERENNIAL_BLOCKED_GAMES` in `config/defaults.py` strips every non-NIHC variable for rounds 1 and 2. When the convenor places a FORCED entry in those rounds (a specific Maitland-vs-Norths opener at Maitland Park, say), the perennial BLOCKED deletes the variable before the FORCED can match it — the FORCED entry then silently fails (no var to constrain). Convenor is forced to comment out the perennial rule for the season.

## Definition of Done

1. A variable that matches both a `PERENNIAL_BLOCKED_GAMES` scope **and** any `FORCED_GAMES` scope is **kept** (FORCED wins).
2. A variable that matches `PERENNIAL_BLOCKED_GAMES` but no FORCED scope is **eliminated** (current behaviour preserved).
3. A regression test exists with a fixture that places a FORCED entry in round 1 at Maitland Park, runs `generate_X`, asserts the variable survives, and asserts no other round 1/2 non-NIHC vars survive.
4. `docs/system/CONSTRAINT_INVENTORY.md` has a new row (or annotation on `PERENNIAL_BLOCKED_GAMES`) documenting the exemption rule.
5. `docs/operator-human/PERENNIAL_RULES.md` mentions the FORCED-overrides-perennial-BLOCKED rule.
6. `CLAUDE.md` section "Per-venue / per-day game counts use FORCED_GAMES, NOT constraints" updated if needed to reflect the precedence rule.

## Implementation units

### Unit 1 — generate_X resolution rule

- **Files touched:** `utils.py` (`generate_X` / `apply_blocked_games` or wherever PERENNIAL_BLOCKED is applied).
- **Change:** Before eliminating a variable matching a PERENNIAL_BLOCKED scope, check if any FORCED_GAMES scope matches the same variable using `_get_matching_forced_scopes` (or equivalent). If yes, skip elimination.
- **No-mock test:** in `tests/test_perennial_blocked_forced_exemption.py`, build a tiny data dict with one FORCED entry in round 1 at Maitland Park and assert the variable is in `X` after `generate_X`.

### Unit 2 — Docs

- **Files touched:** `docs/system/CONSTRAINT_INVENTORY.md`, `docs/operator-human/PERENNIAL_RULES.md`, `CLAUDE.md`.
- **Change:** Document the precedence rule. Each doc gets one sentence + a 2-line example.

## Doc registry (per /basic)

- `docs/system/CONSTRAINT_INVENTORY.md` — annotate PERENNIAL_BLOCKED entry with FORCED-exemption note
- `docs/operator-human/PERENNIAL_RULES.md` — add the precedence rule
- `CLAUDE.md` — single-line mention near the FORCED/BLOCKED section
- `docs/todo/GOALS.md` — flip spec-001 status to "done" in the spec table

## Out of scope

- Refactoring BLOCKED_GAMES (non-perennial) to also respect FORCED — the perennial-only fix is targeted; if user wants the same rule for season-specific BLOCKED, spawn a new plan.
- Per-atom forced-exemption documentation across the whole registry — covered by the atom-registry expansion already in `CONSTRAINT_INVENTORY.md`.
