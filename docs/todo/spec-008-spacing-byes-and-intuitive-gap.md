<!-- status: ready -->
<!-- owner: unassigned -->
<!-- depends_on: none -->

# spec-008 — Spacing: byes as first-class + intuitive gap semantics

**Spec source:** Convenor request 2026-05-18 (research session).

## Why

Two related problems with the current `EqualMatchUpSpacing` atom (`constraints/unified.py::_matchup_spacing_hard`):

1. **Byes are invisible.** A team's *bye* rounds (rounds in which it doesn't play) carry the same "we'd like these spread evenly" intent as repeat matchups, but they aren't modelled at all. With unequal grade sizes some teams get 3 byes in a row early then play every round; the spacing constraint never sees this.
2. **`gap` semantics are off by one.** Today `min_gap = max(1, T-1 - base_slack)` and the forbidden-gap check is `r2 - r1 < min_gap`. Result: a "spacing of 2" enforces only ONE played round between the two meetings (rounds 1 and 3). The convenor reads "spacing 2" as "two rounds of breathing room between meetings" — i.e. rounds 1 and 4. Off-by-one is a footgun every time it's tuned.

## Definition of Done

### Part A — intuitive gap semantics

1. The spacing input number `S` means **S played rounds between meetings**, NOT `r2 - r1 = S`.
2. The hard check becomes `r2 - r1 - 1 < S` (equivalently `r2 - r1 <= S`).
3. `space = T - 1` (the ideal-meeting distance for a balanced round-robin) is preserved as the *default* `S`, but expressed via a named helper `ideal_gap(T)` so the off-by-one fix is localised.
4. `CONSTRAINT_DEFAULTS['spacing_base_slack']` semantics documented in `docs/system/COUNT_ADJUSTERS.md`: slack `K` means the convenor accepts the gap shrinking by up to `K` from ideal.
5. Updated parity tests: existing tests that expect the old off-by-one behaviour are migrated (not deleted) — they should pass against the new math after the semantic shift, because the *physical schedule* a healthy solver produces is unchanged.
6. CLI/`--slack N` semantics unchanged from the convenor's perspective — `--slack 3` still loosens by 3 played rounds.

### Part B — byes as first-class

1. New atom `BalancedByeSpacing` in `constraints/atoms/balanced_bye_spacing.py`.
2. Math:
   - For each grade `g` with `T` teams, total playable rounds in season is `R` (from `data['num_rounds']` / playable-weeks count).
   - Per-team games scheduled is `games_per_team[g]` (from `max_games_per_grade`).
   - Bye count per team is `byes_per_team = R - games_per_team[g]`.
   - For each team `t` in grade `g`, the rounds in which `t` does NOT play are forced to be evenly distributed using the same `ideal_gap` logic as matchup spacing: pairwise check on bye-rounds with `min_bye_gap = ideal_gap(R, byes_per_team)` (formula in atom; see Implementation notes).
3. Atom is HARD with slack key `BalancedByeSpacing` (separate from `EqualMatchUpSpacing` slack so the convenor can loosen one without the other).
4. Atom respects locked weeks — only enforces over the unlocked portion.
5. Co-exists with `EqualMatchUpSpacing` cleanly (independent dimensions).
6. Registry entry severity 2 (HIGH) — byes spread well is important but not a feasibility blocker; can be relaxed.

### Part C — tests + docs

1. `tests/atoms/test_balanced_bye_spacing.py` — small real CP-SAT fixture:
   - Given grade with 6 teams, 18 playable rounds, max 16 games per team → 2 byes per team. When solved, Then no team has both byes in adjacent rounds.
   - Given a 5-team grade (one bye per round structurally) where teams should rotate the bye → bye rounds for each team are spread `ideal_gap(R, byes)` apart.
   - Given locked weeks 1-3, byes inside the locked window are ignored.
2. `tests/test_spacing_intuitive_gap.py` — parity test demonstrating that `S=2` now forbids `(r1, r2=r1+1)` and `(r1, r2=r1+2)` (was: only `r2=r1+1`).
3. `docs/system/CONSTRAINT_INVENTORY.md` — new `BalancedByeSpacing` row; `EqualMatchUpSpacing` row updated with the new gap semantics.
4. `docs/operator-human/RULES.md` — plain-English: "If your team has byes, they'll be spread across the season just like repeat matchups."
5. `docs/operator-ai/CONFIGURATION_REFERENCE.md` — `--slack BalancedByeSpacing N` documented.

## Implementation units

### Unit 1 — Intuitive gap (Part A)

- **Files touched:** `constraints/unified.py::_matchup_spacing_hard`, `constraints/atoms/_adjusters.py::equal_matchup_spacing_adjuster` (sanity-check the adjuster math against the new semantics), tests under `tests/test_spacing_integration.py` and `tests/atoms/test_*spacing*`.
- Single-line semantic change (`gap < min_gap` → `gap <= min_gap`, with the ideal value definition shifted). Carefully audit every reference to "spacing base slack" in the codebase to ensure no downstream consumer hard-codes the old semantics.

### Unit 2 — Bye atom (Part B)

- **Files touched:** `constraints/atoms/balanced_bye_spacing.py` (new), `constraints/registry.py`, `config/defaults.py::DEFAULT_STAGES` (add to `critical_feasibility` or a new `spacing` stage — implementer's call), `tests/atoms/test_balanced_bye_spacing.py` (new).
- Helper: a per-team "not playing in round r" BoolVar built from `1 - sum(team's vars in round r)`. Pairwise no-double-bye check matches the structure of `_matchup_spacing_hard`.

### Unit 3 — Docs

- **Files touched:** `docs/system/CONSTRAINT_INVENTORY.md`, `docs/operator-human/RULES.md`, `docs/operator-ai/CONFIGURATION_REFERENCE.md`, `docs/system/COUNT_ADJUSTERS.md` (note the new `gap` semantics).

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — `BalancedByeSpacing` row, `EqualMatchUpSpacing` semantics note
- `docs/operator-human/RULES.md` — byes section
- `docs/operator-ai/CONFIGURATION_REFERENCE.md` — new slack key
- `docs/system/COUNT_ADJUSTERS.md` — gap semantics shift note
- `docs/todo/GOALS.md` — add spec-008 row, flip to "done" on completion

## Out of scope

- Soft penalty for bye unevenness (a separate atom if it ever turns out hard mode is too tight).
- Reshaping `EqualMatchUpSpacing` into the new atom registry pattern (still a `unified.py` legacy method — separate atomisation plan).
