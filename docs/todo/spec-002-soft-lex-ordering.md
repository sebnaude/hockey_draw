<!-- status: in_progress -->
<!-- owner: session=spec-002-impl claimed=2026-05-17T00:00:00Z -->
<!-- depends_on: none -->

# spec-002 — Soft lexicographical ordering of matchups across the season

**Spec source:** [`docs/todo/GOALS.md` → spec-002](GOALS.md#spec-002--soft-lexicographical-ordering-of-matchups-across-the-season)

## Why

Without a tie-break, the solver's choice of "matchup X happens in week K" is arbitrary among feasibility-equivalent options. Clubs reading the draw see no pattern. A tiny soft penalty that prefers alphabetically-earlier matchups in earlier rounds gives a deterministic, predictable structure — purely a tie-break, never overrides a real constraint.

## Definition of Done

1. A new atom `SoftLexMatchupOrdering` exists in `constraints/atoms/` and is registered with severity 5 (VERY LOW).
2. Per (grade, week), the atom adds a penalty proportional to the number of matchups in that week whose alphabetical rank exceeds the alphabetical rank of any matchup deferred to a later week.
3. The penalty weight is tiny — `PENALTY_WEIGHTS['soft_lex_ordering']` defaults to `1` (vs typical soft penalties of 10–100k). The weight value is configurable.
4. The atom is wired into the `soft_optimisation` stage in `config/defaults.py::DEFAULT_STAGES`.
5. Unit test with a tiny fixture (3 teams, 2 rounds) confirms: with no other constraints active, the solver schedules `(Norths, Tigers)` before `(Tigers, Wests)` because N-T sorts before T-W.
6. Test confirms the atom never blocks feasibility — even with all matchups forced into the "wrong" order, the model remains SAT (just with a higher objective penalty).
7. `docs/system/CONSTRAINT_INVENTORY.md` gets a new row.
8. `docs/operator-human/RULES.md` mentions the predictable-ordering soft preference.

## Implementation units

### Unit 1 — Atom implementation

- **Files touched:** `constraints/atoms/soft_lex_matchup_ordering.py` (new), `constraints/registry.py`, `config/defaults.py::DEFAULT_STAGES`, `config/defaults.py::PENALTY_WEIGHTS`.
- **Change:** New atom subclassing `Constraint`. For each (grade, pair) compute the alphabetical rank `r`. For each round in which the pair plays, penalty contribution = `weight * (r - expected_round_for_rank_r)` clamped at 0. Or simpler: for each pair `(a,b)` and round `k`, penalty = `weight * sum(rank_of_pair_in_round_k - rank_in_alphabetical_order)`. (Exact formulation up to implementor — DoD is the behavioural outcome.)
- **No-mock test:** synthetic 3-team, 2-round fixture, GWT scenarios:
  - Given empty constraints + soft-lex, When solved, Then `(Norths, Tigers)` appears in round 1 and `(Tigers, Wests)` in round 2.
  - Given a hard FORCED that flips the order, When solved, Then model is still SAT (objective just higher).

### Unit 2 — Docs

- **Files touched:** `docs/system/CONSTRAINT_INVENTORY.md`, `docs/operator-human/RULES.md`.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — new row for `SoftLexMatchupOrdering`
- `docs/operator-human/RULES.md` — one paragraph under "soft preferences"
- `docs/todo/GOALS.md` — flip spec-002 status to "done"

## Out of scope

- Hard lexicographical ordering (would conflict with too many real constraints).
- Multi-key sort (e.g. alphabetical then by club size) — single-key implementation only.
