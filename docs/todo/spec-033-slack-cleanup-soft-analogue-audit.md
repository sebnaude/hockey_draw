<!-- status: ready -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-032 -->
<!-- owner: session=slack claimed=2026-05-24 -->
<!-- reviewed: adversarial Sonnet review 2026-05-24 — fixes applied inline -->

# spec-033 — Remove ClubVsClubAlignment slack + audit soft-analogue coverage of slack-tolerant constraints

## Why

The `--slack` system on `final-form` is uneven and partly dead. Three findings (all verified against the worktree this session) motivate this spec:

1. **`ClubVsClubAlignment` slack is dead in the solver but live in the tester — a divergence.** The production club-alignment path is the spec-005 *stacked* cluster (`ClubVsClubStackedWeekends` + `ClubVsClubStackedCoLocation`, tagged `groups={'core','club_alignment'}`), and **neither stacked atom reads slack** (verified: `grep -L slack` on both atom files). The only code that reads `slack['ClubVsClubAlignment']` is the engine method `_club_alignment_hard`/`_club_alignment_soft` (`constraints/unified.py:630,678,846`), whose engine key `ClubVsClubAlignment` carries **no `groups=`** (`registry.py:333`) so it is never dispatched in a real build — the registry itself calls it "legacy parity-reference engine path" (`registry.py:344-345`). Yet the tester still applies the slack (`analytics/tester.py:2011,2066`). Net: `--slack ClubVsClubAlignment N` loosens the *checker* against a rule the *solver* enforces strictly. The convenor wants this knob gone entirely so alignment is a fixed hard rule.

2. **Two config base-slack knobs are dead.** `club_vs_club_alignment_base_slack` is read only in `constraints/archived/original.py:1097` (parity reference), never in the live engine. (`club_game_spread_max_gap`/`max_overlap` are likewise dead but are **out of scope** here — see Out of scope.)

3. **`BalancedByeSpacing` tolerates deviation (it has a slack key and a hard floor `S`) but has NO normal-mode soft analogue pushing toward optimal bye spread.** Verified: there is no `_bye_spacing_soft` in `unified.py` and `apply_stage_2_soft` (`unified.py:760-785`) never adds a bye-spacing penalty. A soft analogue exists ONLY for regeneration mode (`constraints/atoms/balanced_bye_spacing_regen_soft.py`, group `regen_soft`, which `stages.py:243` excludes from every normal build). This violates the convenor's stated principle: *every constraint that tolerates some failure should carry a soft constraint pushing toward perfection.* The two other kept slack constraints already satisfy it — `EqualMatchUpSpacing` (`_matchup_spacing_soft`, `unified.py:796`) and `ClubGameSpread` (`_club_game_spread_soft`, `unified.py:1094`, sharing the hard atom's `_cgs_hole_vars`) — so bye-spacing is the lone gap among the slack-tolerant constraints.

Cost of not solving: a misleading tester (1), stale dead config that misleads future seasons (2), and uneven bye distribution with no optimisation pressure in normal solves (3).

### Audit result (the "check soft-analogue coverage" deliverable)

Constraints that *tolerate deviation* (have slack and/or a deviation-allowed cap) and therefore SHOULD carry an engine/atom soft analogue:

| Constraint | Tolerant? | Soft analogue today | Verdict |
|---|---|---|---|
| EqualMatchUpSpacing | yes (slack + base_slack) | `_matchup_spacing_soft` (engine, sliding-window density) | ✅ present — verify only |
| ClubGameSpread | yes (per-field hole cap, default +1 for ≥4-game fields) | `_club_game_spread_soft` (engine, shares `_cgs_hole_vars`, drives holes→0 + off-primary) | ✅ present — verify only |
| BalancedByeSpacing | yes (slack + hard floor S) | **none in normal mode** (regen-only) | ❌ **gap — Unit B adds it** |
| ClubVsClubAlignment | being made **non-tolerant** (slack removed) | n/a after this spec — becomes a fixed hard rule | n/a (Unit A) |

Constraints that are **hard-only / cannot deviate** and correctly need NO soft analogue (confirmed, no action): `NoDoubleBookingTeams`, `NoDoubleBookingFields`, `EqualGamesAndBalanceMatchUps`, `FiftyFiftyHomeandAway`, `TeamConflict`, `SameGradeSameClubNoConcurrency`, `ClubNoConcurrentSlot`, the `ClubDay*` atom set, `PHLConcurrency`/`PHLAnd2ndConcurrency`/`PHLAnd2ndAdjacency`/`PHLAndSecondGradeTimes`. (Constraints already soft-by-nature — `NIHCFill*`, `TeamPairNoConcurrency`, `PreferredGames`, `PreferredWeekendsAwayGround`, `VenueEarliestSlotFill` — are out of scope; they are not hard rules with slack.)

## Definition of Done

**Unit A — ClubVsClubAlignment slack removal + kept-constraint soft verification:**

1. `registry.py`: `slack_key='ClubVsClubAlignment'` is removed from all THREE entries — `ClubVsClubAlignment` (line ~339), `ClubVsClubStackedWeekends` (~364), `ClubVsClubStackedCoLocation` (~375). `get_slack_key('ClubVsClubAlignment')`, `get_slack_key('ClubVsClubStackedWeekends')`, and `get_slack_key('ClubVsClubStackedCoLocation')` all return `None`.
2. `run.py`: the `'ClubVsClubAlignment': slack_value,` line is removed from the `constraint_slack` dict (~line 814). `--slack N` no longer populates any ClubVsClubAlignment key.
3. `analytics/tester.py::_check_club_vs_club_alignment`: the `config_slack` read (~line 2011) is removed and `min_required` becomes `num_games` (was `max(0, num_games - config_slack)`, ~line 2066). The severity-map entry `'ClubVsClubAlignment': 3` (~line 84) is LEFT unchanged (it is a severity level, not slack).
4. `constraints/unified.py`: `_club_alignment_hard` and `_club_alignment_soft` no longer read `self.slack.get('ClubVsClubAlignment', …)`; `min_req`/`min_required` is `num_games` (strict). (These methods are the dead parity path; we only strip slack, we do not delete them — see Out of scope.)
5. `config/defaults.py`, `config/season_2026.py`, `config/season_template.py`: the `club_vs_club_alignment_base_slack` key is removed. `tests/test_constraint_defaults_merge.py` no longer asserts that key. `constraints/archived/original.py:1097` is LEFT unchanged (archived parity reference).
6. The stale EqualMatchUpSpacing formula comment in `config/season_2026.py:780` (`min_gap = max(T//2+1, …)`) is corrected to the spec-008 `ideal_gap(T)` semantics, and the `ClubVsClubAlignment: … slack reduces required coincidences` comment (season_2026.py:781) is removed. (review fix — H1: actual lines are 780/781, not 776/777 as originally cited; verified against current file.)
7. **Verification (no code change expected; assert + document):** a no-mock test confirms `EqualMatchUpSpacing` enforces `spacing_base_slack` (default 2 for 2026) AND emits sliding-window soft penalties under key `'EqualMatchUpSpacing'`; and `ClubGameSpread` permits ≤1 interior hole per field for a ≥4-game field, ≤0 for a ≤3-game field, and emits a per-hole soft penalty under key `'ClubGameSpread'` that drives residual holes to zero. If either is found defective, fix inline (re-grade if it balloons).
8. Audit table (the "Audit result" section above) is recorded in `docs/system/CONSTRAINT_INVENTORY.md` as the canonical soft-analogue coverage reference.
9. Full test suite green; registry parity/count tests pass; `run.py generate --year 2026 --simple` builds a model without error (slack flag still accepted, ClubVsClubAlignment no longer in the dict).

**Unit B — BalancedByeSpacing soft analogue (normal mode):**

10. `constraints/atoms/balanced_bye_spacing.py` gains a soft component that REUSES the per-round `bye_var` indicators it already builds (sharing variables; no second variable family), adding a penalty term per bye-pair that sits closer than ideal (or a sliding-window bye density, mirroring `_matchup_spacing_soft`), appended to `data['penalties']['BalancedByeSpacing']['penalties']`. The penalty bucket key MUST be `'BalancedByeSpacing'` (NOT `'regen_balanced_bye_spacing'` — that is the regen-only atom's key). The aggregation path is confirmed: `main_staged.py:1279-1284` reads `data.get('penalties', {})` and passes ALL bucket entries to `_build_normalized_penalty`, so any bucket key flows to the objective automatically. (review fix — H2: clarified that `regen_balanced_bye_spacing_regen_soft.py` "proves the mechanism" — i.e. that atoms can emit to `data['penalties']` — but its bucket key is `regen_balanced_bye_spacing`, not `BalancedByeSpacing`. The normal-mode atom must use the unprefixed `'BalancedByeSpacing'` key. Objective aggregation is verified at `main_staged.py:1279-1284`.)
11. `registry.py`: the `BalancedByeSpacing` entry sets `has_soft_component=True`.
12. `config/season_2026.py::PENALTY_WEIGHTS` gains a `'BalancedByeSpacing'` weight (sibling to the existing `'EqualMatchUpSpacing': 100_000`); the atom reads it via `data.get('penalty_weights', {}).get('BalancedByeSpacing', DEFAULT_WEIGHT)` — the same direct dict-lookup the regen-soft atom uses (`balanced_bye_spacing_regen_soft.py:79-80`). NOTE: `_get_penalty_weight` is a method of `UnifiedConstraintEngine` — do NOT call it from within the atom, which has no engine reference. (review fix — H3: clarified the lookup mechanism so the implementer doesn't attempt to call the engine's `_get_penalty_weight` from atom code.)
13. The soft penalty reaches the objective `Maximize(... - soft_penalties)` via `main_staged.py:1279-1284` (confirmed by reviewer): `data.get('penalties', {})` is iterated and ALL buckets passed to `_build_normalized_penalty`, which normalises each bucket's weight by var-count and returns `(coefficient, var)` pairs summed into the objective expression. The new `'BalancedByeSpacing'` bucket is therefore automatic — no additional wiring required. Verify post-solve that the bucket appears in the printed penalty summary. (review fix — M3: the aggregation path was "confirm" work in DoD; it is now confirmed and cited.)
14. The hard floor `S` and slack semantics of `BalancedByeSpacing` are UNCHANGED — the soft term only adds pressure below the hard threshold, never relaxes or tightens the hard rule.
15. No-mock test: a synthetic grade with byes_per_team≥2 where two feasible schedules differ only in bye spread; the solver prefers the better-spread one (hand-computed objective delta in the test comment); and a degenerate single-bye grade adds zero bye penalties.
16. Full suite green; `BalancedByeSpacing` soft does not regress `EqualMatchUpSpacing`/`ClubGameSpread` penalty counts.

## Implementation units

### Unit A — Remove ClubVsClubAlignment slack + verify kept soft analogues  (S2; executor: Opus — cross-file removal with a live/dead-path trap)

**Files touched:**
- `constraints/registry.py` — drop `slack_key` from the 3 ClubVsClub* entries.
- `run.py` — drop the ClubVsClubAlignment line from the `constraint_slack` dict (~814); ALSO remove `ClubVsClubAlignment` from the help text at line ~118 (the `'..., ClubVsClubAlignment, ...'` substring in the `--slack` flag help string).
- `analytics/tester.py` — `_check_club_vs_club_alignment`: remove `config_slack`, set `min_required = num_games`.
- `constraints/unified.py` — `_club_alignment_hard` (630/678) + `_club_alignment_soft` (846): strip `config_slack`, `min_req`/`min_required = num_games`.
- `config/defaults.py`, `config/season_2026.py`, `config/season_template.py` — remove `club_vs_club_alignment_base_slack`; fix the two stale comments (season_2026 780-781). (review fix — H1: line numbers corrected from 776-777 to 780-781.)
- `tests/test_constraint_defaults_merge.py` — drop the removed key from the expected-keys list.
- `docs/system/CONSTRAINT_INVENTORY.md` — add the soft-analogue audit table; note ClubVsClubAlignment is now slack-free.
- `docs/operator-ai/AI_OPERATIONS_MANUAL.md`, `docs/operator-ai/SYSTEM_OPERATION.md` — strike `ClubVsClubAlignment` from any `--slack` "applies to" list.
- `CLAUDE.md` (project, in worktree) — remove `ClubVsClubAlignment` from the `--slack` "Applies to" list (~line 269) AND from the example `constraint_slack` dict (~line 374). Verified: the worktree CLAUDE.md names it in both places. (review fix — M2: CLAUDE.md was in the Doc Registry but omitted from the Unit A file list; added here as a required edit.)
- New/updated test: `tests/` — assert `get_slack_key` returns `None` for the 3 names; assert EqualMatchUpSpacing base-slack + soft and ClubGameSpread per-field hole cap + soft (DoD 7).

**Change summary:** purely subtractive for slack + two comment fixes + a verification test for the two kept soft analogues + the audit-table doc. Do NOT delete the dead `_club_alignment_*` methods (that is a separate concern — Out of scope).

**Dependency:** none within this plan (first unit). Blocked at the plan level by `spec-032` landing (shared `registry.py`/`run.py`/`CONSTRAINT_INVENTORY.md`/`CLAUDE.md`).

**No-mock test outline (hand oracle):**
- *Given* the registry, *when* `get_slack_key('ClubVsClubStackedWeekends')`, *then* `None` (was `'ClubVsClubAlignment'`).
- *Given* a 10-team grade, *when* the engine applies EqualMatchUpSpacing with `spacing_base_slack=2`, *then* `effective_spacing(10, base_slack=2) = ideal_gap(10) - 2 = 5 - 2 = 3` (hand-computed: `_legacy_min_gap(10)=6`, `ideal_gap=5`), and a `'EqualMatchUpSpacing'` penalty bucket is non-empty.
- *Given* a club with 4 games on one field across slots {1,2,4,5} (hole at 3), *when* `_club_game_spread_hard`+`_soft` apply with slack 0, *then* the hard cap is `1+0=1` (feasible) and the soft penalty bucket contains the hole indicator (drives toward filling slot 3).

### Unit B — Add BalancedByeSpacing soft analogue  (S2; executor: Opus — CP-SAT penalty wiring + objective path)

**Files touched:**
- `constraints/atoms/balanced_bye_spacing.py` — add the soft term reusing `bye_var`; append to `data['penalties']['BalancedByeSpacing']`.
- `constraints/registry.py` — `BalancedByeSpacing` entry: `has_soft_component=True`.
- `config/season_2026.py` — `PENALTY_WEIGHTS['BalancedByeSpacing']`.
- `docs/system/CONSTRAINT_INVENTORY.md` — flip the bye-spacing row in the audit table from ❌ to ✅; note the new soft component + weight.
- New test: `tests/atoms/` (or alongside existing bye-spacing tests) per DoD 15.

**Change summary:** make the existing bye-spacing atom carry a soft analogue with the same structure as `balanced_bye_spacing_regen_soft.py` (proven atom→penalty mechanism), but emitted in normal builds under the bucket key `'BalancedByeSpacing'` (NOT `'regen_balanced_bye_spacing'`). Mirror `_matchup_spacing_soft`'s sliding-window shape using the bye indicators. (review fix — H2: disambiguates the regen-soft atom's `regen_` prefixed key from the normal-mode atom's unprefixed key.)

**Dependency:** depends on Unit A — both edit `constraints/registry.py`, `config/season_2026.py`, and `docs/system/CONSTRAINT_INVENTORY.md`; serialise B after A to avoid collisions.

**No-mock test outline (hand oracle):**
- *Given* a grade of 9 teams over R=18 rounds → games_per_team computed, byes_per_team = R−games; pick a config where byes_per_team=2 so `ideal_bye_gap(18,2)=8`. *When* two feasible schedules exist — one with byes 9 rounds apart (gap 9 > 8, zero hard violation, low soft) and one 8 apart (gap 8 ≤ 8, hard-forbidden) vs a relaxed-slack case where 8-apart is allowed but penalised — *then* the solver's objective prefers the 9-apart schedule; hand-compute the penalty delta = weight × (pairs within the soft window).
- *Given* a grade where byes_per_team ≤ 1, *then* zero bye penalties added (atom short-circuits).

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — **(both units)** add the soft-analogue coverage audit table; record ClubVsClubAlignment as slack-free; flip the bye-spacing row to ✅ once Unit B lands; adjust any slack-key listing.
- `docs/operator-ai/AI_OPERATIONS_MANUAL.md` — **(Unit A)** remove `ClubVsClubAlignment` from `--slack` documentation; note bye-spacing now has a soft component.
- `docs/operator-ai/SYSTEM_OPERATION.md` — **(Unit A)** same `--slack` list correction.
- `config/season_2026.py` inline comments — **(Unit A)** fix the stale EqualMatchUpSpacing formula (line 780) + ClubVsClubAlignment slack comments (line 781); remove `club_vs_club_alignment_base_slack` entry (line 792). (review fix — H1: corrected line references from ~776-788 to 780/781/792.)
- `CLAUDE.md` (project, in worktree) — **(Unit A)** remove `ClubVsClubAlignment` from `--slack` "Applies to" list (~line 269) and from the example `constraint_slack` dict (~line 374). Verified present in both places. (review fix — M2: confirmed needed; moved from "confirm whether" to a definite required edit.)

## Out of scope

- **Deleting the dead `_club_alignment_hard`/`_club_alignment_soft` engine methods and the unused `ClubVsClubAlignment` registry engine key.** They are superseded by the stacked cluster (spec-005) and are dead-code-by-archaeology, but removing them is a distinct cleanup with its own parity-test blast radius. Not this spec. (If pursued, file as a follow-on plan — do not leave a TODO.)
- **The other dead config knobs `club_game_spread_max_gap` / `club_game_spread_max_overlap`.** Also dead (read only in `constraints/archived/`), but ClubGameSpread keeps its slack and works; cleaning these is unrelated to the slack-removal/soft-audit goal here. Excluded by design.
- **Adding slack to the stacked alignment atoms.** The convenor explicitly wants alignment to become a fixed hard rule, not a tunable one. Not doing it.
- **`spec-030`/`spec-031`/`spec-032`** registry/group work — separate filed plans; this spec sequences behind them (see Dependencies).

## Dependencies

- `depends_on: spec-032`. spec-030→031→032 form a serial chain that all edit `constraints/registry.py` and `docs/system/CONSTRAINT_INVENTORY.md` (spec-032 also `run.py` and `CLAUDE.md`). This spec edits the same files, so it must start only after spec-032 is `done` and merged, to avoid registry/inventory/run.py merge collisions.
- Within this plan: Unit B depends on Unit A (shared `registry.py`, `config/season_2026.py`, `CONSTRAINT_INVENTORY.md`).

## Risks & blast radius

- **Tester behaviour shift (intended):** removing tester slack makes `_check_club_vs_club_alignment` stricter — previously-passing draws solved with `--slack N` may now report alignment violations. This is the correct, intended outcome (solver was already strict); flag to the convenor so it isn't read as a regression. Touches `analytics/tester.py` only.
- **Dead-path edit trap:** `_club_alignment_hard/_soft` look live (called in `apply_stage_1_hard`/`_2_soft`) but are gated out by `skip_constraints` because the engine key is in no group. An implementer might "fix" their now-strict behaviour thinking it ships — it does not in production, but parity tests may still exercise it. Keep the edit minimal (strip slack only) and run the parity suite.
- **Objective-wiring for Unit B:** the new atom-emitted bye penalty reaches `Maximize(... - penalties)` via `main_staged.py:1279-1284` — ALL `data['penalties']` buckets are included automatically; no separate wiring code is needed. The regen-soft atom uses a `regen_`-prefixed bucket key (`regen_balanced_bye_spacing`); the normal-mode atom must use the unprefixed `'BalancedByeSpacing'` key (matching the PENALTY_WEIGHTS entry and the registry canonical name). Verify post-solve with the printed penalty summary. (review fix — H2/M3: confirmed aggregation path; clarified bucket-key naming.)
- **Penalty weight balance:** a too-high `BalancedByeSpacing` weight could fight `EqualMatchUpSpacing`/`ClubGameSpread`. Start at parity with siblings (100_000) and note the knob; do not auto-tune.
- **`PENALTY_WEIGHTS['ClubVsClubAlignment']` in season_2026.py stays in place** (review note — Low): `season_2026.py:873` has `'ClubVsClubAlignment': 50_000` and line 878 `'ClubVsClubAlignmentField': 0`. Since the dead parity methods (`_club_alignment_hard/_soft`) are kept (Out of Scope), these weights remain valid for parity-reference runs. Do NOT remove them as part of this spec.
- **`ENGINE_HARD_KEYS`/`ENGINE_SOFT_KEYS` in `constraints/stages.py` still contain `'ClubVsClubAlignment'`** (review note — Low): these sets are used by `apply_constraint_set` to route engine vs. non-engine atoms. Since `ClubVsClubAlignment` never appears in any production stage (not in `DEFAULT_STAGES`, not in any group that `resolve_groups` yields for a normal build), these set entries are dead in practice. No change required for this spec — they belong to the "delete dead engine path" cleanup that is explicitly Out of Scope.

## Open Questions

None. (The convenor's directive — remove ClubVsClubAlignment slack; keep + verify EqualMatchUpSpacing/ClubGameSpread soft; tolerant constraints carry a soft analogue — fully determines scope, including the BalancedByeSpacing gap surfaced by the audit.)

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Autonomous: run end-to-end without waiting for the user, except where this hits `blocked`. -->
1. Status must be `ready` (carries a `reviewed:` stamp from /adversarial Mode A). If `review_pending`/`under_review`, let review finish — do not implement. If `blocked`, STOP. **Also confirm `spec-032` is `done` and merged before starting — `depends_on` is unsatisfied otherwise and this plan is not startable.**
2. Stamp `building`, claim `owner`. You are the orchestrator (Opus).
3. Unit A first (own worktree+branch). Then Unit B (own worktree+branch, branched from source AFTER Unit A merges). Delegate each to a subagent (Opus per the per-unit note); run the S2 gates.
4. After each unit, launch `/adversarial` Mode B to verify the diff against this plan's DoD. Route fixes, re-verify. NEVER merge an unverified unit.
5. Merge → push origin → post-merge verify (`run.py generate --year 2026 --simple` builds; penalty report shows the bye bucket after Unit B) → remove worktree. Tick the unit's checkbox.
6. When both units pass: stamp the plan `done`, move it to `docs/todo/done/`, and update `docs/todo/00-dependency-tree.md` (drop this node, note any newly-unblocked spec).
