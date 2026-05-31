<!-- status: ready -->
<!-- reviewed: adversarial Sonnet review 2026-05-30 — fixes applied inline -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-039, spec-040 -->

# spec-041 — Soft-constraint outcome measurement: per-rule 0–100 satisfaction scores + raw metrics

**Spec source:** convenor request (this session) — "we need a way to assess the impact of the soft
constraints to ensure they are working correctly, for instance, the preferred weekends for Maitland
being away from Newcastle. For each soft constraint you should figure out a way to represent how well
its outcome was met … per team, per club breakdowns of probably each rule." The convenor chose
"headline score + raw metrics" for representation.

## Why

Hard constraints have a clean post-hoc pass/fail. Soft constraints don't — a soft rule can be
"mostly met" and the only question worth answering is *how well*. Today that question is largely
unanswerable:

- **Penalty values are never persisted.** At solve time each soft atom registers penalty IntVars in
  `data['penalties'][bucket]`, but post-solve only the game `X` values are extracted — `solver.Value(penalty_var)`
  is never read, and the draw metadata stores only the **count** of penalty vars per bucket, not
  their realised values (verified in `analytics/versioning.py` + `main_staged.py` this session). So a
  soft outcome must be **recomputed from the finished draw's geometry**, not read back.
- **`DrawTester` already computes raw metrics for *most* soft atoms** (the `Violation.metric_value` /
  `affected_clubs` fields and the spec-034 `ViolationBreakdown.soft_pressure` rollup with
  `over_limit`/`total_penalty`/`worst_club`/`worst_value`; note: the `at_limit` key is initialised in
  the default dict but is never incremented by `from_violations` — it is always 0 and must NOT be
  relied upon as a live metric), but several soft atoms are
  **entirely unmeasured** — they have empty `tester_check_methods` (labelled by spec-039 with
  `no_tester_check_reason`): most importantly **`PreferredWeekendsAwayGround`** (the convenor's
  flagship example — Maitland kept away from Maitland Park on the six NRL-Knights home dates in
  `PREFERRED_WEEKENDS`). (review fix — H1: `at_limit` is dead/always-0 in `ViolationBreakdown.from_violations`; removed from the live-fields claim.)
- **There is no normalized score.** Raw metrics ("2 games at 7pm", "spread holes = 3") aren't
  comparable across rules or seasons, and the convenor wants a headline number per rule to trend and
  to drive the report's at-a-glance layout.
- **There is no per-team dimension.** `soft_pressure` rolls up `by_club` and has `worst_club`, but no
  per-team breakdown, which the convenor explicitly wants.

This spec builds one measurement layer that, for every soft atom, computes a raw domain metric AND a
normalized 0–100 satisfaction score from the finished draw, fills the unmeasured gaps (turning
spec-039's `no_tester_check_reason` placeholders into real metrics where measurable), and produces
per-team and per-club rollups — all computed at the draw's faithful effective slack (spec-040), so a
"score" is never penalised for leniency the convenor deliberately granted.

## Definition of Done

### Measurement layer

1. **New module `analytics/soft_outcomes.py` with `measure_soft_outcomes(draw, data) ->
   SoftOutcomeReport`** that, for every soft atom in the live registry (`has_soft_component=True` OR
   membership in the `soft` / `regen_soft` / `symmetry_breakers` groups), returns a `SoftOutcome`.
   Note: `regen_soft` atoms are objective-only (no draw-geometry tester check, run only in regen
   solves) and will each receive `score=None` with their spec-039 `no_tester_check_reason`. The
   measurement layer still enumerates them so they appear in the report — they are not silently skipped.
   (review fix — M3: regen_soft atoms all score None by design; document this upfront to prevent the
   executor attempting to score them via geometry.)
   Per `SoftOutcome`:
   - `atom: str` (canonical name)
   - `raw_metric: dict` — the domain-specific measurement (e.g.
     `{"maitland_games_on_avoid_dates": 0, "possible": 6, "dates": {...}}`,
     `{"games_at_1900": 2}`, `{"spread_holes_by_club_day": [...]}`,
     `{"matchup_spacing_gaps": {...}}`, `{"aligned_weekend_coincidence_rate": 0.83}`,
     `{"bye_spacing_gaps": {...}}`)
   - `score: float` in `[0, 100]` — normalized satisfaction, **100 = ideal outcome, 0 = worst
     plausible outcome**, with a per-atom normalization formula documented in that atom's measurement
     function docstring AND in `docs/system/REGEN_CONSTRAINTS.md` / `CONSTRAINT_INVENTORY.md`.
   - `by_club: dict[str, {raw, score}]` and `by_team: dict[str, {raw, score}]` — the same metric +
     score rolled to each club and team the atom touches (empty where a rule has no per-team meaning,
     e.g. venue-level symmetry breakers).
   - `provenance_note: str` — the effective slack/limit this outcome was measured at (from spec-040).
2. **Effective slack from spec-040 is applied before measurement** — `measure_soft_outcomes` resolves
   the draw's `EffectiveConstraintConfig` and uses it for any limit-relative metric (e.g. a
   spread/overlap score is computed against the slack the draw was actually solved at). A score is
   never computed against a stricter limit than the draw was granted.
3. **Normalization is honest and documented per atom.** Each atom's `score` formula is explicit
   (e.g. PreferredWeekendsAwayGround: `100 * (1 - violations / possible)`; 7pm: `100 * (1 -
   games_at_1900 / total_non_phl_games)` capped at 0; spacing: `100 * (fraction of pairs meeting the
   ideal gap)`). Where "worst plausible" is unbounded, the formula uses a documented denominator
   (total games / total pairs / total club-days), never an arbitrary constant. No score is faked to
   look good; a genuinely-bad outcome reads near 0.

### Closing the unmeasured gaps

4. **New measurement for `PreferredWeekendsAwayGround` (the flagship).** From the finished draw,
   count Maitland (and any away-ground club the atom covers) games at the avoid venue on each
   `PREFERRED_WEEKENDS` `mode='avoid'` date; `raw_metric` lists per-date hits; `score = 100 * (1 -
   total_hits / total_avoid_slots_possible)`; `by_club`/`by_team` break it to the affected club/teams.
   This replaces the spec-039 `no_tester_check_reason` label for this atom with a real metric.
5. **New observability measurement for `AwayClubHomeWeekendsCount`** — per-club realised Sunday-home
   count vs the derived `[min_sundays_home, max_sundays_home]` target range (computed by the same
   `away_club_min/max_sundays_home` helpers the atom uses).
   **IMPORTANT correction (review fix — C1):** `AwayClubHomeWeekendsCount` is a **hard-only** atom
   (`has_soft_component=False` in the registry; the atom adds only `model.Add` clauses and creates no
   penalty IntVars). It is not "the soft side" of anything. Its `tester_check_methods` is
   `['_check_fifty_fifty_home_away']` (non-empty), so spec-039 will NOT label it with
   `no_tester_check_reason`. The outcome metric here is: for each away club, report `realised_sundays`,
   `target_min`, `target_max`, and `in_range: bool`; `score = 100` if in range, `0` if outside (since
   the solver hard-enforces the range, any out-of-range in a finished draw is a solver failure, not a
   soft shortfall). Do NOT attempt to score this as a soft gradient — it is a binary hard-constraint
   check masquerading as a soft measurement. Where the tester already surfaces this via
   `_check_fifty_fifty_home_away` (shared with `AwayClubPerOpponentAndAggregateHomeBalance`),
   the measurement layer derives from that existing computation.
   Any other production soft atom spec-039 labels with `no_tester_check_reason` is enumerated by
   reading the registry's `has_soft_component=True` / `soft`/`regen_soft` group membership at build
   time. Note: all `regen_soft` atoms are **objective-only** with no draw-geometry tester check by
   design (they run only in regen solves, not in production builds); they will all receive
   `score = None` with a `no_tester_check_reason` note from spec-039. Do not attempt to score them.
   Atoms that are genuinely outcome-free (e.g. `SoftLexMatchupOrdering` — an alphabetical
   tie-break with no observable outcome) keep their `no_tester_check_reason` and are reported with
   `score = None` + an explanatory note, NOT a fabricated 100.
6. **Reuse, don't duplicate, the existing tester metrics.** Where `DrawTester` already computes a raw
   metric (`metric_value`/`soft_pressure`) for a soft atom, the measurement layer derives its
   `raw_metric`/`by_club` from that existing computation rather than re-deriving the geometry — one
   source of truth for each metric. The new code is the *scoring + per-team rollup + gap-filling*, not
   a reimplementation of checks spec-034 already shipped.

### Per-team / per-club dimension

7. **`ViolationBreakdown` (or the SoftOutcome layer) gains a `by_team` rollup** alongside the
   existing `by_club`, populated from `Violation.affected_games` → team membership, so every soft
   outcome can be sliced per team. The existing `worst_club`/`worst_value` rollup gains a
   `worst_team`/`worst_team_value` analogue.
   (review fix — M1: `Violation.affected_games` stores game IDs (`str`), not team names; the join
   path is: game_id → `DrawStorage.games` list → `game.team1` / `game.team2`. The implementation
   must accept the `DrawStorage` object to resolve game IDs to teams. Wire this in `measure_soft_outcomes`
   where the draw is already in scope, or pass `draw` into `ViolationBreakdown.from_violations` as
   an optional param.)

### Output + proof

8. **`SoftOutcomeReport` is a dataclass that serialises cleanly to JSON** (for spec-042's sidecar and
   the future UI) and exposes a `summary()` for console/log.
9. **No-mock tests (`tests/test_soft_outcomes.py`) with hand-computed oracle scores** on real-data
   draw fixtures (reuse spec-034 fixtures):
   - *Given* a real draw fixture with a known number of Maitland games on an avoid date (construct the
     fixture so exactly 1 of 6 avoid dates is hit), *when* measured, *then* PreferredWeekendsAwayGround
     `raw_metric.maitland_games_on_avoid_dates == 1`, `score == 83.3` (hand-computed:
     `round(100*(1 - 1/6), 1) = round(83.333..., 1) = 83.3`).
     (review fix — M2: oracle was `round(100*(1-1/6), 1)` without the resolved value; made explicit.)
   - *Given* a draw with exactly 2 games at 19:00, *when* measured, *then* the 7pm metric == 2 and the
     score matches the hand-computed `100*(1 - 2/total_non_phl)`.
   - *Given* a draw solved at slack 3 and a club-spread metric that is compliant at slack 3 but not at
     slack 0, *when* measured (provenance → slack 3 via spec-040), *then* the spread score is the
     high/clean value, NOT the strict-slack penalised one. (Proves DoD-2.)
   - *Given* `SoftLexMatchupOrdering`, *then* `score is None` with the documented note (proves no fabrication).
   - Per-team rollup: a soft outcome's `by_team` sums back to the aggregate raw metric (conservation check).
10. **≥85% coverage on `analytics/soft_outcomes.py`** via the e2e fixtures, honestly reported.

## Implementation units

### Unit A — Measurement layer + scoring for already-metric'd atoms + per-team rollup

- **Files touched:** `analytics/soft_outcomes.py` (new — `measure_soft_outcomes`, `SoftOutcome`,
  `SoftOutcomeReport`, normalization formulas), `analytics/tester.py` (extend `ViolationBreakdown`
  with `by_team` / `worst_team` per DoD-7), `tests/test_soft_outcomes.py` (new — the
  already-measured atoms + per-team conservation + slack-provenance score test).
- **Change summary:** the scoring/rollup layer over the existing tester metrics; reuses spec-034's
  `soft_pressure`, adds the per-team dimension and the 0–100 normalization, applies spec-040's
  effective slack.
- **Depends on:** spec-039 (registry `no_tester_check_reason` + alignment), spec-040 (resolver). Within
  plan: none.
- **Suggested executor:** Opus (normalization design + slack-aware scoring + the per-team rollup
  touch `tester.py` shared structures; subtle).
- **No-mock test outline:** DoD-9 bullets for the 7pm/spacing/spread/per-team cases on real fixtures.

### Unit B — Gap-filling measurements for the unmeasured soft atoms (incl. the Maitland flagship)

- **Files touched:** `analytics/soft_outcomes.py` (add measurement functions for
  `PreferredWeekendsAwayGround`, `AwayClubHomeWeekendsCount` observability (hard-only; see DoD-5
  correction), and any other registry-enumerated soft atom with empty `tester_check_methods` after
  spec-039 labels them), `constraints/registry.py` (where spec-041 supplies a real metric, replace the
  spec-039 `no_tester_check_reason` placeholder with the measurement reference — keep the reason only
  for the genuinely outcome-free atoms), `tests/test_soft_outcomes.py` (extend — the Maitland flagship
  oracle + AwayClubHomeWeekendsCount observability oracle + the `score is None` no-fabrication case).
- **Change summary:** closes the measurement gaps spec-039 documented; turns labels into metrics.
  (review fix — C1 applied: removed "soft side of AwayClubHomeWeekendsCount" framing; it is
  hard-only and its observability metric is a binary in-range check, not a soft score.)
- **Depends on:** Unit A merged (uses the layer + `SoftOutcome` shape).
- **Suggested executor:** Opus (the Maitland-away semantics, the hard-range AwayClub observability, and
  the honest scoring of edge cases need care — erring to Opus).
- **No-mock test outline:** DoD-9 Maitland + AwayClubHomeWeekendsCount observability + SoftLex bullets, on real fixtures with
  hand-computed oracles.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — per soft atom: its raw metric + score formula + per-team
  applicability; mark which previously-unmeasured atoms now have a metric.
- `docs/system/REGEN_CONSTRAINTS.md` — for each `regen_soft` atom, document the outcome metric/score
  used to assess it (the regen-soft analogues are soft by definition and need outcome reporting too).
- `docs/system/TESTING.md` — register `tests/test_soft_outcomes.py`.
- `docs/todo/00-dependency-tree.md` — spec-041 is already listed as a live entry (no change needed).
  (review fix — L1: the dep-tree already has the spec-041 entry authored 2026-05-30.)
- `docs/todo/GOALS.md` — add the spec-041 row.

## Out of scope

- **Rendering / charts / the report file** — spec-042 consumes `SoftOutcomeReport`; this spec
  produces the data only.
- **The alignment gate** — spec-039 (dependency).
- **Slack provenance persistence/resolution** — spec-040 (dependency; consumed here).
- **Re-tuning penalty weights or changing any soft atom's solver behaviour** — measurement only; the
  objective and atoms are untouched. If a measured outcome reveals a weight is wrong, that is a
  separate convenor decision / spec.
- **Persisting realised penalty values at solve time.** Out of scope by design — outcomes are
  recomputed from draw geometry (the established, mock-free approach). (If the convenor later wants
  the solver to dump `solver.Value(penalty_var)`, that is its own spec.)

## Dependencies

- `depends_on: spec-039, spec-040`. Needs spec-039's registry labelling/alignment (to enumerate soft
  atoms and know which are deliberately outcome-free) and spec-040's resolver (to score at the draw's
  effective slack). Both must be `done` and merged before this starts.
- Within this plan: Unit B depends on Unit A.

## Risks & blast radius

- **Normalization is a judgement call.** A bad denominator makes a score meaningless. Mitigation:
  every formula is documented, defended in the atom docstring, hand-oracle-tested, and Mode-B-reviewed;
  "worst plausible" always uses a concrete total (games/pairs/club-days), never a magic constant.
- **Score conservation across per-team rollups.** Per-team slices that don't sum back to the aggregate
  would mislead. Mitigation: DoD-9 conservation test asserts `sum(by_team raw) == aggregate raw` for
  additive metrics; non-additive metrics (rates) document that they don't conserve and are reported as
  weighted, not summed.
- **Touching `analytics/tester.py` shared structures** (the `by_team` rollup) races spec-040's tester
  edits. Mitigation: the `depends_on: spec-040` ordering means spec-040's tester changes land first;
  Unit A branches off the post-spec-040 source.
- **`soft_pressure.at_limit` is currently dead (always 0).** `ViolationBreakdown.from_violations`
  initialises `at_limit` in the default dict but never increments it. Unit A MUST NOT build logic
  that reads `soft_pressure[key]['at_limit']` as a live metric. If `at_limit` semantics are wanted
  (violations exactly at the limit, vs over), that requires a coordinated tester.py change to
  populate it — which is its own targeted fix, not this spec. (review fix — H1 propagated to risks.)
- **Double-counting with the tester.** Re-deriving a metric the tester already computes risks
  divergence. Mitigation: DoD-6 mandates deriving from the existing computation, and a test asserts
  the layer's raw metric for an already-measured atom equals the tester's `metric_value`.

## Open Questions

0 — representation ("headline score + raw metrics"), the flagship example (Maitland away on NRL
weekends), and the per-team/per-club requirement are all settled by the convenor.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Requires an explicit user go-ahead to START (a `ready` plan does not self-start). Once authorised, run the units end-to-end, pausing only on `blocked` or an unrecoverable failure. -->

0. **Do NOT start without an explicit user instruction to implement this plan.** `ready` ≠ "build now".
1. Status must be `ready` (carries a `reviewed:` stamp). Verify `depends_on` (spec-039, spec-040) are
   both `done` and merged on `final-form` — if not, STOP (unsatisfied dependency = cannot start).
2. Only after the user says to implement: stamp `building`, claim `owner`. Orchestrator = Opus.
3. **Unit A** on `spec041-unitA` worktree off the post-spec-040 `final-form`: delegate to Opus.
   Gates: type-check; AST sweep; `pytest tests/test_soft_outcomes.py -v` (the Unit-A subset green);
   ≥85% coverage on new code; `/adversarial` Mode B. Merge → push → tear down.
4. **Unit B** after A on `spec041-unitB`: delegate to Opus. Gates: type-check; full
   `pytest tests/test_soft_outcomes.py -v` green (incl. Maitland flagship + no-fabrication cases);
   AST sweep; `/adversarial` Mode B. Merge → push → tear down.
5. When both pass: stamp `done`, archive to `docs/todo/done/`, update
   `docs/todo/00-dependency-tree.md` (spec-042 now one dependency closer).
