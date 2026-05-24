# Constraint Inventory (Phase 0 / kept current through Phase 7)

Single-source-of-truth table of every registered constraint, what it actually does (extracted from code, not docstring), its severity / slack key, and the atom-target name(s) it splits into during atomization.

Generated against `final-form` and updated through spec-002 + spec-003 + spec-004 + spec-005 + spec-006 + spec-007 + spec-018 + spec-025. The registry currently has **38 entries** (spec-025 added `LockedPairings` as a `tester_only` entry, bringing the count from 37 to 38). spec-018 removed the three venue-sequencing constraints (`NonDefaultHomeGrouping` / `MaitlandHomeGrouping` alias, `AwayAtNonDefaultGrouping` / `AwayAtMaitlandGrouping` alias, and the soft `MaitlandAlternateHomeAway`) — the convenor no longer wants ANY home/away weekend sequencing (back-to-back home weekends and long away runs are both acceptable). The per-club home-weekend totals (`AwayClubHomeWeekendsCount`) and per-pair/aggregate 50/50 balance (`AwayClubPerOpponentAndAggregateHomeBalance`) from spec-004 remain.

Legend
- **Source** is the legacy class location. Parity is asserted between `original.py` and `ai.py` versions (5 + 1 historical bug-fixes documented in `CLAUDE.md`).
- **Severity** matches `ConstraintInfo.severity_level` (1=CRITICAL ... 5=VERY LOW).
- **Slack key** matches `ConstraintInfo.slack_key` (where applicable).
- **Atom target** lists the post-atomization atom name(s). Single-idea constraints stay as one atom (sometimes renamed for the generic-home-ground refactor in Phase 6). Multi-idea constraints split.

## 0. Group membership (spec-023 — SSoT for selection)

Each constraint's `ConstraintInfo.groups` frozenset in `constraints/registry.py`
is the **single source of truth for which groups select it**. A solve applies the
deduped union of the selected `--groups` (see `docs/system/STAGES.md`); selecting
a constraint via two groups applies it once. Derived groups (`severity_1..5`,
`default`/`all`/`production`) are computed from `severity_level` / a non-empty
`groups` set and are never stored. The table below is generated from the registry
— **read the registry, not this table, when in doubt**.

| Canonical name | groups |
|---|---|
| NoDoubleBookingTeams | core, critical_feasibility |
| NoDoubleBookingFields | core, critical_feasibility |
| EqualGamesAndBalanceMatchUps | core, critical_feasibility |
| FiftyFiftyHomeandAway | — (obsolete; no production group) |
| AwayClubHomeWeekendsCount | core, home_away_balance |
| AwayClubPerOpponentAndAggregateHomeBalance | core, home_away_balance |
| PHLAnd2ndAdjacency | core, critical_feasibility |
| PHLAndSecondGradeTimes | — (obsolete; no production group) |
| PHLConcurrencyAtBroadmeadow | core, critical_feasibility |
| EqualMatchUpSpacing | core, critical_feasibility |
| BalancedByeSpacing | core, critical_feasibility |
| ClubDay | — (obsolete legacy combined; no production group) |
| ClubDayParticipation | club_day, core |
| ClubDayIntraClubMatchup | club_day, core |
| ClubDayOpponentMatchup | club_day, core |
| ClubDaySameField | club_day, core |
| ClubDayContiguousSlots | club_day, core |
| TeamConflict | — (no production group) |
| ClubGradeAdjacency | — (obsolete; no production group) |
| SameGradeSameClubNoConcurrency | core, critical_feasibility |
| TeamPairNoConcurrency | soft, soft_optimisation |
| NIHCFillWFBeforeEF | soft, soft_optimisation |
| NIHCFillEFBeforeSF | soft, soft_optimisation |
| ClubVsClubAlignment | — (obsolete; no production group) |
| ClubVsClubStackedWeekends | club_alignment, core |
| ClubVsClubStackedCoLocation | club_alignment, core |
| ClubGameSpread | club_day, core |
| ClubNoConcurrentSlot | core, critical_feasibility |
| ClubFieldConcentration | — (tester-only; no production group) |
| VenueEarliestSlotFill | core, critical_feasibility |
| PreferredTimes | soft, soft_optimisation |
| SoftLexMatchupOrdering | soft, soft_optimisation |
| PreferredWeekendsAwayGround | soft, soft_optimisation |
| PreferredGames | soft, soft_optimisation |
| ForcedGames | — (tester-only; no production group) |
| BlockedGames | — (tester-only; no production group) |
| LockedPairings | — (spec-025 tester-only; no production group) |

Entries with no production group (`tester_only` diagnostics and the obsolete
legacy-only classes `FiftyFiftyHomeandAway`, `PHLAndSecondGradeTimes`, `ClubDay`,
`ClubGradeAdjacency`, `ClubVsClubAlignment`) keep resolving for parity/legacy
lookups but are **not selected by any group** — a `--groups default` run does not
apply them. `severity_N` derived groups still cover every registry entry by
`severity_level` regardless of `groups`.

### Regeneration group (`regen`) — see `REGEN_CONSTRAINTS.md`

The `regen` group is a DERIVED group used **only** by scoped draw regeneration
(`--regen-from`, spec-026). It is defined as `core_hard ∪ regen_soft ∪ soft`
and resolves to 32 constraints. It softens the non-physical rules (replacing them
with 13 `regen_soft` penalty atoms, severity 5, `groups={'regen_soft'}`) so that
a frozen-but-retimed draw does not go INFEASIBLE.

The full reference — core-hard table, regen-soft atom table (bucket / weight /
penalty unit), group definition, dispatch wiring, and engine-key design note — is
in `docs/system/REGEN_CONSTRAINTS.md`. The per-atom engineering-detail rows for
the 13 `regen_soft` atoms are in section 4 of this file (added alongside the
spec-027 implementation).

## 1. Solver-applied constraints

| Canonical name | Source | Actual behavior | Severity | Slack key | Atom target(s) |
|---|---|---|---|---|---|
| NoDoubleBookingTeams | original.py:NoDoubleBookingTeamsConstraint | Sum of team's vars in each (week, team) ≤ 1; skips locked weeks; skips dummy timeslots | 1 | — | NoDoubleBookingTeams |
| NoDoubleBookingFields | original.py:NoDoubleBookingFieldsConstraint | Sum of vars per (day, day_slot, week, field_name) ≤ 1; skips locked weeks | 1 | — | NoDoubleBookingFields |
| EqualGamesAndBalanceMatchUps | original.py:EnsureEqualGamesAndBalanceMatchUps | Per team sum == num_rounds[grade]; per pair sum in [base, base+1] where base = R/(T-1) for even T or R/T for odd; includes dummy slots | 1 | — | EqualGames + BalancedMatchups (two atoms in one cluster) |
| FiftyFiftyHomeandAway | original.py:FiftyFiftyHomeandAway | **OBSOLETE (spec-004).** Superseded by `AwayClubPerOpponentAndAggregateHomeBalance` (per-pair + aggregate balance) + `AwayClubHomeWeekendsCount` (per-club weekend totals, FORCED-Friday aware). The legacy class remains importable in `constraints/archived/` for parity tests but is NOT in any production stage. The §5.1 finding about the aggregate block is now moot — `AwayClubPerOpponentAndAggregateHomeBalance` preserves both blocks intentionally as documented in spec-004. | 1 | — | replaced by AwayClubHomeWeekendsCount + AwayClubPerOpponentAndAggregateHomeBalance |
| AwayClubHomeWeekendsCount | constraints/atoms/away_club_home_weekends_count.py (spec-004) | For each club in `home_field_map`: builds OR-indicators per (week, Friday) and (week, Sunday) over vars at the club's home venue. Enforces THREE sums explicitly: friday_indicators == `phl_forced_friday_count(data, club)`; sunday_indicators == `away_club_required_sundays(data, club)`; all-week indicators == `away_club_total_weekends(data, club)`. Helper module `constraints/atoms/_phl_forced_friday_helper.py` does the FORCED-aware Friday counting (greedy partition handles umbrella + per-pair scopes without double-counting). | 1 | — | AwayClubHomeWeekendsCount (atom, new) |
| AwayClubPerOpponentAndAggregateHomeBalance | constraints/atoms/away_club_home_balance.py (spec-004) | For each team in each away-based club: (a) per-opponent home_games * 2 within total ± 1; (b) aggregate per-team agg_home * 2 within agg_total ± 1. Both blocks intentional — `[floor(t/2), ceil(t/2)]` window for both per-pair and per-team-aggregate. Reuses tester `_check_fifty_fifty_home_away` for violation reporting. | 1 | — | AwayClubPerOpponentAndAggregateHomeBalance (atom, new) |
| ~~MaitlandHomeGrouping / NonDefaultHomeGrouping~~ | _removed (spec-018)_ | **REMOVED (spec-018).** Was a hard sliding-window ban on consecutive home weekends for away-based clubs (+ per-week home/away imbalance soft penalty), plus the folded `MaxMaitlandHomeWeekends` per-venue weekend cap. The convenor no longer wants any sequencing of home/away weekends — back-to-back home weekends are fine. Registry entry, engine method, adjuster, severity entry, tester check, slack key (`MaitlandHomeGrouping`), and config keys (`maitland_max_consecutive_home`) all deleted. | — | — | — |
| PHLAndSecondGradeAdjacency | original.py:PHLAndSecondGradeAdjacency | Per (club, 2nd-team, week, day): for each PHL game (time, location), sum of PHL var + 2nd-grade vars within ±180 min that satisfy "same loc when within window OR different loc when outside window" ≤ 1 | 1 | — | **REWRITTEN as `PHLAnd2ndAdjacency` atom (spec-014).** New rule *forces* back-to-back same-field at one venue, or ≥150-min real-minute start gap across venues (spec-030: 180→150) — the legacy ±180 forbid window never forced adjacency. Legacy class kept in `constraints/archived/` for parity. |
| PHLAndSecondGradeTimes | original.py:PHLAndSecondGradeTimes (lines 214-372 — multi-idea, contains user-flagged HACK) | (a) PHL no-concurrent at Broadmeadow per (week, day_slot, location); (b) PHL+same-club 2nd no-concurrent per (week, day, location, day_slot, club, 2nd-team); (c) ≤max_friday_broadmeadow PHL Fridays at NIHC (HACK: minus locked); (d) sum Friday Gosford PHL == gosford_friday_games (HACK: minus locked); (e) sum Friday Maitland PHL == maitland_friday_games (HACK: minus locked); (f) Gosford Friday rounds {2,4,5,9,10} sum == 1 each; (g) PHL each team plays in round 1 (skipped if locked); (h) PHL preferred-dates penalty | 1 | — | (Phase 3a partially shipped @ `1956608`, post-retraction reduces to 4 atoms.) **Atoms:** PHLConcurrencyAtBroadmeadow + PHLAnd2ndConcurrencyAtBroadmeadow (deleted spec-030 — subsumed by PHLAnd2ndAdjacency) + PHLRoundOnePlay + PreferredDates. **Expressed as FORCED_GAMES entries instead** (see `docs/FORCED_GAMES_AS_COUNT_RULES.md`): items (c)/(d)/(e) (per-venue Friday counts) and (f) (Gosford rounds — already covered by per-round FORCED entries in season_2026.py). The HACK locked-week count adjustments disappear because each FORCED entry is per-season and the count is the season target, not a runtime computation. |
| EqualMatchUpSpacing | original.py:EqualMatchUpSpacingConstraint | HARD: for pair p with rounds (r1, r2), forbid `gap <= S` ⇒ sum(p_r1) + sum(p_r2) ≤ 1 (spec-008 Part A: "S free rounds between meetings", off-by-one fix). `S = effective_spacing(T, base_slack, config_slack)` from `constraints/atoms/_spacing.py`; default `S = ideal_gap(T) = legacy_min_gap(T) - 1` so the physical schedule at default slack is unchanged. Slack subtracts from S; each FORCED meeting tightens via the Phase-4 adjuster (negative net slack). SOFT: sliding window of size `space=T-2`, penalize sum-1 when >1. HACK: when locked_weeks set, only applies to PHL/2nd. **spec-017: this atom now lives in `critical_feasibility` (was only in the soft_only `soft_optimisation` stage, so the HARD part never applied in production); both HARD and SOFT parts now apply — `--slack EqualMatchUpSpacingConstraint N` still loosens S.** | 1 | EqualMatchUpSpacingConstraint | EqualMatchUpSpacing (Phase 4 adds adjuster for FORCED meetings → reduces effective rounds; spec-008 shifts gap semantics to "rounds between meetings"; spec-017 promotes to hard-in-production) |
| BalancedByeSpacing | constraints/atoms/balanced_bye_spacing.py (spec-008 Part B) | HARD: for each team in each grade with `byes_per_team >= 2`, per-round bye-indicator BoolVar `B_t_r := 1 - sum(team_vars_r)`. Forbid `B_t_r1 + B_t_r2 <= 1` for every pair of rounds with `r2 - r1 <= S`, where `S = max(0, ideal_bye_gap(R, byes) - base_slack - config_slack)` and `ideal_bye_gap(R, b) = max(0, R//b - 1)`. Skips (locked, locked) pairs. Own slack key `BalancedByeSpacing` and own base-slack `CONSTRAINT_DEFAULTS['bye_spacing_base_slack']` so convenor can loosen byes independently of matchup spacing. | 2 | BalancedByeSpacing | BalancedByeSpacing (atom, new) |
| ClubDay | original.py:ClubDayConstraint (lines 632-750 — multi-idea) | For each entry in CLUB_DAYS (parsed via `normalize_club_day` → date + optional opponent): (a) every team in club plays on that date; (b) if opponent set, force ≥min(host_grade_count, opp_grade_count) cross-club games per grade; (c) if no opponent / opponent missing grade, force intra-club derbies = #host_grade // 2; (d) all club games on same field; (e) timeslots are contiguous via "if middle slot empty, prior + following ≤ 1" | 2 | — | **Phase 3b ✅** atoms shipped: ClubDayParticipation + ClubDayIntraClubMatchup + ClubDayOpponentMatchup + ClubDaySameField + ClubDayContiguousSlots |
| ~~AwayAtMaitlandGrouping / AwayAtNonDefaultGrouping~~ | _removed (spec-018)_ | **REMOVED (spec-018).** Was a hard cap on how many away clubs visit a non-default venue in one weekend (+ soft penalty). The convenor no longer wants any clustering cap on away-club variety per weekend. Registry entry, engine method, adjuster (`away_at_maitland_grouping_adjuster`), severity entry, tester check, slack key (`AwayAtMaitlandGrouping`), and config key (`away_maitland_max_clubs`) all deleted. | — | — | — |
| TeamConflict | original.py:TeamConflictConstraint | For each (team1, team2) in `team_conflicts`: per (week, day_slot), sum vars involving either ≤ 1; skips locked weeks | 2 | — | TeamConflict |
| ClubGradeAdjacency | original.py:ClubGradeAdjacencyConstraint | **OBSOLETE (spec-007).** Hard portion (duplicate same-club-same-grade) split out as `SameGradeSameClubNoConcurrency` (severity 1). Soft portion (adjacent-grade penalty) **REMOVED ENTIRELY** — convenor experience was that it was over-restrictive. Registry entry kept so solver class names still resolve and the tester can keep emitting violations under the same name for the same-grade-same-club case. | 3 | — | `SameGradeSameClubNoConcurrency` (hard) + `TeamPairNoConcurrency` (new soft, per-pair convenor list) |
| SameGradeSameClubNoConcurrency | atoms/same_grade_same_club_no_concurrency.py | HARD: per (club, grade, week, day_slot) bucket, sum of cross-club games involving any duplicate-set team from that club ≤ 1. Intra-club derbies (single shared variable) are excluded. | 1 | — | SameGradeSameClubNoConcurrency (atom_group: ClubGradeAdjacency) |
| TeamPairNoConcurrency | atoms/team_pair_no_concurrency.py | SOFT: reads `constraint_defaults['TEAM_PAIR_NO_CONCURRENCY']` of (team_a, team_b) or (team_a, team_b, weight_multiplier) entries. Per (week, day_slot), pen = max(0, sum(vars_team_a) + sum(vars_team_b) - 1), scaled by multiplier. Base penalty weight from `PENALTY_WEIGHTS['TeamPairNoConcurrency']` (default 1000). | 3 | — | TeamPairNoConcurrency |
| ClubVsClubAlignment | original.py:ClubVsClubAlignment (lines 972-1198 — multi-idea) | Lower-grade block (3rd-6th, sorted by per_team_games asc): for each grade-pair X,Y where Y > X by num_games — for each club_pair seen in both — coincide vars per round; HARD ≥ num_games - slack; HARD ≤ 2 fields when coinciding; SOFT field excess penalty; SOFT coincide deficit penalty. PHL/2nd Sunday block: same coincide algorithm but additionally requires "back-to-back same field" (∃ pair with same field name and abs(slot1-slot2)==1) when coinciding | 3 | ClubVsClubAlignment | **OBSOLETE (spec-005).** Phase-3c atoms (`ClubVsClubCoincidence`, `ClubVsClubFieldLimit`, `ClubVsClubDeficitPenalty`, `PHLAnd2ndBackToBackSameField`) remain in the registry + on disk as parity reference but are NOT in any production stage. Replaced by `ClubVsClubStackedWeekends` + `ClubVsClubStackedCoLocation` (precise stacking + co-location, PHL-Sunday-budget-aware). |
| ClubVsClubStackedWeekends | constraints/atoms/club_vs_club_stacked_weekends.py (spec-005) | For each unordered club pair (A,B): builds per-(grade, week) `play` indicator (OR of pair's Sunday vars). Pins `sum_w play[g, w] == sunday_budget(g)` per grade (PHL budget = total - FORCED Fridays via `phl_forced_friday_meetings`). Enforces consec-pair implication chain in descending-count order: `play[g_{k+1}, w] <= play[g_k, w]` for every week → nested-superset structure across weeks. Helper-pool key `(STACK_PLAY_PREFIX, pair, grade, week)` shared with `ClubVsClubStackedCoLocation`. Raises `ValueError` if a grade's Sunday budget exceeds available weeks. **spec-019: per-pair FORCED-Friday budget reduction is regression-tested in `tests/atoms/test_cvc_stacked_friday_aware.py` — incl. the A-shared isolation case (a FORCED Friday for pair (A,B) must not reduce pair (A,C)).** | 3 | ClubVsClubAlignment | ClubVsClubStackedWeekends (atom, new) |
| ClubVsClubStackedCoLocation | constraints/atoms/club_vs_club_stacked_co_location.py (spec-005) | For each (pair, week) where the 2nd-ranked-by-count grade plays (≥ 2 grades active): (a) at most one field used (`sum field_used <= 1`) OnlyEnforceIf `stack_active`; (b) contiguous day_slots via "empty middle slot ⇒ prior+next ≤ 1" reified gate (`gate = stack_active AND mid.Not()`). Generalises the legacy PHL/2nd back-to-back rule to every stacked grade. Reads `play` indicators from registry — MUST run after `ClubVsClubStackedWeekends` (raises `RuntimeError` otherwise). Helper kinds `cvc_stack_field_used`, `cvc_stack_slot_used`, `cvc_stack_active` declared parallel to the `club_day_*` family (different key shapes — `(pair, week, ...)` vs `(club, ...)` — no collision). | 3 | ClubVsClubAlignment | ClubVsClubStackedCoLocation (atom, new) |
| ClubGameSpread | original.py:ClubGameSpread | **spec-021 rewrite.** Per (club, week, day): `slot_used` indicators (shared `_contiguity`) + prefix/suffix "used-before/after" channels + per-slot hole indicator. HARD: holes ≤ `max(0, min(1, n_games-3))` + slack (≤3 games → 0 holes; ≥4 → 1). SOFT: penalty = each residual hole. NO range/min/max/spread/overlap IntVars (only BoolVars). Lower no-double-up bound EXTRACTED to `ClubNoConcurrentSlot`. Now in the **hard** `club_day` stage (was dead in soft_only). | 3 | ClubGameSpread | ClubGameSpread |
| ClubNoConcurrentSlot | constraints/atoms/club_no_concurrent_slot.py (spec-021) | Extracted from ClubGameSpread's lower no-double-up bound (concurrency, not contiguity). Per (club, week, day_slot, location): club games ≤ `cap = max(1, ceil(club_team_count / no_field_slots[location]))`. Capacity-aware — small venues (Central Coast = 2 times) allow forced double-ups. HARD, severity 2, `critical_feasibility`. | 2 | — | ClubNoConcurrentSlot (atom, new) |
| VenueEarliestSlotFill | constraints/atoms/venue_earliest_slot_fill.py (spec-021, replaces EnsureBestTimeslotChoices) | Per (week, date, location): combined-field `slot_used` indicator (OR across fields per slot) + `enforce_monotone_fill` (shared `_contiguity`): use slot s ⇒ use slot s-1 → games pack into the EARLIEST slots (no gaps + earliest start). NO AddDivisionEquality/nts/BROADMEADOW_MAX_SLOTS IntVars. Earliest-packing structurally avoids 7pm → no 7pm penalty; WF order owned by NIHCFillWFBeforeEF. **HARD, severity 2, `critical_feasibility`** (was soft_only, hard part dead). | 2 | — | VenueEarliestSlotFill (atom, new) |
| PreferredTimes | original.py:PreferredTimesConstraint (dispatched via `unified.py::_preferred_times`) | Normalize PREFERENCE_NO_PLAY (legacy + 2026 structured); for each (entry, club, club_teams, restriction): match X keys via two key orderings; soft penalty = the var when match. **spec-012:** the normaliser (`utils.normalize_preference_no_play`) now passes `time`, `day`, `day_slot`, `week`, `field_name`, `field_location` from the entry into the emitted restriction dict; entries WITHOUT a `date`/`dates` key emit one restriction without `'date'` that the dispatcher applies across every week. The dispatcher only locked-week-short-circuits when `'date'` is present. Entries with neither a date nor any matchable filter are silently a no-op. | 5 | — | PreferredTimes |
| ~~MaitlandAlternateHomeAway~~ | _removed (spec-018)_ | **REMOVED (spec-018).** Was a soft penalty (spec-012) pushing H-A-H-A alternation across Maitland weekends. The convenor no longer wants any home/away sequencing. Atom file, registry entry, and penalty weight (`maitland_alternate_home_away`) all deleted. | — | — | — |
| SoftLexMatchupOrdering | constraints/atoms/soft_lex_matchup_ordering.py (spec-002) | Soft tie-break: for each grade, sort pairs alphabetically (team1, team2). Assign rank r (0-indexed). Penalty = weight * r * X[key] per var. Encourages alphabetically-earlier pairs in earlier rounds. Pure objective; never hard constraint. PENALTY_WEIGHTS['soft_lex_ordering'] defaults to 1 | 5 | — | SoftLexMatchupOrdering (atom, new) |
| NIHCFillWFBeforeEF | constraints/atoms/nihc_fill_wf_before_ef.py (spec-003; SOFT per spec-016) | Per (date, day_slot) at NIHC where BOTH WF and EF have ≥1 decision var: channel `wf_used`/`ef_used`, then add a soft penalty term = `ef_used AND NOT wf_used` to the shared `nihc_fill_order` bucket (weight `PENALTY_WEIGHTS['nihc_fill_order']`=5). No hard implication. Skips buckets where either field has no variables. | 5 | — | NIHCFillWFBeforeEF (atom — atom_group `NIHCFieldFillOrder`; SOFT symmetry-breaker per spec-016) |
| NIHCFillEFBeforeSF | constraints/atoms/nihc_fill_ef_before_sf.py (spec-003; SOFT per spec-016) | Per (date, day_slot) at NIHC where BOTH EF and SF have ≥1 decision var: soft penalty term = `sf_used AND NOT ef_used` into the shared `nihc_fill_order` bucket (no hard implication). Together with `NIHCFillWFBeforeEF` makes WF→EF→SF the cheapest fill order. Shares the `nihc_field_used` helper kind and the `nihc_fill_order` bucket with the WF/EF atom. | 5 | — | NIHCFillEFBeforeSF (atom — atom_group `NIHCFieldFillOrder`; SOFT symmetry-breaker per spec-016) |
| PreferredWeekendsAwayGround | constraints/atoms/preferred_weekends_away_ground.py (spec-006) | Soft penalty for scheduling (avoid) or missing (prefer) games at a specific away venue on specific dates. Reads `data['preferred_weekends']` from season config. avoid: penalty = weight × games_at_venue_on_date. prefer: penalty = weight × max(0, target_count - games_at_venue_on_date). Never hard. PENALTY_WEIGHTS['preferred_weekends_away_ground'] defaults to 1000. 2026: 6 NRL-Knights-at-Maitland dates as avoid entries. | 5 | — | PreferredWeekendsAwayGround (atom, new) |

## 1b. Phase-6 generic home-grouping aliases — REMOVED (spec-018)

The Phase-6 generic `NonDefaultHomeGrouping` / `AwayAtNonDefaultGrouping`
constraints and their `MaitlandHomeGrouping` / `AwayAtMaitlandGrouping`
back-compat aliases were **deleted in spec-018** along with their solver
classes, registry entries, slack keys, and `AWAY_VENUE_RULES` consumption
for these two rules. The convenor no longer wants any sequencing of
home/away weekends (consecutive home weekends and long away runs are both
fine). The only home/away rules that remain are the spec-004 atoms
`AwayClubHomeWeekendsCount` (per-club home-weekend totals) and
`AwayClubPerOpponentAndAggregateHomeBalance` (per-pair + aggregate 50/50).

## 2. Tester-only diagnostics

| Canonical name | Source | Actual behavior | Severity |
|---|---|---|---|
| ClubFieldConcentration | tester only | Reports clubs concentrated on a small number of fields (no solver enforcement) | 3 |
| ForcedGames | enforced via `generate_X` (variable-elimination) | Verifies each FORCED entry's scope sum matches `count` and `constraint` (default sum==1) | 1 |
| BlockedGames | enforced via `generate_X` (variable-elimination) | Verifies no game key matches a BLOCKED scope+team-matcher. **spec-001 exemption:** a BLOCKED entry whose source dict carries `'perennial': True` (e.g. every entry in `PERENNIAL_BLOCKED_GAMES` in `config/defaults.py`) is *overridable* — a variable matched by a perennial scope is kept iff any `FORCED_GAMES` entry also matches it. Vars matched by ANY non-perennial BLOCKED scope are always eliminated, even when FORCED matches. Implementation: `utils._build_blocked_game_rules_with_perennial` + `_matching_blocked_scope_keys` + the `matched_block_scopes`/`all_perennial` branch in `generate_X`. Validator `_check_forced_game_feasibility` applies the same rule when counting surviving vars for each FORCED entry. | 1 |
| LockedPairings | `analytics/tester.py::_check_locked_pairings` (registered in `constraints/registry.py` as `tester_only`, spec-025) | Verifies each `LOCKED_PAIRINGS` entry is satisfied: the pinned pairing appears in the draw on its specified date. Analogous to `ForcedGames` but scoped to date-pin entries only (time/slot/field are free). Draw metadata gains a `locked_pairing_outcomes` section with per-pin matched-var count, resolved time/slot/field, and `satisfied: true/false`. | 1 |

## 3. Atomization summary (count)

| Cluster | Legacy classes | Atoms after split | Net change |
|---|---|---|---|
| PHLAndSecondGradeTimes | 1 | 1 — `PHLConcurrencyAtBroadmeadow`. (`PHLAnd2ndConcurrencyAtBroadmeadow` deleted spec-030 — subsumed by `PHLAnd2ndAdjacency`; `PHLRoundOnePlay` deleted spec-010; `GosfordFridayRoundsForced` deleted spec-015 — both expressed as FORCED_GAMES entries, see `docs/system/FORCED_GAMES_AS_COUNT_RULES.md`; `PreferredDates` deleted spec-020 — replaced by the generic `PreferredGames` soft atom, no longer in this cluster) | +1 |
| ClubDayConstraint | 1 | 5 | +4 |
| ClubVsClubAlignment | 1 | 4 (Phase 3c — OBSOLETE per spec-005, parity reference only) | +3 |
| ClubVsClubStackedAlignment (spec-005) | 0 (new cluster — replaces the 4 Phase-3c atoms in `DEFAULT_STAGES`) | 2 — `ClubVsClubStackedWeekends`, `ClubVsClubStackedCoLocation` | +2 |
| ~~MaitlandHomeGrouping + MaxMaitlandHomeWeekends~~ | 2 | 0 (**REMOVED spec-018** — was 1 generic `NonDefaultHomeGrouping`) | -2 |
| ~~AwayAtMaitlandGrouping~~ | 1 | 0 (**REMOVED spec-018** — was 1 generic `AwayAtNonDefaultGrouping`) | -1 |
| ~~MaitlandAlternateHomeAway (spec-012)~~ | 0 | 0 (**REMOVED spec-018** — soft H-A-H-A atom deleted) | 0 |
| ClubGradeAdjacency (spec-007) | 1 | 1 hard (`SameGradeSameClubNoConcurrency`) + 1 new soft (`TeamPairNoConcurrency`). Adjacent-grade soft rule removed entirely (net +1 atom vs cluster, +1 from the new convenor-list atom). | +1 |
| All other entries | 13 | 13 (1:1, with renames in Phase 6) | 0 |
| SoftLexMatchupOrdering (spec-002) | 0 (new atom, no legacy class) | 1 — `SoftLexMatchupOrdering` | +1 |
| NIHCFieldFillOrder (spec-003) | 0 (new atoms, no legacy class; replaces a review-only perennial rule) | 2 — `NIHCFillWFBeforeEF`, `NIHCFillEFBeforeSF` | +2 |
| PreferredWeekendsAwayGround (spec-006) | 0 (new atom, no legacy class) | 1 — `PreferredWeekendsAwayGround` | +1 |
| PreferredGames (spec-020) | 0 (new atom; replaces the deleted `PreferredDates` 1:1) | 1 — `PreferredGames` (generic soft FORCED analogue) | 0 |
| **Total** | **18 solver + 3 tester-only** | **29 solver + 4 tester-only** (was 32 before spec-018 removed the 3 venue-sequencing atoms; spec-020 swaps `PreferredDates`→`PreferredGames`, net 0; spec-025 adds `LockedPairings` tester-only entry) | **+11** |

Registry entry count (`len(CONSTRAINT_REGISTRY)`) is now **38** (the test assert at `tests/test_constraint_registry.py` is `== 38`). History: was 43 before spec-018 deleted the 5 home/away-grouping entries (`NonDefaultHomeGrouping`, `MaitlandHomeGrouping` alias, `AwayAtNonDefaultGrouping`, `AwayAtMaitlandGrouping` alias, `MaitlandAlternateHomeAway`); spec-020 deleted `PreferredDates` and added `PreferredGames` (net 0); spec-021 replaced `EnsureBestTimeslotChoices` with `VenueEarliestSlotFill` (net 0) and added `ClubNoConcurrentSlot` (39); spec-024 then deleted `MaximiseClubsPerTimeslotBroadmeadow` and `MinimiseClubsOnAFieldBroadmeadow` (net −2 → 37); spec-025 added `LockedPairings` as a `tester_only` entry (+1 → **38**). spec-023 added the `groups` field to every entry but no new entries (net 0).

## 4. Per-atom engineering detail

Engineering-level table for every registered atom + non-atomised legacy constraint. The columns answer the questions an engineer asks before touching code: does it skip forced vars? does it skip locked weeks? what slack key does it read? what helper vars does it consume?

**Rule:** when an atom is added or changed, this table gets a row in the same commit. A missing or `?` row is treated as a bug (see `docs/system/README.md`).

| Canonical name | File | Forced-games | Locked-week skip | Dummy-key skip | Slack key | Helper vars | Caveats |
|---|---|---|---|---|---|---|---|
| **Atomised (Phase 3) — PHL/2nd cluster** | | | | | | | |
| `PHLConcurrencyAtBroadmeadow` | `constraints/atoms/phl_concurrency.py` | n/a | yes | yes | — | — | Broadmeadow-only; prevents concurrent PHL games per (week, day_slot) |
| `PHLAnd2ndConcurrencyAtBroadmeadow` | _deleted (spec-030)_ | n/a | n/a | n/a | — | — | **DELETED (spec-030).** Same-club PHL/2nd same-Broadmeadow-slot was a strict subset of `PHLAnd2ndAdjacency`'s same-venue branch (only same-field adjacent slots allowed), so the same-slot case was already forbidden. Atom file + registry entry removed. |
| `GosfordFridayRoundsForced` | _deleted (spec-015)_ | n/a | n/a | n/a | — | — | **DELETED (spec-015).** Per-round `sum == 1` Gosford-Friday rule is now a generic `FORCED_GAMES` count entry (scope + count + constraint type) in the season config — see `FORCED_GAMES_AS_COUNT_RULES.md`. Atom file, registry entry, and `CONSTRAINT_DEFAULTS['gosford_friday_rounds']` removed. |
| `PHLRoundOnePlay` | `constraints/atoms/phl_round_one_play.py` | n/a | yes | yes | — | — | **OBSOLETE (spec-010).** Removed from `critical_feasibility` stage and `_PHL_HARD_ATOMS`; `solver_class_names` emptied in registry. File kept on disk as parity reference. Convenor uses `FORCED_GAMES` entries to express deliberate round-1 placement when needed. |
| `PreferredDates` | _deleted (spec-020)_ | n/a | n/a | n/a | — | — | **DELETED (spec-020).** The narrow PHL-only `\|sum − 1\|`-on-a-date soft constraint is now expressed as a `PREFERRED_GAMES` config entry handled by the generic `PreferredGames` soft atom (below). `PHL_PREFERENCES` / `phl_preferences` and `PENALTY_WEIGHTS['phl_preferences']` removed. |
| **Atomised (Phase 3) — ClubDay cluster** | | | | | | | |
| `ClubDayParticipation` | `constraints/atoms/club_day_participation.py` | excluded | yes | yes | — | — | Skips locked-week club days via `parse_club_day_entries` |
| `ClubDayIntraClubMatchup` | `constraints/atoms/club_day_intra_club_matchup.py` | excluded | yes | yes | — | — | Derby logic when opponent undefined / empty for grade |
| `ClubDayOpponentMatchup` | `constraints/atoms/club_day_opponent_matchup.py` | excluded | yes | yes | — | — | Cross-club matchups per grade when opponent set |
| `ClubDaySameField` | `constraints/atoms/club_day_same_field.py` | excluded | yes | yes | `club_day_field_used` | `club_day_field_used` | Channels field indicators via registry |
| `ClubDayContiguousSlots` | `constraints/atoms/club_day_contiguous_slots.py` | excluded | yes | yes | `club_day_slot_used` | `club_day_slot_used` | spec-021: now calls the shared `_contiguity` `slot_used_indicators` + `enforce_no_gaps` (behaviour-identical; same pool key). Requires ≥3 slots to apply |
| **Atomised (Phase 3) — ClubVsClub cluster** *(OBSOLETE — replaced by spec-005, kept as parity reference)* | | | | | | | |
| `ClubVsClubCoincidence` | `constraints/atoms/club_vs_club_coincidence.py` | excluded (via adjuster) | yes | yes | `ClubVsClubAlignment` | `cvc_coincide` | **OBSOLETE (spec-005).** Not in any production stage. Adjuster reduces expected counts for FORCED off-Sunday / BLOCKED on-Sunday. |
| `ClubVsClubFieldLimit` | `constraints/atoms/club_vs_club_field_limit.py` | n/a (reads coincide vars) | yes | yes | `ClubVsClubAlignment` | `cvc_coincide` (reads) | **OBSOLETE (spec-005).** Soft penalty `ClubVsClubAlignmentField`=50000; hard ≤2 fields. Co-location atom now enforces `sum(field_used) <= 1` on every stacked weekend (gated by `stack_active`). |
| `ClubVsClubDeficitPenalty` | `constraints/atoms/club_vs_club_deficit_penalty.py` | n/a (reads coincide vars) | yes | yes | — | `cvc_coincide` (reads) | **OBSOLETE (spec-005).** Soft penalty into `ClubVsClubAlignment` bucket=100000. Replaced by HARD `sum == budget` in `ClubVsClubStackedWeekends` — deficits are structurally impossible at solve time. |
| `PHLAnd2ndBackToBackSameField` | `constraints/atoms/phl_2nd_back_to_back.py` | n/a | yes | yes | `ClubVsClubAlignment` | `cvc_phl_btb_coincide` | **OBSOLETE (spec-005).** PHL/2nd Sunday back-to-back same-field rule generalised to every stacked grade pair via `ClubVsClubStackedCoLocation` (not just PHL/2nd). |
| **Atomised (spec-005) — ClubVsClubStackedAlignment cluster** | | | | | | | |
| `ClubVsClubStackedWeekends` | `constraints/atoms/club_vs_club_stacked_weekends.py` | included (FORCED-aware via `phl_forced_friday_meetings`) | yes | yes | `ClubVsClubAlignment` | `cvc_stack_play` | HARD. Per (pair, grade) pins `sum_w play[g, w] == sunday_budget`. PHL Sunday budget subtracts FORCED Fridays. Per-grade `R` from `num_rounds[grade]` (per-grade override, falls back to `num_rounds['max']`); `R // (T-1)` for even T or `R // T` for odd T. Multi-team-per-club-per-grade handled by `per_pair_grade_matchup_counts` (distinct matchups, not 1). Implication chain `play[g_lower, w] <= play[g_higher, w]` in descending-Sunday-budget order (alphabetical tie-break) → strict nested-superset structure. Raises `ValueError` if Sunday budget > available weeks. |
| `ClubVsClubStackedCoLocation` | `constraints/atoms/club_vs_club_stacked_co_location.py` | included | yes | yes | `ClubVsClubAlignment` | `cvc_stack_play` (reads), `cvc_stack_field_used`, `cvc_stack_slot_used`, `cvc_stack_active` | HARD. Gate `stack_active = play[2nd-ranked-grade, w]` — by the implication chain this is exactly "≥ 2 grades active." When gated on: (1) `sum(field_used) <= 1` (single-field rule); (2) reified contiguity gate `(stack_active AND mid_slot.Not())` ⇒ `prior_slot + next_slot <= 1` (no internal slot gaps). MUST run AFTER `ClubVsClubStackedWeekends` — raises `RuntimeError` otherwise. |
| **Non-atomised (still routed through `constraints/archived/`)** | | | | | | | |
| `NoDoubleBookingTeams` | `constraints/archived/original.py` | included | yes | yes | — | — | Fundamental; no forced exclusion needed |
| `NoDoubleBookingFields` | `constraints/archived/original.py` | included | yes | yes | — | — | Uses `date`, not `week`, for slot comparisons |
| `EqualGamesAndBalanceMatchUps` | `constraints/archived/original.py` | included | yes | yes | — | — | Includes dummy vars (overflow); two distinct tester violation types |
| `FiftyFiftyHomeandAway` | `constraints/archived/original.py` | included | yes | yes | — | — | **OBSOLETE (spec-004).** Replaced by `AwayClubHomeWeekendsCount` + `AwayClubPerOpponentAndAggregateHomeBalance`. Legacy class remains importable for parity but is NOT wired into any production stage. |
| **Atomised (spec-004) — away-club home/away** | | | | | | | |
| `AwayClubHomeWeekendsCount` | `constraints/atoms/away_club_home_weekends_count.py` | included (FORCED-aware via helper) | yes | yes | — | — | Hard. Reads `phl_forced_friday_count`, `away_club_required_sundays`, `away_club_total_weekends` from `_phl_forced_friday_helper`. Enforces THREE sums per club: Friday-home == FORCED-fri count; Sunday-home == required Sundays; all-home == total weekends. Locked-week contribution subtracted from each target. Raises ValueError if a target > 0 but no candidate vars exist. |
| `AwayClubPerOpponentAndAggregateHomeBalance` | `constraints/atoms/away_club_home_balance.py` | included | yes | n/a (no day filter) | — | — | Hard. Per-opponent: `home * 2 in [total-1, total+1]`. Aggregate: `agg_home * 2 in [agg_total-1, agg_total+1]`. Iterates `home_field_map`; intra-club derbies excluded. Tester check: `_check_fifty_fifty_home_away` (shared). |
| `PHLAnd2ndAdjacency` | `constraints/atoms/phl_2nd_adjacency.py` | included | yes | yes | — | — | **spec-014 rewrite.** Per (club, week, day) with BOTH grades: same venue ⇒ same field + adjacent day_slots; different venue ⇒ start times ≥ `phl_2nd_cross_venue_min_minutes` (150, real minutes; spec-030: 180→150) apart. Zero added decision vars (forbid infeasible candidate pairs). Tester: `_check_phl_2nd_adjacency`. Replaces the legacy ±180-min *forbid* window in archived `original.py` which never forced adjacency. |
| ~~`NonDefaultHomeGrouping`~~ | _removed (spec-018)_ | — | — | — | — | — | **REMOVED (spec-018).** Consecutive-home-weekend spacing rule deleted; adjuster `maitland_home_grouping_adjuster`, slack key, and config key gone. `AwayClubHomeWeekendsCount` (spec-004) still enforces per-club home-weekend TOTALS — only the SPACING dimension was dropped. |
| ~~`AwayAtNonDefaultGrouping`~~ | _removed (spec-018)_ | — | — | — | — | — | **REMOVED (spec-018).** Away-clubs-per-weekend cap deleted; adjuster `away_at_maitland_grouping_adjuster`, slack key, and config key gone. |
| `TeamConflict` | `constraints/archived/original.py` | included | yes | yes | — | — | Per (team1, team2) in `team_conflicts` |
| `ClubGradeAdjacency` | `constraints/archived/original.py` | n/a | n/a | n/a | — | — | **OBSOLETE (spec-007).** Not dispatched by the engine or by any stage. Registry entry kept for back-compat lookups; the tester still emits violations under this name for the same-grade-same-club case only. |
| `SameGradeSameClubNoConcurrency` | `constraints/atoms/same_grade_same_club_no_concurrency.py` | included | yes | yes | — | — | **spec-007.** Hard severity 1. Dispatched via non-engine fallback in `constraints/stages.py` (`_resolve_solver_class` now searches `constraints.atoms`). Intra-club derbies are excluded (single shared variable). |
| `TeamPairNoConcurrency` | `constraints/atoms/team_pair_no_concurrency.py` | included | yes | yes | — | — | **spec-007.** Soft severity 3. Reads `constraint_defaults['TEAM_PAIR_NO_CONCURRENCY']`. Penalty bucket `TeamPairNoConcurrency`; default base weight 1000; per-entry weight multiplier scales the penalty IntVar. |
| `ClubGameSpread` | `constraints/unified.py` | included | yes | yes | `ClubGameSpread` | `club_spread_slot_used` | spec-024: PER-FIELD `(club,week,day,field)` games-derived hole cap `max(0,min(1,n_field-3))` + slack via shared `_contiguity` indicators (BoolVars). HARD in `club_day` stage + soft: per-field hole penalty AND off-primary-field game count (`total - max_field_count`), all venues |
| `VenueEarliestSlotFill` | `constraints/atoms/venue_earliest_slot_fill.py` | excluded | yes | yes | `venue_slot_used` | `venue_slot_used` | spec-021: anchored monotone-fill per (week,date,location); replaces `EnsureBestTimeslotChoices`. HARD, `critical_feasibility`. Earliest-fill avoids 7pm structurally (no 7pm penalty) |
| `ClubNoConcurrentSlot` | `constraints/atoms/club_no_concurrent_slot.py` | excluded | yes | yes | — | — | spec-021: per (club,week,day_slot,location) cap `max(1, ceil(team_count/no_field_slots[loc]))`; capacity-aware double-up allowance. HARD, `critical_feasibility` |
| `PreferredTimes` | `constraints/archived/original.py` | included | yes | yes | — | — | Dual-orderings hack (t1/t2 swap); see §5.8 |
| `EqualMatchUpSpacing` | `constraints/archived/original.py` | excluded (via adjuster) | yes | yes | `EqualMatchUpSpacingConstraint` | — | Adjuster narrows forced rounds per pair |
| **Atomised (spec-003) — NIHC field-fill order** | | | | | | | |
| `NIHCFillWFBeforeEF` | `constraints/atoms/nihc_fill_wf_before_ef.py` | included | yes | yes | — | `nihc_field_used` | **SOFT (spec-016, severity 5).** Per (date, day_slot) at NIHC where BOTH WF and EF have ≥1 decision var; soft penalty = `ef_used AND NOT wf_used` into the shared `nihc_fill_order` bucket. No hard implication (was `ef_used <= wf_used`). Buckets skipped when either field has no real variables that slot. |
| `NIHCFillEFBeforeSF` | `constraints/atoms/nihc_fill_ef_before_sf.py` | included | yes | yes | — | `nihc_field_used` | **SOFT (spec-016, severity 5).** Same shape; penalty = `sf_used AND NOT ef_used`. Shares the `nihc_field_used` helper cache and the `nihc_fill_order` penalty bucket with the WF/EF atom. Together they make WF→EF→SF the cheapest fill order. |
| **Soft-only penalty atoms (no tester violation check)** | | | | | | | |
| `SoftLexMatchupOrdering` | `constraints/atoms/soft_lex_matchup_ordering.py` | included | yes | yes | — | — | Pure soft tie-break (weight=1 default); no tester violation check; ranks pairs 0-indexed alphabetically per grade |
| `PreferredWeekendsAwayGround` | `constraints/atoms/preferred_weekends_away_ground.py` | n/a | yes | yes | — | — | Pure soft; reads `data['preferred_weekends']`; no tester violation check. Penalty key `preferred_weekends_away_ground` (default 1000). avoid mode: penalises each game at venue on date. prefer mode: penalises shortage vs target_count. |
| `PreferredGames` | `constraints/atoms/preferred_games.py` | reuses FORCED scope/team parser (`_build_scope_count_rules`, `unique_per_entry=True`) | yes (per-variable, `key[6] in locked_weeks`) | yes | — | — | **Soft (spec-020), severity 5, no atom_group.** Generic analogue of the whole FORCED_GAMES grammar. Reads `data['preferred_games']`; per scope adds a penalty-on-deviation from `count` per `constraint` type (equal→`\|sum−N\|`; lesse→`max(0,sum−N)`; greatere→`max(0,N−sum)`; greater→`max(0,(N+1)−sum)`; less→`max(0,sum−(N−1))`). Single shared bucket `preferred_games` (default weight 10000); optional per-entry `weight` is a multiplier. Empty/zero-candidate scope → 0 penalty + logged warning, NEVER `sys.exit`. Tester check `_check_preferred_games` reports deviations as soft pressure (metric_value), not hard violations. |
| **Tester-only (no solver enforcement)** | | | | | | | |
| `ForcedGames` | `constraints/registry.py` | n/a | n/a | n/a | — | — | Diagnostic; enforced via variable-elimination |
| `BlockedGames` | `constraints/registry.py` | n/a | n/a | n/a | — | — | Diagnostic; enforced via variable-elimination |
| `ClubFieldConcentration` | tester only | n/a | n/a | n/a | — | — | Reports clubs concentrated on few fields |

**Forced-games handling glossary:**
- `excluded` — atom skips variables that match a `FORCED_GAMES` scope (either via shared iterator or local `_get_matching_forced_scopes` check).
- `excluded (via adjuster)` — atom reads pre-computed `data['count_adjustments'][canonical_name]` to reduce expected counts; raw vars are not skipped but the constraint is loosened to match what FORCED has already pinned.
- `included` — atom treats FORCED-matched vars like any other; no special handling.
- `n/a` — atom is structural (count-adjuster framework, tester-only) or doesn't iterate vars.

**Uniformity notes** (apply to ALL atomised entries):
- Locked-week skip is enforced uniformly via shared iterators (`iter_phl_keys`, `iter_grade_keys`, `collect_*` in `constraints/atoms/_*_shared.py`). The atom-level `yes` reflects that the iterator filters by `data['locked_weeks']`.
- Dummy-key skip (`len(key) < 11 or not key[3]`) is also at the iterator level.
- Helper-var declarations are namespaced by `kind` (e.g. `cvc_coincide`); no collisions detected.

## 5. Findings vs the plan's assumptions

While walking the source for this inventory, I noticed:

1. **`FiftyFiftyHomeandAway` aggregate block — RESOLVED (spec-004).** The legacy class is now obsolete. The aggregate block is intentionally preserved in `AwayClubPerOpponentAndAggregateHomeBalance` because the convenor wanted both per-pair AND per-team aggregate balance (the intersection gives the best outcome — see spec-004 DoD #3). The earlier "removed deliberately by design" claim in `CLAUDE.md` was inaccurate; the spec-004 plan locks in retaining the aggregate constraint.
2. **Gosford Friday rounds {2, 4, 5, 9, 10} — RESOLVED (spec-015).** The bespoke `GosfordFridayRoundsForced` atom (and `CONSTRAINT_DEFAULTS['gosford_friday_rounds']`) were DELETED. The per-round `sum == 1` rule is now a generic `FORCED_GAMES` count entry (scope + count + constraint type) in the season config — see `FORCED_GAMES_AS_COUNT_RULES.md`. Pinned by `tests/test_forced_games_count_rules.py`.
3. **PHL preferred-dates handling uses `phl_preferences['preferred_dates']` only** — `PHLAndSecondGradeTimes` raises if any other key is present. Confirms Phase 3 atom `PreferredDates` keeps the same single-key contract; expansion to other prefs is a separate behavior change.
4. **`PHLAndSecondGradeTimes` HACK lines (~242-301)**: locked-week PHL Friday counts are accumulated from `data['locked_keys_set']` with substring matching on location (`'Central Coast'`, `'Maitland'`, `'Newcastle'`). Phase 4 should replace substring matching with `home_field_map` lookups so the adjuster works for any non-default-home venue.
5. **`ClubDayConstraint` opponent semantics**: matches `original.py` only — `ai.py`'s simpler date-only form was a regression (Decision #4 already documents this).
6. **`PHLAndSecondGradeAdjacency` time math** — RESOLVED (spec-014). Rewritten as the `PHLAnd2ndAdjacency` atom: the cross-venue gap reads `CONSTRAINT_DEFAULTS['phl_2nd_cross_venue_min_minutes']` (150, real minutes; spec-030: 180→150); the same-venue rule is slot-adjacency (no minute threshold). The legacy ±180 forbid window is gone.
7. **`MaitlandHomeGrouping`** and **`MaxMaitlandHomeWeekends`** — RESOLVED by removal (spec-018). These were folded into the generic `NonDefaultHomeGrouping` in Phase 6, then **deleted entirely in spec-018** — the convenor no longer wants any consecutive-home-weekend / weekend-cap sequencing. No constraint enforces it now.
8. **`PreferredTimesConstraint`** uses two `allowed_keys` orderings to match games (`team_name` swapped between t1/t2 positions). This works because `team1 <= team2` alphabetically — the two orderings handle both. Phase 3 keeps this intact; the atom version may want a cleaner club-team membership filter to avoid the dual-orderings hack, but that's a refactor, not a behavior change.
9. **`EnsureBestTimeslotChoices.WORST_TIME = '19:00'`** is a class-level constant. Phase 5 punch list does not list it explicitly; recommend adding `CONSTRAINT_DEFAULTS['worst_timeslot_time'] = '19:00'`.

These items are flagged for the user; nothing in this phase changes behavior.
