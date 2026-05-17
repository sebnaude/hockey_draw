# Constraint Inventory (Phase 0 / kept current through Phase 7)

Single-source-of-truth table of every registered constraint, what it actually does (extracted from code, not docstring), its severity / slack key, and the atom-target name(s) it splits into during atomization.

Generated against `final-form` and updated through spec-007. The registry currently has **39 entries**: 21 originals + 5 PHL atoms (3a) + 5 ClubDay atoms (3b) + 4 ClubVsClub atoms (3c) + 2 Phase-6 generic aliases (`NonDefaultHomeGrouping`, `AwayAtNonDefaultGrouping`) + 2 spec-007 atoms (`SameGradeSameClubNoConcurrency`, `TeamPairNoConcurrency`).

Legend
- **Source** is the legacy class location. Parity is asserted between `original.py` and `ai.py` versions (5 + 1 historical bug-fixes documented in `CLAUDE.md`).
- **Severity** matches `ConstraintInfo.severity_level` (1=CRITICAL ... 5=VERY LOW).
- **Slack key** matches `ConstraintInfo.slack_key` (where applicable).
- **Atom target** lists the post-atomization atom name(s). Single-idea constraints stay as one atom (sometimes renamed for the generic-home-ground refactor in Phase 6). Multi-idea constraints split.

## 1. Solver-applied constraints

| Canonical name | Source | Actual behavior | Severity | Slack key | Atom target(s) |
|---|---|---|---|---|---|
| NoDoubleBookingTeams | original.py:NoDoubleBookingTeamsConstraint | Sum of team's vars in each (week, team) ≤ 1; skips locked weeks; skips dummy timeslots | 1 | — | NoDoubleBookingTeams |
| NoDoubleBookingFields | original.py:NoDoubleBookingFieldsConstraint | Sum of vars per (day, day_slot, week, field_name) ≤ 1; skips locked weeks | 1 | — | NoDoubleBookingFields |
| EqualGamesAndBalanceMatchUps | original.py:EnsureEqualGamesAndBalanceMatchUps | Per team sum == num_rounds[grade]; per pair sum in [base, base+1] where base = R/(T-1) for even T or R/T for odd; includes dummy slots | 1 | — | EqualGames + BalancedMatchups (two atoms in one cluster) |
| FiftyFiftyHomeandAway | original.py:FiftyFiftyHomeandAway | Per (Maitland/Gosford team, opponent) pair: home games × 2 within total ± 1; PLUS aggregate per-team home × 2 within all_pair_games ± 1 (lines 426–447) — the aggregate block exists in original.py despite CLAUDE.md noting it was "removed by design" | 1 | — | FiftyFiftyHomeandAway (per-pair only — aggregate block to be removed in Phase 3 to match documented intent) |
| MaitlandHomeGrouping | original.py:MaitlandHomeGrouping | (1) Per Maitland week, soft penalty = min(home_games, away_games) (imbalance). (2) Sliding window of (max_consecutive+1) Maitland weeks: at most max_consecutive home weeks; max_consecutive = base_max + slack | 1 | MaitlandHomeGrouping | NonDefaultHomeGrouping (Phase 6 generic) |
| MaitlandHomeGrouping (alias) | original.py:MaxMaitlandHomeWeekends | Distinct: weeks[(week, location)] for non-NIHC locations; per week indicator = max(week vars); per location sum(indicators) ≤ max_games_at_home_field // 2 + 1 | 1 | MaitlandHomeGrouping | folded into NonDefaultHomeGrouping (redundant with MaitlandHomeGrouping per plan) |
| PHLAndSecondGradeAdjacency | original.py:PHLAndSecondGradeAdjacency | Per (club, 2nd-team, week, day): for each PHL game (time, location), sum of PHL var + 2nd-grade vars within ±180 min that satisfy "same loc when within window OR different loc when outside window" ≤ 1 | 1 | — | PHLAndSecondGradeAdjacency |
| PHLAndSecondGradeTimes | original.py:PHLAndSecondGradeTimes (lines 214-372 — multi-idea, contains user-flagged HACK) | (a) PHL no-concurrent at Broadmeadow per (week, day_slot, location); (b) PHL+same-club 2nd no-concurrent per (week, day, location, day_slot, club, 2nd-team); (c) ≤max_friday_broadmeadow PHL Fridays at NIHC (HACK: minus locked); (d) sum Friday Gosford PHL == gosford_friday_games (HACK: minus locked); (e) sum Friday Maitland PHL == maitland_friday_games (HACK: minus locked); (f) Gosford Friday rounds {2,4,5,9,10} sum == 1 each; (g) PHL each team plays in round 1 (skipped if locked); (h) PHL preferred-dates penalty | 1 | — | (Phase 3a partially shipped @ `1956608`, post-retraction reduces to 4 atoms.) **Atoms:** PHLConcurrencyAtBroadmeadow + PHLAnd2ndConcurrencyAtBroadmeadow + PHLRoundOnePlay + PreferredDates. **Expressed as FORCED_GAMES entries instead** (see `docs/FORCED_GAMES_AS_COUNT_RULES.md`): items (c)/(d)/(e) (per-venue Friday counts) and (f) (Gosford rounds — already covered by per-round FORCED entries in season_2026.py). The HACK locked-week count adjustments disappear because each FORCED entry is per-season and the count is the season target, not a runtime computation. |
| EqualMatchUpSpacing | original.py:EqualMatchUpSpacingConstraint | HARD: for pair p with rounds (r1, r2), gap < min_gap ⇒ sum(p_r1) + sum(p_r2) ≤ 1; min_gap = max(min(T//2, T-2), T-2 - base_slack - config_slack). SOFT: sliding window of size `space=T-2`, penalize sum-1 when >1. HACK: when locked_weeks set, only applies to PHL/2nd | 1 | EqualMatchUpSpacingConstraint | EqualMatchUpSpacing (Phase 4 adds adjuster for FORCED meetings → reduces effective rounds) |
| ClubDay | original.py:ClubDayConstraint (lines 632-750 — multi-idea) | For each entry in CLUB_DAYS (parsed via `normalize_club_day` → date + optional opponent): (a) every team in club plays on that date; (b) if opponent set, force ≥min(host_grade_count, opp_grade_count) cross-club games per grade; (c) if no opponent / opponent missing grade, force intra-club derbies = #host_grade // 2; (d) all club games on same field; (e) timeslots are contiguous via "if middle slot empty, prior + following ≤ 1" | 2 | — | **Phase 3b ✅** atoms shipped: ClubDayParticipation + ClubDayIntraClubMatchup + ClubDayOpponentMatchup + ClubDaySameField + ClubDayContiguousSlots |
| AwayAtMaitlandGrouping | original.py:AwayAtMaitlandGrouping | For each week: indicator per away club at Maitland Park; sum(indicators) ≤ HARD_LIMIT (= away_maitland_max_clubs + slack); soft penalty = max(0, num_clubs-1) | 2 | AwayAtMaitlandGrouping | AwayAtNonDefaultGrouping (Phase 6 generic — per non-default-home club) |
| TeamConflict | original.py:TeamConflictConstraint | For each (team1, team2) in `team_conflicts`: per (week, day_slot), sum vars involving either ≤ 1; skips locked weeks | 2 | — | TeamConflict |
| ClubGradeAdjacency | original.py:ClubGradeAdjacencyConstraint | **OBSOLETE (spec-007).** Hard portion (duplicate same-club-same-grade) split out as `SameGradeSameClubNoConcurrency` (severity 1). Soft portion (adjacent-grade penalty) **REMOVED ENTIRELY** — convenor experience was that it was over-restrictive. Registry entry kept so solver class names still resolve and the tester can keep emitting violations under the same name for the same-grade-same-club case. | 3 | — | `SameGradeSameClubNoConcurrency` (hard) + `TeamPairNoConcurrency` (new soft, per-pair convenor list) |
| SameGradeSameClubNoConcurrency | atoms/same_grade_same_club_no_concurrency.py | HARD: per (club, grade, week, day_slot) bucket, sum of cross-club games involving any duplicate-set team from that club ≤ 1. Intra-club derbies (single shared variable) are excluded. | 1 | — | SameGradeSameClubNoConcurrency (atom_group: ClubGradeAdjacency) |
| TeamPairNoConcurrency | atoms/team_pair_no_concurrency.py | SOFT: reads `constraint_defaults['TEAM_PAIR_NO_CONCURRENCY']` of (team_a, team_b) or (team_a, team_b, weight_multiplier) entries. Per (week, day_slot), pen = max(0, sum(vars_team_a) + sum(vars_team_b) - 1), scaled by multiplier. Base penalty weight from `PENALTY_WEIGHTS['TeamPairNoConcurrency']` (default 1000). | 3 | — | TeamPairNoConcurrency |
| ClubVsClubAlignment | original.py:ClubVsClubAlignment (lines 972-1198 — multi-idea) | Lower-grade block (3rd-6th, sorted by per_team_games asc): for each grade-pair X,Y where Y > X by num_games — for each club_pair seen in both — coincide vars per round; HARD ≥ num_games - slack; HARD ≤ 2 fields when coinciding; SOFT field excess penalty; SOFT coincide deficit penalty. PHL/2nd Sunday block: same coincide algorithm but additionally requires "back-to-back same field" (∃ pair with same field name and abs(slot1-slot2)==1) when coinciding | 3 | ClubVsClubAlignment | **Phase 3c ✅** atoms shipped: ClubVsClubCoincidence + ClubVsClubFieldLimit + ClubVsClubDeficitPenalty + PHLAnd2ndBackToBackSameField. Atoms re-introduce the PHL/2nd back-to-back rule the pre-atomization unified engine silently dropped. |
| ClubGameSpread | original.py:ClubGameSpread | Per (club, week, day) with ≥2 distinct day_slots used: gaps = range_size - num_used; HARD gaps ≤ base_limit + slack; SOFT penalty = gaps. Skips locked weeks | 3 | ClubGameSpread | ClubGameSpread |
| MaximiseClubsPerTimeslotBroadmeadow | original.py:MaximiseClubsPerTimeslotBroadmeadow | At NIHC Sat/Sun per (week, day, day_slot): per-club presence indicator; HARD num_clubs ≥ floor(total_teams_playing/2) - slack (floored at 0); SOFT penalty = max(0, total_teams_playing - num_clubs) when slot used | 4 | MaximiseClubsPerTimeslotBroadmeadow | MaximiseClubsPerTimeslotBroadmeadow |
| MinimiseClubsOnAFieldBroadmeadow | original.py:MinimiseClubsOnAFieldBroadmeadow | At NIHC Sat/Sun per (week, date, field_name): per-club presence indicator; HARD num_clubs ≤ max_clubs_per_field + slack; SOFT penalty = abs(num_clubs - 2) | 4 | MinimiseClubsOnAFieldBroadmeadow | MinimiseClubsOnAFieldBroadmeadow |
| EnsureBestTimeslotChoices | original.py:EnsureBestTimeslotChoices | Per (week, day, location): build per-(field, day_slot) max-equality indicators; for adjacent slot pair (curr, next) for every (f, f2): next_used ⇒ curr_used (cross-field stacking + contiguity). SOFT: 7pm games incur penalty 1 each | 5 | — | EnsureBestTimeslotChoices |
| PreferredTimes | original.py:PreferredTimesConstraint | Normalize PREFERENCE_NO_PLAY (legacy + 2026 structured); for each (entry, club, club_teams, restriction): match X keys via two key orderings; soft penalty = the var when match | 5 | — | PreferredTimes |

## 1b. Phase-6 canonical names + back-compat aliases

| Canonical name | Back-compat alias(es) | Purpose |
|---|---|---|
| `NonDefaultHomeGrouping` | `MaitlandHomeGrouping` (also resolves `MaxMaitlandHomeWeekends*` solver classes) | Generic per-club non-default-home back-to-back constraint. Iterates `home_field_map.keys()`; per-club tuning from `AWAY_VENUE_RULES[club]['max_consecutive_home']` (None disables). Falls back to `CONSTRAINT_DEFAULTS['maitland_max_consecutive_home']`. |
| `AwayAtNonDefaultGrouping` | `AwayAtMaitlandGrouping` | Generic per-venue away-clubs-per-week constraint. Iterates `home_field_map.values()`; per-club `AWAY_VENUE_RULES[club]['max_away_clubs']` (None disables). Falls back to `CONSTRAINT_DEFAULTS['away_maitland_max_clubs']`. |

The `NonDefaultHomeGrouping` / `AwayAtNonDefaultGrouping` entries are the
**canonical** ones — they own the `solver_class_names` (incl. the legacy
`MaitlandHomeGrouping*` / `AwayAtMaitlandGrouping*` class names in
`constraints/archived/`), so `get_canonical_for_solver_name(...)` resolves
any legacy class name to the generic canonical.

The `MaitlandHomeGrouping` / `AwayAtMaitlandGrouping` entries remain as
back-compat aliases (empty `solver_class_names`, otherwise mirror the
canonical entry's tester / severity / slack metadata) so older configs,
data dicts, slack-key lookups, severity-table walks, and tests that look
them up by name continue to work. The runtime slack key string in
`data['constraint_slack']['MaitlandHomeGrouping']` (read literally inside
`unified.py` / `tester.py`) is kept as-is; renaming it is internal and
out of scope for Phase 6.

## 2. Tester-only diagnostics

| Canonical name | Source | Actual behavior | Severity |
|---|---|---|---|
| ClubFieldConcentration | tester only | Reports clubs concentrated on a small number of fields (no solver enforcement) | 3 |
| ForcedGames | enforced via `generate_X` (variable-elimination) | Verifies each FORCED entry's scope sum matches `count` and `constraint` (default sum==1) | 1 |
| BlockedGames | enforced via `generate_X` (variable-elimination) | Verifies no game key matches a BLOCKED scope+team-matcher | 1 |

## 3. Atomization summary (count)

| Cluster | Legacy classes | Atoms after split | Net change |
|---|---|---|---|
| PHLAndSecondGradeTimes | 1 | 5 — `PHLConcurrencyAtBroadmeadow`, `PHLAnd2ndConcurrencyAtBroadmeadow`, `GosfordFridayRoundsForced`, `PHLRoundOnePlay`, `PreferredDates` (per-venue Friday counts moved to FORCED_GAMES entries — see `docs/FORCED_GAMES_AS_COUNT_RULES.md`) | +4 |
| ClubDayConstraint | 1 | 5 | +4 |
| ClubVsClubAlignment | 1 | 4 | +3 |
| MaitlandHomeGrouping + MaxMaitlandHomeWeekends | 2 | 1 (NonDefaultHomeGrouping, with per-club AWAY_VENUE_RULES) | -1 |
| ClubGradeAdjacency (spec-007) | 1 | 1 hard (`SameGradeSameClubNoConcurrency`) + 1 new soft (`TeamPairNoConcurrency`). Adjacent-grade soft rule removed entirely (net +1 atom vs cluster, +1 from the new convenor-list atom). | +1 |
| All other entries | 13 | 13 (1:1, with renames in Phase 6) | 0 |
| **Total** | **18 solver + 3 tester-only** | **26 solver + 3 tester-only** | **+11** |

## 4. Per-atom engineering detail

Engineering-level table for every registered atom + non-atomised legacy constraint. The columns answer the questions an engineer asks before touching code: does it skip forced vars? does it skip locked weeks? what slack key does it read? what helper vars does it consume?

**Rule:** when an atom is added or changed, this table gets a row in the same commit. A missing or `?` row is treated as a bug (see `docs/system/README.md`).

| Canonical name | File | Forced-games | Locked-week skip | Dummy-key skip | Slack key | Helper vars | Caveats |
|---|---|---|---|---|---|---|---|
| **Atomised (Phase 3) — PHL/2nd cluster** | | | | | | | |
| `PHLConcurrencyAtBroadmeadow` | `constraints/atoms/phl_concurrency.py` | n/a | yes | yes | — | — | Broadmeadow-only; prevents concurrent PHL games per (week, day_slot) |
| `PHLAnd2ndConcurrencyAtBroadmeadow` | `constraints/atoms/phl_2nd_concurrency.py` | n/a | yes | yes | — | — | PHL + same-club-2nd no-concurrency at Broadmeadow |
| `GosfordFridayRoundsForced` | `constraints/atoms/gosford_friday_rounds.py` | n/a | yes | yes | — | — | Reads `CONSTRAINT_DEFAULTS['gosford_friday_rounds']` (currently `{2,4,5,9,10}` per 2026 AGM) |
| `PHLRoundOnePlay` | `constraints/atoms/phl_round_one_play.py` | n/a | yes | yes | — | — | Skipped when locked_weeks includes round 1 |
| `PreferredDates` | `constraints/atoms/preferred_dates.py` | excluded | yes | yes | — | — | Soft; weight `PENALTY_WEIGHTS['phl_preferences']=10000` |
| **Atomised (Phase 3) — ClubDay cluster** | | | | | | | |
| `ClubDayParticipation` | `constraints/atoms/club_day_participation.py` | excluded | yes | yes | — | — | Skips locked-week club days via `parse_club_day_entries` |
| `ClubDayIntraClubMatchup` | `constraints/atoms/club_day_intra_club_matchup.py` | excluded | yes | yes | — | — | Derby logic when opponent undefined / empty for grade |
| `ClubDayOpponentMatchup` | `constraints/atoms/club_day_opponent_matchup.py` | excluded | yes | yes | — | — | Cross-club matchups per grade when opponent set |
| `ClubDaySameField` | `constraints/atoms/club_day_same_field.py` | excluded | yes | yes | `club_day_field_used` | `club_day_field_used` | Channels field indicators via registry |
| `ClubDayContiguousSlots` | `constraints/atoms/club_day_contiguous_slots.py` | excluded | yes | yes | `club_day_slot_used` | `club_day_slot_used` | Sliding window; requires ≥3 slots to apply |
| **Atomised (Phase 3) — ClubVsClub cluster** *(to be replaced by spec-005)* | | | | | | | |
| `ClubVsClubCoincidence` | `constraints/atoms/club_vs_club_coincidence.py` | excluded (via adjuster) | yes | yes | `ClubVsClubAlignment` | `cvc_coincide` | Adjuster reduces expected counts for FORCED off-Sunday / BLOCKED on-Sunday |
| `ClubVsClubFieldLimit` | `constraints/atoms/club_vs_club_field_limit.py` | n/a (reads coincide vars) | yes | yes | `ClubVsClubAlignment` | `cvc_coincide` (reads) | Soft penalty `ClubVsClubAlignmentField`=50000; hard ≤2 fields |
| `ClubVsClubDeficitPenalty` | `constraints/atoms/club_vs_club_deficit_penalty.py` | n/a (reads coincide vars) | yes | yes | — | `cvc_coincide` (reads) | Soft penalty into `ClubVsClubAlignment` bucket=100000 |
| `PHLAnd2ndBackToBackSameField` | `constraints/atoms/phl_2nd_back_to_back.py` | n/a | yes | yes | `ClubVsClubAlignment` | `cvc_phl_btb_coincide` | PHL/2nd Sunday only; back-to-back same-field pairing logic |
| **Non-atomised (still routed through `constraints/archived/`)** | | | | | | | |
| `NoDoubleBookingTeams` | `constraints/archived/original.py` | included | yes | yes | — | — | Fundamental; no forced exclusion needed |
| `NoDoubleBookingFields` | `constraints/archived/original.py` | included | yes | yes | — | — | Uses `date`, not `week`, for slot comparisons |
| `EqualGamesAndBalanceMatchUps` | `constraints/archived/original.py` | included | yes | yes | — | — | Includes dummy vars (overflow); two distinct tester violation types |
| `FiftyFiftyHomeandAway` | `constraints/archived/original.py` | included | yes | yes | — | — | **To be replaced by spec-004 atoms.** Aggregate-block parity issue flagged in §5.1 below |
| `PHLAndSecondGradeAdjacency` | `constraints/archived/original.py` | included | yes | yes | — | — | 180-min window — hardcoded; Phase 5 punch list `phl_adjacency_window_minutes` |
| `NonDefaultHomeGrouping` | `constraints/archived/original.py` | excluded (via adjuster) | yes | yes | `MaitlandHomeGrouping` | — | Phase-6 generic rename; iterates `home_field_map.keys()`; per-club `AWAY_VENUE_RULES[club]` |
| `AwayAtNonDefaultGrouping` | `constraints/archived/original.py` | excluded (via adjuster) | yes | yes | `AwayAtMaitlandGrouping` | — | Phase-6 generic rename; iterates `home_field_map.values()` |
| `TeamConflict` | `constraints/archived/original.py` | included | yes | yes | — | — | Per (team1, team2) in `team_conflicts` |
| `ClubGradeAdjacency` | `constraints/archived/original.py` | n/a | n/a | n/a | — | — | **OBSOLETE (spec-007).** Not dispatched by the engine or by any stage. Registry entry kept for back-compat lookups; the tester still emits violations under this name for the same-grade-same-club case only. |
| `SameGradeSameClubNoConcurrency` | `constraints/atoms/same_grade_same_club_no_concurrency.py` | included | yes | yes | — | — | **spec-007.** Hard severity 1. Dispatched via non-engine fallback in `constraints/stages.py` (`_resolve_solver_class` now searches `constraints.atoms`). Intra-club derbies are excluded (single shared variable). |
| `TeamPairNoConcurrency` | `constraints/atoms/team_pair_no_concurrency.py` | included | yes | yes | — | — | **spec-007.** Soft severity 3. Reads `constraint_defaults['TEAM_PAIR_NO_CONCURRENCY']`. Penalty bucket `TeamPairNoConcurrency`; default base weight 1000; per-entry weight multiplier scales the penalty IntVar. |
| `ClubGameSpread` | `constraints/archived/original.py` | included | yes | yes | `ClubGameSpread` | — | Soft; per (club, week, day) when ≥2 slots used |
| `MaximiseClubsPerTimeslotBroadmeadow` | `constraints/archived/original.py` | included | yes | yes | `MaximiseClubsPerTimeslotBroadmeadow` | — | Broadmeadow Sat/Sun only |
| `MinimiseClubsOnAFieldBroadmeadow` | `constraints/archived/original.py` | included | yes | yes | `MinimiseClubsOnAFieldBroadmeadow` | — | Broadmeadow Sat/Sun only |
| `EnsureBestTimeslotChoices` | `constraints/archived/original.py` | included | yes | yes | — | — | Hardcoded WORST_TIME = `'19:00'`; Phase 5 punch list to migrate to defaults |
| `PreferredTimes` | `constraints/archived/original.py` | included | yes | yes | — | — | Dual-orderings hack (t1/t2 swap); see §5.8 |
| `EqualMatchUpSpacing` | `constraints/archived/original.py` | excluded (via adjuster) | yes | yes | `EqualMatchUpSpacingConstraint` | — | Adjuster narrows forced rounds per pair |
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

1. **`FiftyFiftyHomeandAway` aggregate block still present** — `original.py` lines 426-447 enforce per-team aggregate balance (`agg_home * 2 ∈ [agg_total - 1, agg_total + 1]`), even though `CLAUDE.md` and `ATOMIZATION_PLAN.md` both state aggregate balance was "removed deliberately — by design". `ai.py` likely has the same. This is a parity point for Phase 3: when atomizing, the aggregate block should be dropped from the new atom to match documented intent. **Flag for user sign-off** before deletion.
2. **Gosford Friday rounds {2, 4, 5, 9, 10}** are hardcoded in `PHLAndSecondGradeTimes` (line 355). Phase 5 punch list flags this for migration to `CONSTRAINT_DEFAULTS['gosford_friday_rounds']`. The atomized `GosfordFridayRoundsForced` atom (added separately from the count atom because it enforces per-round forced placement, not the total) should consume that config key.
3. **PHL preferred-dates handling uses `phl_preferences['preferred_dates']` only** — `PHLAndSecondGradeTimes` raises if any other key is present. Confirms Phase 3 atom `PreferredDates` keeps the same single-key contract; expansion to other prefs is a separate behavior change.
4. **`PHLAndSecondGradeTimes` HACK lines (~242-301)**: locked-week PHL Friday counts are accumulated from `data['locked_keys_set']` with substring matching on location (`'Central Coast'`, `'Maitland'`, `'Newcastle'`). Phase 4 should replace substring matching with `home_field_map` lookups so the adjuster works for any non-default-home venue.
5. **`ClubDayConstraint` opponent semantics**: matches `original.py` only — `ai.py`'s simpler date-only form was a regression (Decision #4 already documents this).
6. **`PHLAndSecondGradeAdjacency` time math** uses `datetime.strptime(t.time, '%H:%M')` with a 180-min window. Phase 5 should migrate the magic 180 to `CONSTRAINT_DEFAULTS['phl_adjacency_window_minutes']`.
7. **`MaitlandHomeGrouping`** and **`MaxMaitlandHomeWeekends`** are separate classes both registered under canonical name `MaitlandHomeGrouping`. `MaxMaitlandHomeWeekends` enforces `sum(home_indicators) ≤ num_games // 2 + 1` per location and is structurally redundant with the sliding-window in `MaitlandHomeGrouping` once that uses sliding-window math. Phase 6 plan correctly folds them; flag here for sign-off.
8. **`PreferredTimesConstraint`** uses two `allowed_keys` orderings to match games (`team_name` swapped between t1/t2 positions). This works because `team1 <= team2` alphabetically — the two orderings handle both. Phase 3 keeps this intact; the atom version may want a cleaner club-team membership filter to avoid the dual-orderings hack, but that's a refactor, not a behavior change.
9. **`EnsureBestTimeslotChoices.WORST_TIME = '19:00'`** is a class-level constant. Phase 5 punch list does not list it explicitly; recommend adding `CONSTRAINT_DEFAULTS['worst_timeslot_time'] = '19:00'`.

These items are flagged for the user; nothing in this phase changes behavior.
