# Constraint Inventory (Phase 0)

Single-source-of-truth table of every registered constraint, what it actually does (extracted from code, not docstring), its severity / slack key, and the atom-target name(s) it splits into during atomization (Phase 3).

Generated against `final-form` @ commit `d99e8c0` from `constraints/registry.py` (21 entries) + `constraints/original.py` (19 solver classes) + `constraints/ai.py` (19 solver classes).

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
| PHLAndSecondGradeTimes | original.py:PHLAndSecondGradeTimes (lines 214-372 — multi-idea, contains user-flagged HACK) | (a) PHL no-concurrent at Broadmeadow per (week, day_slot, location); (b) PHL+same-club 2nd no-concurrent per (week, day, location, day_slot, club, 2nd-team); (c) ≤max_friday_broadmeadow PHL Fridays at NIHC (HACK: minus locked); (d) sum Friday Gosford PHL == gosford_friday_games (HACK: minus locked); (e) sum Friday Maitland PHL == maitland_friday_games (HACK: minus locked); (f) Gosford Friday rounds {2,4,5,9,10} sum == 1 each; (g) PHL each team plays in round 1 (skipped if locked); (h) PHL preferred-dates penalty | 1 | — | PHLConcurrencyAtBroadmeadow + PHLAnd2ndConcurrencyAtBroadmeadow + BroadmeadowFridayCount + GosfordFridayCount + MaitlandFridayCount + GosfordFridayRoundsForced + PHLRoundOnePlay + PreferredDates |
| EqualMatchUpSpacing | original.py:EqualMatchUpSpacingConstraint | HARD: for pair p with rounds (r1, r2), gap < min_gap ⇒ sum(p_r1) + sum(p_r2) ≤ 1; min_gap = max(min(T//2, T-2), T-2 - base_slack - config_slack). SOFT: sliding window of size `space=T-2`, penalize sum-1 when >1. HACK: when locked_weeks set, only applies to PHL/2nd | 1 | EqualMatchUpSpacingConstraint | EqualMatchUpSpacing (Phase 4 adds adjuster for FORCED meetings → reduces effective rounds) |
| ClubDay | original.py:ClubDayConstraint (lines 632-750 — multi-idea) | For each entry in CLUB_DAYS (parsed via `normalize_club_day` → date + optional opponent): (a) every team in club plays on that date; (b) if opponent set, force ≥min(host_grade_count, opp_grade_count) cross-club games per grade; (c) if no opponent / opponent missing grade, force intra-club derbies = #host_grade // 2; (d) all club games on same field; (e) timeslots are contiguous via "if middle slot empty, prior + following ≤ 1" | 2 | — | ClubDayParticipation + ClubDayIntraClubMatchup + ClubDayOpponentMatchup + ClubDaySameField + ClubDayContiguousSlots |
| AwayAtMaitlandGrouping | original.py:AwayAtMaitlandGrouping | For each week: indicator per away club at Maitland Park; sum(indicators) ≤ HARD_LIMIT (= away_maitland_max_clubs + slack); soft penalty = max(0, num_clubs-1) | 2 | AwayAtMaitlandGrouping | AwayAtNonDefaultGrouping (Phase 6 generic — per non-default-home club) |
| TeamConflict | original.py:TeamConflictConstraint | For each (team1, team2) in `team_conflicts`: per (week, day_slot), sum vars involving either ≤ 1; skips locked weeks | 2 | — | TeamConflict |
| ClubGradeAdjacency | original.py:ClubGradeAdjacencyConstraint | (1) HARD: duplicate (same club, same grade ≥2 teams) games at same (week, day_slot) ≤ 1; (2) SOFT: for each adjacent grade pair in GRADE_ORDER per club per (week, day_slot), penalty = max(0, sum(g1)+sum(g2)-1) | 3 | — | ClubGradeAdjacency (rename optional — current behavior preserved) |
| ClubVsClubAlignment | original.py:ClubVsClubAlignment (lines 972-1198 — multi-idea) | Lower-grade block (3rd-6th, sorted by per_team_games asc): for each grade-pair X,Y where Y > X by num_games — for each club_pair seen in both — coincide vars per round; HARD ≥ num_games - slack; HARD ≤ 2 fields when coinciding; SOFT field excess penalty; SOFT coincide deficit penalty. PHL/2nd Sunday block: same coincide algorithm but additionally requires "back-to-back same field" (∃ pair with same field name and abs(slot1-slot2)==1) when coinciding | 3 | ClubVsClubAlignment | ClubVsClubCoincidence + ClubVsClubFieldLimit + ClubVsClubDeficitPenalty + PHLAnd2ndBackToBackSameField |
| ClubGameSpread | original.py:ClubGameSpread | Per (club, week, day) with ≥2 distinct day_slots used: gaps = range_size - num_used; HARD gaps ≤ base_limit + slack; SOFT penalty = gaps. Skips locked weeks | 3 | ClubGameSpread | ClubGameSpread |
| MaximiseClubsPerTimeslotBroadmeadow | original.py:MaximiseClubsPerTimeslotBroadmeadow | At NIHC Sat/Sun per (week, day, day_slot): per-club presence indicator; HARD num_clubs ≥ floor(total_teams_playing/2) - slack (floored at 0); SOFT penalty = max(0, total_teams_playing - num_clubs) when slot used | 4 | MaximiseClubsPerTimeslotBroadmeadow | MaximiseClubsPerTimeslotBroadmeadow |
| MinimiseClubsOnAFieldBroadmeadow | original.py:MinimiseClubsOnAFieldBroadmeadow | At NIHC Sat/Sun per (week, date, field_name): per-club presence indicator; HARD num_clubs ≤ max_clubs_per_field + slack; SOFT penalty = abs(num_clubs - 2) | 4 | MinimiseClubsOnAFieldBroadmeadow | MinimiseClubsOnAFieldBroadmeadow |
| EnsureBestTimeslotChoices | original.py:EnsureBestTimeslotChoices | Per (week, day, location): build per-(field, day_slot) max-equality indicators; for adjacent slot pair (curr, next) for every (f, f2): next_used ⇒ curr_used (cross-field stacking + contiguity). SOFT: 7pm games incur penalty 1 each | 5 | — | EnsureBestTimeslotChoices |
| PreferredTimes | original.py:PreferredTimesConstraint | Normalize PREFERENCE_NO_PLAY (legacy + 2026 structured); for each (entry, club, club_teams, restriction): match X keys via two key orderings; soft penalty = the var when match | 5 | — | PreferredTimes |

## 2. Tester-only diagnostics

| Canonical name | Source | Actual behavior | Severity |
|---|---|---|---|
| ClubFieldConcentration | tester only | Reports clubs concentrated on a small number of fields (no solver enforcement) | 3 |
| ForcedGames | enforced via `generate_X` (variable-elimination) | Verifies each FORCED entry's scope sum matches `count` and `constraint` (default sum==1) | 1 |
| BlockedGames | enforced via `generate_X` (variable-elimination) | Verifies no game key matches a BLOCKED scope+team-matcher | 1 |

## 3. Atomization summary (count)

| Cluster | Legacy classes | Atoms after split | Net change |
|---|---|---|---|
| PHLAndSecondGradeTimes | 1 | 8 | +7 |
| ClubDayConstraint | 1 | 5 | +4 |
| ClubVsClubAlignment | 1 | 4 | +3 |
| MaitlandHomeGrouping + MaxMaitlandHomeWeekends | 2 | 1 (NonDefaultHomeGrouping, with per-club AWAY_VENUE_RULES) | -1 |
| All other entries | 13 | 13 (1:1, with renames in Phase 6) | 0 |
| **Total** | **18 solver + 3 tester-only** | **27 solver + 3 tester-only** | **+13** |

## 4. Findings vs the plan's assumptions

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
