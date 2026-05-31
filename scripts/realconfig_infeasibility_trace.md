# REAL-2026-config infeasibility trace — what to loosen

Generated 2026-05-30T20:16:22 · year=2026 · workers=8 · cap=200s/probe · raw (no slack).

Goal: localize which HARD `core` atoms over-constrain the **real** 2026 config (18 forced / 69 blocked games / LOCKED_PAIRINGS) so they can be given `--slack`. Not a bug hunt — a slack-targeting trace.

| probe | groups | dropped atoms | verdict |
|-------|--------|---------------|---------|
| P0_core_full | core | — | **INFEASIBLE_PRESOLVE** |
| P1_core_no_club_alignment | core | ClubVsClubStackedWeekends, ClubVsClubStackedCoLocation | **REACHED_SEARCH** |
| P2_core_no_club_day | core | ClubDayParticipation, ClubDayIntraClubMatchup, ClubDayOpponentMatchup, ClubDaySameField, ClubDayContiguousSlots | **INFEASIBLE_PRESOLVE** |
| P3_core_no_home_away | core | AwayClubHomeWeekendsCount, AwayClubPerOpponentAndAggregateHomeBalance | **INFEASIBLE_PRESOLVE** |
| P4_core_no_align_no_clubday | core | ClubVsClubStackedWeekends, ClubVsClubStackedCoLocation, ClubDayParticipation, ClubDayIntraClubMatchup, ClubDayOpponentMatchup, ClubDaySameField, ClubDayContiguousSlots | **REACHED_SEARCH** |
| P5_core_bye_spacing | core,bye_spacing | — | **INFEASIBLE_INITIAL_COPY** |

## Reading

- A probe that flips to **REACHED_SEARCH / FEASIBLE** when an atom group is dropped identifies that group as an over-constraint → a loosening (slack) candidate.
- **INFEASIBLE_INITIAL_COPY** = a trivial slack-0 contradiction (loosen that atom's base slack).
- **INFEASIBLE_PRESOLVE** with the group still present = the conflict is elsewhere / an interaction.

## Per-probe solver logs
- **P0_core_full** → INFEASIBLE_PRESOLVE · `logs\solver_20260530_200100_realbisect_P0_core_full.log`
- **P1_core_no_club_alignment** → REACHED_SEARCH · `logs\solver_20260530_200318_realbisect_P1_core_no_club_alignment.log`
- **P2_core_no_club_day** → INFEASIBLE_PRESOLVE · `logs\solver_20260530_200703_realbisect_P2_core_no_club_day.log`
- **P3_core_no_home_away** → INFEASIBLE_PRESOLVE · `logs\solver_20260530_201006_realbisect_P3_core_no_home_away.log`
- **P4_core_no_align_no_clubday** → REACHED_SEARCH · `logs\solver_20260530_201218_realbisect_P4_core_no_align_no_clubday.log`
- **P5_core_bye_spacing** → INFEASIBLE_INITIAL_COPY · `logs\solver_20260530_201559_realbisect_P5_core_bye_spacing.log`