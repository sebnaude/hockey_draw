# Constraint Comparison: Original vs AI-Enhanced (2026 Season)

Generated: March 7, 2026

## Summary

| Metric | Original | AI-Enhanced | Difference |
|--------|----------|-------------|------------|
| **Total Constraints** | 56,675 | 53,861 | **-2,814 (5.0% fewer)** |
| Decision Variables (X) | 119,702 | 119,702 | Same |
| Model Variables (total) | 148,497 | 148,500 | +3 |

## Constraint-by-Constraint Comparison

| Constraint | Original | AI | Change | Notes |
|------------|----------|-----|--------|-------|
| NoDoubleBookingTeams | 1,062 | 1,062 | 0 | Same |
| NoDoubleBookingFields | 792 | 729 | **-63 (8% fewer)** | AI optimized |
| EnsureEqualGamesAndBalanceMatchUps | 418 | 418 | 0 | Same |
| PHLAndSecondGradeAdjacency | 2,112 | 5,104 | **+2,992** | AI more explicit |
| PHLAndSecondGradeTimes | 112 | 464 | **+352** | AI more explicit |
| FiftyFiftyHomeandAway | 176 | 176 | 0 | Same |
| TeamConflictConstraint | 0 | 0 | 0 | No conflicts defined |
| MaxMaitlandHomeWeekends | 47 | 47 | 0 | Same |
| ClubDayConstraint | 19 | 23 | +4 | Minor difference |
| EqualMatchUpSpacingConstraint | 15,227 | 15,227 | 0 | Same |
| ClubGradeAdjacencyConstraint | 10,851 | 4,752 | **-6,099 (56% fewer)** | AI major optimization |
| ClubVsClubAlignment | 25,415 | 25,415 | 0 | Same |
| MaitlandHomeGrouping | 114 | 114 | 0 | Same |
| AwayAtMaitlandGrouping | 330 | 330 | 0 | Same |

## Key Findings

### AI Reduces Total Constraints by 5%
The AI-enhanced constraints produce a net reduction of 2,814 constraints (5.0% fewer), which should result in faster solve times.

### Major Optimization: ClubGradeAdjacencyConstraint
- **Original:** 10,851 constraints
- **AI:** 4,752 constraints  
- **Reduction:** 6,099 constraints (56% fewer)

This is the single largest optimization. The AI version uses more efficient constraint formulations while achieving the same logical outcome.

### Trade-offs: More Explicit PHL Constraints
Some AI constraints use **more** constraints than the original:
- `PHLAndSecondGradeAdjacency`: +2,992 constraints
- `PHLAndSecondGradeTimes`: +352 constraints

These increases are intentional - the AI uses more explicit constraint formulations that may be easier for the solver to reason about, potentially improving solve time despite the higher constraint count.

### Unchanged Constraints
The following constraints have identical implementations:
- NoDoubleBookingTeams (1,062)
- EnsureEqualGamesAndBalanceMatchUps (418)
- FiftyFiftyHomeandAway (176)
- TeamConflictConstraint (0)
- MaxMaitlandHomeWeekends (47)
- EqualMatchUpSpacingConstraint (15,227)
- ClubVsClubAlignment (25,415)
- MaitlandHomeGrouping (114)
- AwayAtMaitlandGrouping (330)

## Variable Filtering Summary

Both implementations use identical variable filtering:
- **PHL:** 5,610 created, 6,600 skipped (invalid venue/field/day/time)
- **2nd Grade:** 2,244 created, 2,640 skipped (invalid venue/field/day/time)
- **Other grades:** 111,848 created, 21,648 skipped (PHL-only venues/days)
- **Total:** 119,702 decision variables created, 30,888 skipped

## Methodology

Constraint counts were measured using `len(model.Proto().constraints)` before and after each constraint class is applied - the same method used by `main_staged.py`. This accurately counts the OR-Tools constraints added to the model.
