# System Capabilities

A precise list of every capability implemented in the Hockey Draw Scheduling System.

---

## 1. DRAW GENERATION

### 1.1 Core Solving
| Capability | Location | Description |
|------------|----------|-------------|
| Generate complete draw | `main_staged.py` | Create full season schedule using CP-SAT solver |
| Staged solving (4 stages) | `main_staged.py` | Solve in stages: Required → Strong → Medium → Soft |
| Simple (non-staged) solving | `main_staged.py --simple` | Single-pass solve with all constraints |
| Resume from checkpoint | `main_staged.py` | Continue solving from saved stage checkpoint |
| Solution callback saves | `SaveStateCallback` | Auto-save best solution during solve |

### 1.2 Timeslot Management
| Capability | Location | Description |
|------------|----------|-------------|
| Dummy/overflow slots | `constraints.py` | N configurable slots without field/time constraints |
| Dummy slot penalty | Objective function | Heavy penalty (in `Maximize`) for using dummy slots |
| Dummy in game counting | `EnsureEqualGamesAndBalanceMatchUps` | Dummies count toward team's total games |
| Dummy in matchup balance | `EnsureEqualGamesAndBalanceMatchUps` | Dummies count toward pair matchup totals |

**Config:** `data['num_dummy_timeslots'] = 3` (default)

### 1.3 Game Distribution
| Capability | Location | Description |
|------------|----------|-------------|
| Equal games per team | `EnsureEqualGamesAndBalanceMatchUps` | Each team plays exactly `num_rounds[grade]` games |
| Base matchup balance | `EnsureEqualGamesAndBalanceMatchUps` | Each pair plays `base` or `base+1` times |
| Extra games mode | `EnsureEqualGamesAndBalanceMatchUps` | Allow `base+1` matchups to maximize total games |

**Formula:** 
- Even teams: `base = num_rounds / (teams - 1)`
- Odd teams: `base = num_rounds / teams`

---

## 2. CONSTRAINTS

### 2.1 Hard Constraints (Must Satisfy)
| Constraint | Description |
|------------|-------------|
| `NoPlayAnywhereConstraint` | Block specific team+timeslot combinations |
| `NoPlaySameSlotConstraint` | Conflicting teams can't play same slot |
| `NoDoubleBookField` | One game per field per slot |
| `EnsureEqualGamesAndBalanceMatchUps` | Total games + pair balance |

### 2.2 Structural Constraints
| Constraint | Description |
|------------|-------------|
| `PHLAndSecondGradeAdjacency` | PHL and 2nd can't play adjacent slots at different venues |
| `ClubGradeAdjacency` | Same club's adjacent grades don't play simultaneously |
| `FiftyFiftyHomeAway` | Maitland/Gosford get 50% home games |
| `SecondGradeTimeConstraint` | 2nd grade plays specific time windows |
| `PHLTimeConstraint` | PHL plays specific time windows |

### 2.3 Preference/Optimization Constraints
| Constraint | Description |
|------------|-------------|
| `MaitlandBackToBack` | Minimize consecutive Maitland home weekends |
| `MaitlandHomeAwayBalance` | Balance home/away per week at Maitland |
| `MaitlandAwayClubsLimit` | Max 3 away clubs at Maitland per week |
| `MaximiseClubsPerTimeslotBroadmeadow` | Diversity of clubs in each timeslot |
| `ClubDays` | Preference for club-specific days |
| `HomeGrouping` | Group home games together |
| `PHLSecondGradeSameVenue` | PHL and 2nd at same venue when possible |

---

## 3. STORAGE & PERSISTENCE

### 3.1 Draw Storage
| Capability | Location | Method/Class |
|------------|----------|--------------|
| Save draw to JSON | `analytics/storage.py` | `draw.save(path)` |
| Load draw from JSON | `analytics/storage.py` | `DrawStorage.load(path)` |
| Convert X solution to storage | `analytics/storage.py` | `DrawStorage.from_X_solution(X)` |
| Convert roster to storage | `analytics/storage.py` | `DrawStorage.from_roster(roster)` |
| Convert storage back to X dict | `analytics/storage.py` | `draw.to_X_dict()` |
| Load from Excel | `analytics/storage.py` | `DrawStorage.from_excel(path, data)` |

### 3.2 Partial Draw / Locking
| Capability | Location | Method |
|------------|----------|--------|
| Lock games up to week N | `analytics/storage.py` | `draw.get_locked_games(lock_weeks_up_to)` |
| Get remaining unlocked games | `analytics/storage.py` | `draw.get_remaining_games(lock_weeks_up_to)` |
| Split draw into locked/unlocked | `analytics/storage.py` | `draw.lock_and_split(week)` |
| Load and prepare locked keys | `analytics/storage.py` | `DrawStorage.load_and_lock(path, week)` |
| Merge two draws | `analytics/storage.py` | `draw.merge_with(other_draw)` |

### 3.3 Checkpoints
| Capability | Location | Description |
|------------|----------|-------------|
| Save stage checkpoint | `CheckpointManager` | Save completed stage to `checkpoints/run_N/` |
| Load stage checkpoint | `CheckpointManager` | Resume from `stage{N}_{name}.pkl` |
| Auto-increment run directories | `CheckpointManager` | Creates `run_1`, `run_2`, etc. |
| Solution callback checkpoint | `SaveStateCallback` | Periodic saves during solve |

---

## 4. EXPORTS

### 4.1 Excel Exports
| Capability | Location | Method |
|------------|----------|--------|
| Weekly schedule workbook | `utils.py` | `export_roster_to_excel(roster, data, path)` |
| Full analytics workbook | `analytics/storage.py` | `analytics.export_analytics_to_excel(path)` |
| Club-specific report | `analytics/reports.py` | `ClubReport.generate_excel(path)` |
| Grade-specific report | `analytics/reports.py` | `GradeReport.generate_excel(path)` |
| Compliance certificate | `analytics/reports.py` | `ComplianceCertificate.generate(path)` |

### 4.2 CSV Exports
| Capability | Location | Method |
|------------|----------|--------|
| Rev format CSV | `analytics/storage.py` | `export_draw_to_revformat(draw, data, path)` |
| Rev format with week limit | `analytics/storage.py` | `export_draw_to_revformat(..., week_limit=N)` |

**Rev format fields:** DATE, TIME, FIELD, VENUE, ROUND, GRADE, TEAM 1, TEAM 2

### 4.3 Text/Console Exports
| Capability | Location | Method |
|------------|----------|--------|
| Club report text file | `analytics/reports.py` | `ClubReport.generate_text(path)` |
| Club report to console | `analytics/reports.py` | `ClubReport.print_summary()` |
| Compliance summary to console | `analytics/reports.py` | `ComplianceCertificate.print_summary()` |
| Violation report to console | `analytics/tester.py` | `report.full_report()` |

---

## 5. ANALYTICS

### 5.1 Summary Statistics
| Capability | Location | Method |
|------------|----------|--------|
| Games per team per grade | `DrawAnalytics` | `games_played_by_team_grade()` |
| Home/away balance per team | `DrawAnalytics` | `home_away_analysis()` |
| Away team balance (Maitland/Gosford) | `DrawAnalytics` | `away_team_balance()` |
| Team vs team matchup matrix | `DrawAnalytics` | `team_matchups_crosstab()` |
| Weekly field usage | `DrawAnalytics` | `weekly_field_usage()` |
| Club season schedule | `DrawAnalytics` | `club_season_schedule(club_name)` |

### 5.2 Slot Analysis
| Capability | Location | Method |
|------------|----------|--------|
| Get all possible slots | `SlotAnalyzer` | `get_all_possible_slots()` |
| Get used slots | `SlotAnalyzer` | `get_used_slots(week=None)` |
| Get unused slots | `SlotAnalyzer` | `get_unused_slots(week=None)` |
| Slot usage summary table | `SlotAnalyzer` | `slot_usage_summary()` |
| Print unused slots | `SlotAnalyzer` | `print_unused_slots(week=None)` |
| Find available slots (tester) | `DrawTester` | `find_available_slots(week)` |
| List all unused slots | `DrawTester` | `list_unused_slots(week=None)` |

---

## 6. TESTING & MODIFICATION

### 6.1 Violation Checking
| Capability | Location | Method |
|------------|----------|--------|
| Run all constraint checks | `DrawTester` | `run_violation_check()` |
| Get violation report | `DrawTester` | Returns `ViolationReport` with severity levels |
| Check specific constraint | `DrawTester` | `_check_{constraint_name}()` |

**Checks implemented:**
- NoDoubleBookingTeams
- NoDoubleBookingFields  
- EqualGames
- BalancedMatchups
- FiftyFiftyHomeAway
- MaitlandBackToBack
- MaitlandAwayClubsLimit
- ClubGradeAdjacency
- PHLSecondGradeAdjacency
- PHLSecondGradeTimes

### 6.2 Game Modification
| Capability | Location | Method |
|------------|----------|--------|
| Move game to new timeslot | `DrawTester` | `move_game(game_id, new_week=, new_day_slot=, ...)` |
| Swap two games' timeslots | `DrawTester` | `swap_games(game_id_1, game_id_2)` |
| Move game to available slot | `DrawTester` | `move_game_to_available_slot(game_id, week)` |
| Reset to original | `DrawTester` | `reset()` |
| Save modified draw | `DrawTester` | `tester.draw.save(path)` |

### 6.3 What-If Analysis
| Capability | Location | Method |
|------------|----------|--------|
| Hypothetical move check | `analytics/tester.py` | `what_if_move_game(draw, data, game_id, ...)` |
| Compare violation counts | `ViolationReport` | Returns violations before/after |

---

## 7. STAKEHOLDER REPORTS

### 7.1 Club Reports
| Output | Description |
|--------|-------------|
| All games for club | Every game involving club's teams |
| Home/away counts | Games home vs away per team |
| Bye weeks | Weeks where team doesn't play |
| Opponent frequency | How many times each opponent faced |
| Field usage | Which fields club plays at |

### 7.2 Grade Reports
| Output | Description |
|--------|-------------|
| All games in grade | Every game in that grade |
| Team game counts | How many each team played |
| Matchup matrix | Team vs team counts |
| Home/away balance | Per-team home/away split |

### 7.3 Compliance Certificate
| Output | Description |
|--------|-------------|
| Per-constraint status | Pass/Fail for each constraint |
| Violation counts | Number of violations by severity |
| Violation details | Specific games/teams violating |
| Timestamp | When certificate generated |

---

## 8. CLI COMMANDS

```bash
# Generate draw
python run.py generate --year 2025 --staged

# Test existing draw
python run.py test draws/v1.json

# Full analytics
python run.py analyze draws/v1.json

# Export analytics Excel
python run.py report draws/v1.json --output analytics.xlsx

# Club report
python run.py club-report draws/v1.json Maitland --output reports/

# Compliance certificate
python run.py cert draws/v1.json --output compliance.xlsx

# Swap games (then check violations)
python run.py swap draws/v1.json G00001 G00050
```

---

## 9. CONFIGURATION

| Setting | Location | Default |
|---------|----------|---------|
| `num_dummy_timeslots` | `config/season_*.py` | 3 |
| `num_rounds` | `config/season_*.py` | Per-grade game counts |
| `max_rounds` | `config/season_*.py` | Season length |
| Penalty weights | Constraint classes | Varies (100-50000) |
| Stage time limits | `main_staged.py` | 1-4 hours per stage |

---

## 10. FILE LOCATIONS

| File | Purpose |
|------|---------|
| `run.py` | CLI entry point |
| `main_staged.py` | Staged solver |
| `core/constraints.py` | All constraint implementations |
| `core/models.py` | Pydantic data models |
| `analytics/storage.py` | DrawStorage, DrawAnalytics, SlotAnalyzer, rev export |
| `analytics/tester.py` | DrawTester, violation checks, game modification |
| `analytics/reports.py` | ClubReport, GradeReport, ComplianceCertificate |
| `config/season_2025.py` | Season configuration |
| `docs/DRAW_RULES.md` | Constraint rule documentation |
