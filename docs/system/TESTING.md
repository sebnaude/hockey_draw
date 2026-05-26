# Testing — the green suite, coverage, and the three real-data assurances

> **Audience:** engineers running or extending the test suite. spec-034 deliverable.

## TL;DR — how to run it

```powershell
# Full batched suite + honest coverage (Windows-safe; never segfaults)
.\.venv\Scripts\python.exe scripts\run_green_suite.py

# Batches only, no coverage (faster)
.\.venv\Scripts\python.exe scripts\run_green_suite.py --no-cov

# Print the batch plan and exit
.\.venv\Scripts\python.exe scripts\run_green_suite.py --list
```

Exit code: `0` green + coverage floor met, `1` a batch had failures/errors, `2`
tests green but the DoD-2 coverage floor not met.

## Why batches (the Windows/ortools segfault)

Running the whole `pytest tests/` in a single process **segfaults on Windows** —
repeated CP-SAT model builds in one interpreter eventually crash it (a known
ortools/Windows interaction). `scripts/run_green_suite.py` therefore runs the
suite as a sequence of **batches, each in a fresh subprocess** (ortools state is
reset between them). Batch 0 is the whole `tests/atoms/` package; the remaining
top-level `tests/test_*.py` files are chunked (8 per batch). Coverage is collected
in `--parallel-mode` (each subprocess writes its own `.coverage.*` data file) and
then `coverage combine`d into one total. Config lives in `.coveragerc`.

## The no-mock policy (hard rule)

Per the convenor: **no mocks, no patches, no monkeypatching, no skip/xfail-to-go-green,
no coverage gaming.** Tests build REAL models (`load_season_data(2026)` / the
forced-free `load_season_data('test')`) or minimal REAL CP-SAT fixtures from real
domain objects, and assert against **hand-computed oracles** — never against the
code's own output. A truthful "this module sits at N%, here is the uncovered
branch and why" beats a fabricated 85%.

Shared real-data fixtures live in `tests/conftest.py`:
`real_2026_data` / `test_season_data` (session-cached load, deepcopy per test so
atom mutations stay isolated) and `clean_real_draw` (the committed
`tests/fixtures/draw_2026_first6weeks.json`, a real 5-week partial draw).

## The three assurances

### A — atoms enforce on real data
Every registered atom (`CONSTRAINT_REGISTRY`) has a no-mock test that builds a real
model and asserts the rule **holds** under a satisfying assignment and **bites**
under a hand-built violating one.
- Most atoms: dedicated files in `tests/atoms/`.
- The 7 that lacked a dedicated file: `tests/atoms/test_spec034_assurance_a_realdata.py`
  drives the **live** `UnifiedConstraintEngine` methods (EqualGamesAndBalanceMatchUps,
  FiftyFiftyHomeandAway, TeamConflict, PreferredTimes) on small real CP-SAT models,
  and the real `generate_X` filter for BlockedGames.
- ForcedGames / LockedPairings: existing real-data tests
  `test_forced_games_count_rules.py`, `test_forced_games_multi_scope.py`,
  `test_locked_pairings_generate.py`.

### B — DrawTester detects a failed constraint
For every `DrawTester` check, a test takes a real (or controlled real) draw, breaks
exactly one rule, and asserts the report flags **that** constraint with the right
count — and reports zero when clean.
- ~18 checks: `test_constraints_realdata.py` (clean-pass + injected-violation),
  `test_tester_nihc_field_fill_order.py`, `test_locked_pairings_unit_c.py`.
- The 6 checks that lacked coverage: `tests/test_tester_detects_failures.py`
  (forced_games, blocked_games, preferred_games, team_pair_no_concurrency via
  injected controlled config on the real partial draw; club_no_concurrent_slot,
  balanced_bye_spacing via small controlled real DrawStorage fixtures).

### C — soft constraints are measured
Bending a soft atom shows up in `report.breakdown.soft_pressure[name]` with the
right hand-computed `total_penalty`; honouring it leaves the atom absent.
- `tests/test_soft_pressure_realdata.py`: PreferredGames, ClubNoConcurrentSlot,
  TeamConflict (the soft atoms whose tester check emits a `metric_value` — the
  only thing `soft_pressure` rolls up).
- Soft atoms with **no tester check** (`*RegenSoft` analogues, `SoftLexMatchupOrdering`)
  are measured at the **solver** level: their `data['penalties'][...]` buckets reach
  the `Maximize` objective. That wiring is proven by the live-engine penalty-bucket
  assertions in `tests/atoms/test_spec034_assurance_a_realdata.py` (TeamConflict /
  PreferredTimes) and the spec-027/033 regen tests.

> `over_limit` counts metric-bearing violations; `at_limit` is unused (always 0) —
> never assert on it.

## Coverage floor (DoD-2)

Floor: **≥85%** on the surfaces that matter — `constraints/atoms/`,
`constraints/registry.py`, `constraints/stages.py`, `analytics/tester.py` — measured
by the runner's `--include` filter against the combined data.

### Honest numbers (spec-034 green run, 2026-05-26)

Combined DoD-2 floor surfaces (branch coverage): **82.2%** — below the 85% target.
Per-surface:

| Surface | Coverage | Note |
|---|---|---|
| `constraints/registry.py` | 92.2% | above floor |
| `constraints/stages.py` | 93.1% | above floor |
| `constraints/atoms/*` | mostly 84–100% (a few 73–81%) | near/above floor; the lower ones are the `*RegenSoft` analogues' rarely-hit branches |
| `analytics/tester.py` | **73.1%** | the shortfall driver — see below |

**Why the total sits at 82.2%, not ≥85% (honest sub-floor, per DoD-2):**
`analytics/tester.py` is a 1311-statement module, but a large share of it is
**report/export/formatting** code (HTML/Excel/console violation reports, severity
rollups, pretty-printers) rather than the constraint *checks*. The spec-034
assurances exercise the **check** surface (detection + soft-pressure) thoroughly,
but the report-rendering paths (uncovered ranges incl. ~929–1120, 1128–1159,
2401–2519, 2623–2738) are not driven by the constraint tests and would need
report-rendering tests to lift. Those are **not** padded in here — per the no-cheat
rule, a truthful 73.1% with the reason stated beats trivial getter tests gaming the
number. Raising it is a clean follow-up (render-path tests), not a spec-034 deliverable.

Re-measure any time with `python scripts/run_green_suite.py` (the "DoD-2 FLOOR
surfaces" block).
