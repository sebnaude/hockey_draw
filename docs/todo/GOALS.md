# Project Goals

The north star for the hockey draw scheduler. Read this before any non-trivial change so the work fits the shape we're building toward, not just the immediate ask.

This is the *why*. For the *how*, see the doc map at the bottom.

---

## 1. Product goal

**Generate a publishable hockey season draw that the convenor can hand to clubs with confidence, in one solver run, with every constraint either satisfied or visibly tracked as a known soft penalty.**

Concretely:
- ~47 teams across 6 grades (PHL, 2nd–6th), 10 clubs, multiple venues.
- 22 Sunday rounds + 2 Friday-only PHL weeks, across a 27-week calendar window.
- Output: a versioned draw (JSON + xlsx + per-club + Revo CSV) that the convenor publishes externally.
- Hand-edits after publish are first-class: every change goes through `scripts/test_scenario.py` before promotion, and bumps the MINOR version. Solver re-runs bump MAJOR.

Anything that does not directly serve "publishable draw + auditable changes" is a side quest.

---

## 2. Engineering goal — the atomization model

The codebase is moving from **monolithic "do-everything" constraint classes** to a **registry of atomic constraints, dispatched from a config-driven stage plan**. As of `final-form` this is **shipped**; the goal now is *protecting it* and *extending it the same way*.

### What "atomization" means here

Every rule in the schedule (e.g. "PHL and 2nd grade games at Broadmeadow run back-to-back on the same field") is decomposed into the smallest independently-meaningful unit. Each unit is:

- A **named atom** in `constraints/atoms/` — one file per atom, one purpose per atom.
- **Registered** in `constraints/registry.py` with metadata (severity, slack key, helper-var declarations, the legacy class it descended from).
- **Dispatched** from a **`SOLVER_STAGES`** list in `config/defaults.py` (or a season override / `--stages-config` JSON), via `constraints/stages.py::apply_solver_stage`.
- **Tested in isolation** in `tests/atoms/` against a small CP-SAT fixture, plus a parity test that pins behaviour against the pre-atomization implementation.
- **Visible in the violation report** with per-club / per-type / soft-pressure breakdowns when violated.

Helper variables (coincidence BoolVars, count-summing IntVars) are declared via `HelperVarRegistry` so two atoms asking for the same helper share one variable — no hand-rolled duplicates.

### Why this matters

The legacy `constraints/original.py` and `constraints/ai.py` (now archived) had ~1700 and ~2000 lines of constraints that did 3–5 things each. That made it impossible to:

- **Skip one piece of a rule** without skipping the whole rule.
- **Diagnose infeasibility** beyond "it's somewhere in this 200-line method."
- **Slack a single dimension** without writing a hacked-in conditional.
- **Test a rule** without booting a full season fixture.
- **Reuse helper variables** across rules — every constraint built its own.

Atomization made each of these one-line operations: skip an atom by name, drop it from a stage, slack via its registry key, test it standalone, share helpers via the registry. The whole point of the refactor.

### Status (as of 2026-04-29)

✅ All phases shipped — see `docs/ATOMIZATION_PLAN.md` for the phase table and `docs/ATOMIZATION_HANDOFF.md` for the commit-by-commit chain. Test bar: **1287 passed, 1 skipped**.

Three monolithic constraints have been atomized so far (14 atoms across 3 clusters):

| Cluster | Atoms | Source class (archived) |
|---|---|---|
| PHLAndSecondGradeTimes | 5 | `original.py::PHLAndSecondGradeTimes` |
| ClubDayConstraint | 5 | `original.py::ClubDayConstraint` |
| ClubVsClubAlignment | 4 | `original.py::ClubVsClubAlignment` |

Plus FORCED/BLOCKED count adjusters (Phase 4), generic non-default-home rules (Phase 6), and config-driven solver stages with CLI flags (Phase 7b).

### How we handle atomization going forward

**Default rule: every new behaviour is an atom.** No new code goes into `original.py` / `ai.py` — they're in `constraints/archived/` and a test (`tests/test_no_legacy_imports.py`) actively blocks production imports of them.

When you need to add or change a constraint:

1. **Identify the smallest unit.** If the rule has two clauses you could imagine slacking independently, that's two atoms.
2. **Write the atom** in `constraints/atoms/<descriptive_name>.py`. Subclass `Constraint` (from `constraints.atoms.base`). One `apply(model, X, data)` method. Skip dummy keys (`len(key) < 11`) and locked weeks (`key[6] <= data.get('current_week', 0)`).
3. **Declare helpers** via `data['helper_registry']` (a `HelperVarRegistry`), not by creating BoolVars/IntVars directly. See `docs/HELPER_VARS.md`.
4. **Register** in `constraints/registry.py` with: canonical name, severity, slack key (if applicable), helper-var catalog entries it consumes.
5. **Wire into a stage** in `config/defaults.py::DEFAULT_STAGES` (or whatever stage is appropriate). Use the canonical name.
6. **Test in isolation** in `tests/atoms/test_<name>.py` against a tiny CP-SAT fixture. Add a parity test if there's a legacy implementation to match.
7. **Update docs**: `CONSTRAINT_INVENTORY.md` (the SSoT table), `STAGES.md` if you added a stage, and this file if the goal shifts.

**Atomizing a remaining monolith** (anything in `constraints/archived/` that the registry still aliases to a single legacy class) follows the same pattern as Phases 3a/3b/3c: read the original code, list the independent behaviours, write one atom each, add a parity test against the original, wire into stages, then mark the legacy entry as "atomized" in `CONSTRAINT_INVENTORY.md`.

**Never re-monolithize.** If two atoms have shared logic, extract a helper (see `constraints/atoms/_club_day_shared.py` for the pattern), don't merge the atoms. The merge always feels right in the moment and is always regretted three months later.

---

## 3. Operational goals

The system is run by one convenor (you), with occasional second-opinion solver runs from AI agents. So:

- **Configs over code.** Anything a convenor might tune season-to-season (stages, slack, forced/blocked games, venue rules, game-time dicts) lives in `config/`, not hardcoded in solver classes. See `docs/CONFIGURATION_REFERENCE.md` and `docs/ai/CONFIGURATION_REFERENCE.md`.
- **Solver runs are observable.** Every run writes checkpoints (`checkpoints/run_XX/`), logs (`logs/`), and a versioned draw with full metadata (mode, stages, slack, forced/blocked outcomes, penalties). Never re-run blind.
- **Hand-edits are auditable.** `scripts/test_scenario.py` produces a BEFORE/AFTER violation diff for any proposed change before it touches `current.json`.
- **Tests are real.** No mocking of solver state. Real CP-SAT models with small sampled fixtures. The test bar is a tripwire — every phase must hold or lift it.

---

## 4. What success looks like

You can — without thinking too hard:

1. Add a new constraint by writing one file in `constraints/atoms/`, one registry entry, one test, one stage entry.
2. Skip one piece of a rule for one season by removing one atom name from one stage.
3. Loosen one dimension of a rule by passing `--slack N` and knowing exactly which atoms read that slack.
4. Diagnose an infeasibility down to a single cluster via `run.py diagnose --stage X`.
5. Apply a convenor hand-edit, see the violation diff, promote it as a MINOR version, all in one script.
6. Hand a new convenor (or AI agent) `CLAUDE.md` + this file + `docs/ATOMIZATION_PLAN.md` and have them productive in an hour.

If any of these starts to feel hard, that's the signal that something has drifted from the model. Fix the drift before adding more features.

---

## Doc map

The doc map lives in [`docs/README.md`](../README.md). It groups every doc into six categories (operator-human, operator-ai, system, reports, seasonal, todo) with read/update rules per category. Read it once; refer back when you don't know where a new doc belongs.

---

## Specifications

Each spec below describes a target behaviour. Some are partially implemented, some not started. Each spec has a paired implementation plan in `docs/todo/spec-*.md` carrying the /basic status header (`not_ready | ready | in_progress | done`). When an implementation plan ships, this section gets its status updated and the plan file moves to `docs/todo/done/`.

| ID | Spec | Plan file | Status |
|---|---|---|---|
| spec-001 | Rounds 1–2 Broadmeadow-only rule must exempt FORCED games | `done/spec-001-r1r2-broadmeadow-forced-exempt.md` | done |
| spec-002 | Soft lexicographical ordering of matchups across the season | `done/spec-002-soft-lex-ordering.md` | done |
| spec-003 | Field-fill ordering (WF → EF → SF) replaces "last-game-WF" rule | `done/spec-003-field-fill-order.md` | done |
| spec-004 | Away-club home/away atoms aligned with FORCED-Friday math | `spec-004-away-club-home-counts.md` | ready |
| spec-005 | ClubVsClubAlignment stacks lower grades into upper-grade weekends with club-day-like co-location | `spec-005-clubvsclub-stacking.md` | ready |
| spec-006 | Preferred / non-preferred weekends for away grounds (NRL-overlap–style) | `spec-006-preferred-weekends-away-grounds.md` | ready |
| spec-007 | Adjacent-grade hard atom → soft constraint; keep only same-grade-same-club hard | `done/spec-007-adjacent-grade-soft-replace.md` | done |
| spec-008 | Spacing: byes as first-class + intuitive gap semantics | `spec-008-spacing-byes-and-intuitive-gap.md` | ready |
| spec-009 | FORCED count rules + adjusters — end-to-end verification | `spec-009-forced-count-rules-verification.md` | ready |
| spec-010 | Remove `PHLRoundOnePlay`; verify R1/R2 carve-out for 2nd grade | `spec-010-first-week-rework.md` | ready |
| spec-011 | `ClubVsClubFieldLimit` penalty scales with games on 2nd field; double-up handling | `spec-011-clubvsclub-field-penalty-scaling.md` | ready |
| spec-012 | Team time preferences + Maitland H/A weekend preference (verify + wire up) | `spec-012-team-time-and-home-preferences.md` | ready |

### spec-001 — Rounds 1–2 Broadmeadow-only rule must exempt FORCED games

**Current:** `PERENNIAL_BLOCKED_GAMES` in `config/defaults.py` removes every non-NIHC variable in rounds 1 and 2. This collides with FORCED entries that the convenor places in those rounds (e.g. an opening-round Maitland-vs-Norths PHL game forced to a specific date/field at Maitland Park).

**Target:** The "rounds 1–2 at Broadmeadow only" rule must respect FORCED entries — if a variable matches both a `PERENNIAL_BLOCKED` scope and a `FORCED_GAMES` scope, the FORCED entry wins (the variable is kept).

**Why:** The convenor uses FORCED to express deliberate exceptions; perennial blocks should be the *default*, not absolute. Without this, the convenor has to comment out the perennial rule for the season, defeating the whole "perennial" framing.

**How to apply:** Per-atom registry note that this exemption exists; engine resolution rule that FORCED ⊕ BLOCKED → FORCED in conflict.

---

### spec-002 — Soft lexicographical ordering of matchups across the season

**Target:** In lieu of any other influence, the order in which two teams meet across the season is alphabetical by their concatenated key. With no other forcing, Norths-vs-Wests's first meeting is week K rather than week K+N, simply because N-W comes earliest alphabetically among that round's matchups.

**Why:** Predictability for clubs reading the published draw. Removes solver-arbitrary "this matchup is in week 7 for no reason" outcomes. Pure tie-break, never overrides a real constraint.

**How to apply:** Soft penalty weighted very low — every later occurrence of a lexicographically-earlier matchup adds a tiny penalty. Goes in the soft optimisation stage; never blocks feasibility.

---

### spec-003 — Field-fill ordering (WF → EF → SF) replaces "last-game-WF" rule

**Current perennial rule:** "If only one game is being played in the last slot of the day at NIHC, it must be on WF." Implemented as a tester rule + manual review.

**Target:** Scrap the single-game-last-WF special case. Replace with a strict field-fill ordering at NIHC for every slot: **WF must be filled before EF; EF must be filled before SF.** The SF rule emerges for free since SF is the lowest priority — we only need to force the first two.

**Why:** Field-fill ordering is the general principle; the last-game-WF rule was a special-case approximation of it. Generalising removes a perennial-rules-doc footnote and gives correct behaviour for partial slots throughout the day, not just the last one.

**How to apply:** Two atoms — `NIHCFillWFBeforeEF` and `NIHCFillEFBeforeSF` — each per (date, day_slot) at NIHC, asserting "use of higher-numbered field ⇒ lower-numbered field also used." Hard constraint; deletes the `_check_west_field_last_slot` tester rule.

---

### spec-004 — Away-club home/away atoms aligned (with FORCED-Friday math)

**Target:** For any away-based club (Maitland, Gosford, future expansions), two atoms cooperate to produce exact home/away balance:

1. **`AwayClubHomeWeekendsCount` atom.** Compute the maximum required home games for any team in the away-based club. Force the number of weekends at that home ground to equal exactly that number. *A Friday game counts as the weekend it's part of.*

   PHL-Friday math: the naive max is `max(per_grade_games_required)` — but PHL has forced Friday games that don't consume Sunday weekend slots. So the true sundays-required is `max(phl_required - forced_phl_fridays, max(other_grades_required))`. The adjuster must read FORCED_GAMES, sum the per-club Friday count (handling the case where one variable matches multiple forced scopes — e.g. `count==2 Friday Maitland total` plus `count==1 Friday Maitland vs Tigers` represent two entries in the dict but the same single Friday game).

2. **`AwayClubPerOpponentAndAggregateHomeBalance` atom.** For each team in an away-based club, force:
   - For each opponent in its grade: home games against that opponent ∈ \[half_total_meetings, half_total_meetings] expressed as a constrained IntVar that lands on the correct integer for odd/even totals automatically.
   - Across all its games: home games ∈ \[half_total, half_total] same construction.

   The intersection of the two enforces the best outcome. Note the ±1 isn't literal — it's `model.Add(home_games >= floor(total/2)); model.Add(home_games <= ceil(total/2))`; the integer constraint picks the correct exact value based on parity.

**Why:** Today, two separate atoms (`NonDefaultHomeGrouping`, `FiftyFiftyHomeandAway`) approximate this but miss the FORCED-Friday case, leading to over-counted weekends and sparse Sundays. Codifying both as registry atoms with shared adjuster math fixes it once for all away clubs.

**How to apply:** Two atoms in `constraints/atoms/`, both consuming a new `away_club_required_sundays(data, club)` helper that does the FORCED-aware math. Adjuster must also handle the case where a FORCED entry sets a *total sum* (e.g. `count==2 sum of Maitland Fridays`) alongside per-pair forced games — do NOT double-count.

---

### spec-005 — ClubVsClubAlignment stacks lower grades into upper-grade weekends with club-day-like co-location

**Target:** Replace the current ClubVsClubAlignment cluster with a precise stacking model.

For each unordered pair of clubs `(A, B)`, compute the per-grade meeting counts: e.g. `{PHL: 4, 2nd: 3, 3rd: 2, 4th: 2, 5th: 1, 6th: 0}`. Force a *stacked* weekend layout:

- Number of weekends where **all six grades** that play each other do play each other = `min(per_grade_counts excluding zeros)` — in the example, `min(4, 3, 2, 2, 1) = 1` weekend with all 5 grades together.
- Number of weekends where exactly the **next strictest subset** plays = `next_min - first_min`, and so on, peeling off the smallest count at each layer.
- Net: in the example, 1 weekend with {PHL, 2nd, 3rd, 4th, 5th}, 1 weekend with {PHL, 2nd, 3rd, 4th}, 1 weekend with just {PHL, 2nd}, 1 weekend with just {PHL}. Total 4 PHL weekends, 3 with 2nd, 2 with 3rd/4th, 1 with 5th — matches the input counts.

**Co-location rules** on each stacked weekend: the matched grades' games for that club-pair are subject to the same conditions as a club day — **back-to-back timeslots, no gaps, same field.**

**Cases to handle:**
- A club with multiple teams in the same grade: treat each as participating; the count is the number of distinct matchups, not teams.
- Two clubs each with two teams in the same grade: behaves like four single-team match-ups — the count is just higher.

**PHL Friday handling:** PHL Friday-night forced games consume the matchup count but cannot satisfy the Sunday stacking requirement. Build a `phl_sunday_available_meetings(club_a, club_b)` helper that = `total_phl_meetings - sum(FORCED Friday-night entries scoping this pair)`. Stack only over the Sunday-available count. Critically: this requires reading FORCED_GAMES, not just counting model variables — variables exist for all rounds, but FORCED removes their freedom.

**Why:** Stacked matches mean clubs travel once per matchup-weekend. Co-location means a parent watching multiple grades of their kid's club versus the same opponent does so contiguously on one field. Current implementation gives only the loose "weekends where ≥X grades coincide" — not the precise stack-down-from-PHL structure.

**How to apply:** Single new atom cluster `ClubVsClubStackedAlignment` replacing `ClubVsClubCoincidence` + `ClubVsClubFieldLimit` + `ClubVsClubDeficitPenalty` + `PHLAnd2ndBackToBackSameField`. New helper kind for "is this club-pair playing this grade this weekend" + the existing club-day same-field / contiguous-slot helpers.

---

### spec-006 — Preferred / non-preferred weekends for away grounds

**Target:** A new soft constraint that lets the convenor declare per-(venue, date) **preferences** for away-team scheduling. Two modes:

- **Prefer this venue on this date** — soft penalty when matching dates *don't* have games at that venue (e.g. "the Knights NRL match is at Maitland this weekend, prefer non-home-club games AWAY at Knights' Maitland Park while traffic is already there").
- **Prefer NOT this venue on this date** — soft penalty when matching dates *do* have games at that venue (e.g. NRL Knights home games at Maitland Park — Maitland doesn't want to play that day).

**Optional field-level granularity:** the constraint can target a specific field at the venue, not just the venue itself.

**Why:** Today, NRL-Knights home games are tracked in `seasonal/2026/operational_TODO.md` as "Need to verify where/how this is enforced" — the answer is "they aren't, except via convenor manual review." This spec gives a first-class constraint.

**How to apply:** Soft constraint atom + JSON config scaffold (likely a new `PREFERRED_WEEKENDS` list in season config, or extension of FORCED/BLOCKED with `soft: true` + `weight`). The user notes the scaffold for forced/blocked games "should already exist" — confirm + extend rather than duplicate.

---

### spec-007 — Adjacent-grade hard atom → soft constraint; keep only same-grade-same-club hard

**Target:**
- **Remove** the hard atom that prevents teams in *adjacent* grades within the same club from playing at the same time.
- **Keep** the hard atom: if a club fields multiple teams in the **same** grade, those teams must not play at the same time.
- **Reintroduce** a soft constraint: the convenor can specify two named teams that "should not play at the same time" and the solver tries to respect it.

**Why:** Adjacent-grade-same-time was over-restrictive — convenor experience says many parents handle kids in adjacent grades fine with overlapping slots, and forbidding it hard caused infeasibility on tight weeks. Same-grade-same-club is genuinely fundamental (one parent literally cannot be at both). The soft per-pair version covers the real-world cases (siblings in non-adjacent grades, etc.) without baking in the assumption.

**How to apply:** Demote `ClubGradeAdjacency` adjacent-grade hard block to soft (or delete the hard part entirely if separate). Add new `TeamPairNoConcurrency` soft atom reading from a new `TEAM_PAIR_NO_CONCURRENCY` config list.

---
