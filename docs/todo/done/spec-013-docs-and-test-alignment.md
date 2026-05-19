<!-- status: done -->
<!-- owner: session=spec-013-orchestrator claimed=2026-05-19T08:48:57Z completed=2026-05-19T09:35:00Z -->
<!-- depends_on: none -->

# spec-013 — Doc drift fixes + atom-test GWT compliance pass

## Why

Three doc drifts and one test-suite weak-spot were found during a final-form
audit (2026-05-19):

1. **`docs/system/STAGES.md` lines 23–75** ships a hardcoded
   `DEFAULT_STAGES` example that is severely stale: it lists six obsolete
   atoms removed by specs 002–008/012 (`FiftyFiftyHomeandAway`,
   `ClubGradeAdjacency`, `ClubVsClubCoincidence/FieldLimit/DeficitPenalty`,
   `PHLAnd2ndBackToBackSameField`) and is missing eight atoms that landed in
   specs 002/003/005/006/007/008/012. The actual live list in
   `config/defaults.py::DEFAULT_STAGES` (lines 144–233) is correct. STAGES.md
   is the engineering SSoT for stage configuration; downstream agents reading
   it for stage authoring will be misled.

2. **`docs/todo/GOALS.md` lines 121–138** spec table has three duplicate
   rows: spec-008 (lines 130 and 135), spec-011 (133 and 136), spec-012
   (134 and 137). One pair (spec-012) even has slightly different wording.
   The duplicates were introduced during sequential spec sign-offs.
   Cosmetic but it's the source-of-truth table.

3. **`docs/operator-human/RULES.md` line 77** references the legacy class
   name `PHLAndSecondGradeTimes`. Per the doc-category rules
   (`docs/README.md`), operator-human docs describe rules in plain English
   and must not name atoms or constraint classes. The class was atomized in
   Phase 3a; the name now points at archived code.

4. **`tests/atoms/test_phl_atoms.py:203–229` (`TestPreferredDates`)** is the
   only atom test that violates the /basic test bar: no explicit
   Given/When/Then blocks, no hand-computed oracle. The two assertions are
   loose lower bounds (`n >= 1`, `len(...) >= 1`) — they don't pin the
   exact penalty count, so a behaviour drift that produced *too many*
   penalties would pass silently.

Cost of not fixing: stale SSoT docs will mislead the next agent writing a
stage config (drift #1 is materially wrong, not cosmetic); the test gap
hides a class of regressions in the soft `PreferredDates` atom.

## Definition of Done

1. `docs/system/STAGES.md` no longer contains a hardcoded `DEFAULT_STAGES`
   listing. The "Default stages" section instead describes the shape and
   intent of each stage in prose, and links to
   `config/defaults.py::DEFAULT_STAGES` (with line number) as the single
   source of truth. Verified by reading the rendered Markdown end-to-end.
2. `docs/todo/GOALS.md` spec table contains exactly one row per spec ID
   (12 rows total, spec-001 through spec-012). Verified by grep:
   `grep -c "^| spec-" GOALS.md` == 12, no duplicate IDs.
3. `docs/operator-human/RULES.md` no longer contains the string
   `PHLAndSecondGradeTimes`. The rule description on line ~77 is rephrased
   in plain English without naming the class. Verified by grep returning 0.
4. `tests/atoms/test_phl_atoms.py::TestPreferredDates` is rewritten so
   every test method has:
   - a named scenario in the method docstring (`Scenario: …`),
   - explicit `# Given`, `# When`, `# Then` (and `# And` where applicable)
     comment blocks,
   - a hand-computed oracle calculated in a comment (showing the math),
   - exact-equality assertions on the oracle (no `>= 1` style assertions).
5. Every other atom test in `tests/atoms/test_*.py` is audited against the
   GWT-with-oracle bar. Files that already meet the bar are left alone.
   Files that miss it (no GWT structure OR oracle is a loose bound) are
   tightened in this same unit. Audit findings are recorded as a one-line
   comment header at the top of each tightened file: `# spec-013: GWT pass`.
6. `pytest tests/atoms/ -v` passes — every atom test green, no skips that
   weren't there before. Full suite (`pytest tests/`) is also green.
7. Every doc touched in DoD #1–3 is updated to match its new state; nothing
   in `docs/` still references the deleted strings.
8. All work merged into `final-form`, pushed to origin, worktrees deleted.

## Implementation units

Four independent units; each touches disjoint files so all four worktrees
can run in parallel with zero merge conflict.

### Unit A — STAGES.md realignment

**Files touched:** `docs/system/STAGES.md` (only).

**Change summary:**
- Replace the hardcoded `DEFAULT_STAGES` Python block (current lines 25–75)
  with: (a) a one-paragraph note that the live list lives in
  `config/defaults.py::DEFAULT_STAGES` (current line 144) and must not be
  duplicated here; (b) a prose summary of each stage's *intent* (what each
  stage exists to enforce — not the atom list), so a reader gets the shape
  without depending on a doc that could drift.
- Leave the schema table (lines 9–22) and API/dispatcher sections (80+) as
  they are — those are intrinsic to the doc, not drift-prone.

**Depends on:** nothing.

**Test outline (Given/When/Then):**
- *Scenario: STAGES.md no longer claims facts about specific atom names.*
- Given: `docs/system/STAGES.md`.
- When: grep for any atom name that exists in the obsolete-set
  (`FiftyFiftyHomeandAway`, `ClubGradeAdjacency`, `ClubVsClubCoincidence`,
  `ClubVsClubFieldLimit`, `ClubVsClubDeficitPenalty`,
  `PHLAnd2ndBackToBackSameField`).
- Then: zero matches (hand-computed oracle: zero, because we deleted the
  hardcoded block).
- And: grep for `config/defaults.py::DEFAULT_STAGES` returns ≥ 1 (the
  pointer is now present).

### Unit B — GOALS.md spec-table dedup

**Files touched:** `docs/todo/GOALS.md` (only).

**Change summary:**
- Delete the three duplicate rows (currently lines 135, 136, 137 in the
  spec table). Keep the first occurrence of each spec ID (lines 130, 133,
  134). Confirm the kept wording is the better of the two duplicates for
  each spec (for spec-012 the line-134 wording "alternation" is more
  precise than line-137 "preference (verify + wire up)").
- No other content change.

**Depends on:** nothing.

**Test outline (Given/When/Then):**
- *Scenario: spec table has exactly one row per spec ID.*
- Given: `docs/todo/GOALS.md`.
- When: extract spec IDs from the table via grep `^| spec-\d+`.
- Then: 12 rows total (spec-001 through spec-012), each ID appearing
  exactly once. Hand-computed oracle: 12 unique IDs.

### Unit C — RULES.md legacy class name removal

**Files touched:** `docs/operator-human/RULES.md` (only).

**Change summary:**
- Locate the line(s) referencing `PHLAndSecondGradeTimes` (line ~77 per
  audit). Rewrite the surrounding rule description in plain English: state
  what the rule does for the convenor (PHL and 2nd grade games at
  Broadmeadow run back-to-back on the same field at restricted times)
  without naming the class.
- Confirm no other legacy class name appears in `docs/operator-human/` or
  `docs/operator-ai/`.

**Depends on:** nothing.

**Test outline (Given/When/Then):**
- *Scenario: operator-human docs name no atom/class.*
- Given: every `.md` file under `docs/operator-human/`.
- When: grep for `PHLAndSecondGradeTimes`, `ClubDayConstraint`,
  `ClubVsClubAlignment`, `MaitlandHomeGrouping`, `AwayAtMaitlandGrouping`,
  `FiftyFiftyHomeandAway`, `MaitlandHomeFridayCount`, `PHLRoundOnePlay`.
- Then: zero matches across all files. Hand-computed oracle: 0 (we
  rewrote the only known offender; the audit found no others).

### Unit D — Atom-test GWT compliance pass

**Files touched:** every file under `tests/atoms/test_*.py` (audit). At
minimum `test_phl_atoms.py` (TestPreferredDates rewrite). Any other file
the audit flags as below the bar.

**Change summary:**
- For each `test_*.py` in `tests/atoms/`:
  1. Read every `def test_…` method.
  2. Check: does it have a `Scenario:` line in its docstring + `# Given /
     # When / # Then` (and `# And` where used) comment blocks + a
     hand-computed oracle comment + exact-equality assertions?
  3. If yes, leave alone.
  4. If no, rewrite the method (and only that method) so it does. The
     oracle math must be in the test as a comment, computed from the
     fixture's inputs — *not* copied from the code's current output.
- Append `# spec-013: GWT pass` as a header comment to every file
  that was tightened (so a future audit can grep for the audit signature
  and reproduce it).
- The known offender is `TestPreferredDates`; the rest of the audit
  expectation per the prior survey is "uniformly follow GWT" — confirm
  during execution, do not assume.

**Depends on:** nothing.

**Test outline (Given/When/Then):**
- *Scenario: every atom test reaches /basic standard.*
- Given: every `tests/atoms/test_*.py`.
- When: pytest is run against `tests/atoms/`.
- Then: 100% pass, no new skips, no new xfails. And: every test method
  containing the keyword `assert` has, within ±20 lines, the literal text
  `# Given`, `# When`, `# Then` (grep check).
- Hand-computed oracle for `TestPreferredDates::test_records_penalty_for_preferred_dates`:
  the fixture seeds two preferred dates (2026-03-22 and 2026-03-29) on a
  PHL-only model. Count of slot/date combinations where the existing
  fixture would produce a variable on either date == N. Penalty entries
  count == N (one per preferred-date slot consumed). Compute N by hand
  from the fixture's slot dict and assert exact equality.

## Doc registry

| Doc file | What changes |
|---|---|
| `docs/system/STAGES.md` | Replace hardcoded DEFAULT_STAGES block with prose + pointer to `config/defaults.py:144`. (Unit A) |
| `docs/todo/GOALS.md` | Remove three duplicate spec-table rows. (Unit B) |
| `docs/operator-human/RULES.md` | Remove `PHLAndSecondGradeTimes` legacy class name, rephrase in plain English. (Unit C) |
| `docs/todo/spec-013-docs-and-test-alignment.md` | This plan; status header flips `ready → in_progress → done` and the file moves to `done/` when all four units land. (All units) |
| `docs/todo/GOALS.md` (second touch) | Add spec-013 row at the bottom of the spec table with status `done`. (Closing edit, separate from Unit B) |

No code documentation outside `docs/` is affected. `CLAUDE.md`,
`README.md`, and all `tests/`-internal documentation are confirmed correct
by the audit.

## Out of scope

- **Norths v Wests 12–14 June 2026 weekend** (`docs/seasonal/2026/operational_TODO.md:31–41`).
  This is a forced-game configuration item, not a constraint-code change.
  Belongs in season config / convenor data work. Spawn a separate plan
  only if convenor asks for it.
- **Week 1 external draw import** (`docs/seasonal/2026/operational_TODO.md:3–7`).
  Pure data work, no constraint change.
- **`config/season_2026.py:660` TODO** to move PHL state-championship limit
  into `CONSTRAINT_DEFAULTS`. Config-folder TODOs are out of scope per user
  direction (configs are not docs).
- **Non-atom test files in `tests/`** (e.g. `test_models.py`,
  `test_constraint_registry.py`, integration tests). The /basic test
  standard targets *constraint atom tests*. Non-atom tests are
  infrastructure / smoke / registry checks; the survey confirmed they use
  real objects (no mocks) and that's enough.
- **Adding tests for atoms that already have bundled coverage** (the
  earlier audit's "missing test" list was a false alarm — `test_phl_atoms.py`,
  `test_club_vs_club_atoms.py`, `test_nihc_field_fill_order.py` already
  cover the atoms in question by class name).
- **Repo-root `CLAUDE.md`** — confirmed aligned by the audit; do not touch.
- Any **new** constraint, atom, or stage. This plan is strictly a doc +
  test-quality cleanup; no behavioural code change.
