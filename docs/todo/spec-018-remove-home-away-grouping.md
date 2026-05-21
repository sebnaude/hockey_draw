<!-- status: ready -->
<!-- owner: session=none claimed=none -->
<!-- depends_on: none (LARGEST shared-file footprint — touches registry.py, unified.py, stages.py, severity.py, tester.py, _adjusters.py, defaults.py. Sequence LAST relative to spec-014..017 and rebase before merge.) -->

# spec-018 — Delete the venue back-to-back / alternation rules: `NonDefaultHomeGrouping`, `AwayAtNonDefaultGrouping`, `MaitlandAlternateHomeAway`

## Why

The convenor does not want the solver enforcing anything about the *sequence* of an
away-based club's home/away weekends — back-to-back home weekends are fine, long away runs
are fine. Three constraints encode that unwanted sequencing/grouping behaviour and are all
to be removed:

- **`NonDefaultHomeGrouping`** (alias `MaitlandHomeGrouping`; solver classes
  `MaitlandHomeGrouping*`, `MaxMaitlandHomeWeekends*`) — sliding-window hard ban on
  consecutive home weekends per away-based club (severity 1).
- **`AwayAtNonDefaultGrouping`** (alias `AwayAtMaitlandGrouping`) — hard cap on the number of
  distinct away clubs visiting a non-default venue in one weekend (severity 2).
- **`MaitlandAlternateHomeAway`** (spec-012, soft) — penalises consecutive playable-week pairs
  that are both-home (HH) or both-away (AA), pushing an H-A-H-A pattern. Its whole purpose is
  the sequencing the convenor is discarding; with `NonDefaultHomeGrouping` gone its HH branch
  would be the only HH pressure, but the convenor doesn't want HH discouraged at all — so it
  goes too.

**Explicitly KEPT** (the "others still required" — these are home/away *counts and balance*,
not sequencing): `AwayClubHomeWeekendsCount` and `AwayClubPerOpponentAndAggregateHomeBalance`
(spec-004). They enforce per-club home-weekend totals (Friday/Sunday/total) and per-pair +
aggregate 50/50 home/away balance. Untouched.

Removal must be surgical: these three carry precomputed engine maps, FORCED adjusters, tester
checks, slack keys, severity entries and a penalty-weight key.

## Definition of Done

1. **Registry** (`constraints/registry.py`): remove `NonDefaultHomeGrouping`,
   `AwayAtNonDefaultGrouping`, their alias entries `MaitlandHomeGrouping` /
   `AwayAtMaitlandGrouping`, and `MaitlandAlternateHomeAway`.
2. **Stages** (`constraints/stages.py`): remove from `ENGINE_HARD_KEYS`
   (`MaxMaitlandHomeWeekends`, `MaitlandHomeGrouping`, `AwayAtMaitlandGrouping`),
   `ENGINE_SOFT_KEYS` (`MaitlandHomeGrouping`, `AwayAtMaitlandGrouping`), and
   `_ENGINE_KEY_ALIASES` (`NonDefaultHomeGrouping`, `AwayAtNonDefaultGrouping`).
3. **Engine** (`constraints/unified.py`): delete methods `_maitland_grouping_hard/_soft`,
   `_away_maitland_hard/_soft`; their dispatch blocks in `apply_stage_1_hard` /
   `apply_stage_2_soft`; and the precomputed maps `non_default_all_week`,
   `non_default_home_week`, `non_default_away_club_week` (+ the `maitland_*` aliases) plus
   their population in `build_groupings()`. No dangling references remain.
4. **Adjusters** (`constraints/atoms/_adjusters.py`): delete `maitland_home_grouping_adjuster`,
   `away_at_maitland_grouping_adjuster`, and their registry wiring lines.
5. **Severity** (`constraints/severity.py`): remove the 6 entries
   (`MaxMaitlandHomeWeekends`, `MaitlandHomeGrouping`, `AwayAtMaitlandGrouping` + `AI`
   variants). `test_every_severity_entry_maps_to_registry` still passes.
6. **Tester** (`analytics/tester.py`): remove `_check_maitland_back_to_back` and
   `_check_maitland_away_clubs_limit` + their dispatcher registrations. Registry no longer
   references these methods; `test_every_drawtester_check_in_registry` /
   `test_every_tester_check_method_covered` still pass.
7. **Atom file**: delete `constraints/atoms/maitland_alternate_home_away.py`; remove from
   `constraints/atoms/__init__.py` (import / `__all__` if present) and any direct import in
   `unified.py`/`stages.py`. (`NonDefaultHomeGrouping`/`AwayAtNonDefaultGrouping` are
   legacy-engine, not atom files, so no atom file to delete for those.)
8. **Config** (`config/defaults.py`): remove `NonDefaultHomeGrouping`,
   `AwayAtNonDefaultGrouping` from `DEFAULT_STAGES` `home_away_balance`; remove
   `MaitlandAlternateHomeAway` from `soft_optimisation`; remove
   `CONSTRAINT_DEFAULTS['maitland_max_consecutive_home']` and `['away_maitland_max_clubs']`;
   remove the `max_consecutive_home`/`max_away_clubs` keys from `AWAY_VENUE_RULES` (keep the
   dict + any Friday-game keys); remove `PENALTY_WEIGHTS['maitland_alternate_home_away']`
   (in defaults and `season_2026.py`).
9. `len(CONSTRAINT_REGISTRY)` decremented by the removed entries; count test updated.
   `validate_solver_stages(DEFAULT_STAGES)` == `[]`; full suite green. A generated draw no
   longer reports `MaxMaitlandHomeWeekends` / `AwayAtMaitlandGrouping` violations and does not
   create the `maitland_alternate_home_away` penalty bucket.
10. `--slack MaitlandHomeGrouping` / `--slack AwayAtMaitlandGrouping` become no-ops; tests
    asserting these slack keys exist (`test_constraint_registry::test_known_slack_keys`,
    `test_all_slack_keys_exist_in_known_dicts`) updated.
11. **KEPT, verified:** `AwayClubHomeWeekendsCount` and
    `AwayClubPerOpponentAndAggregateHomeBalance` remain in `home_away_balance`, unchanged; a
    test confirms per-club home-weekend totals + 50/50 balance still apply after the deletion.

## Implementation units

> All units touch heavily-shared files; run them as ONE sequenced worktree (not parallel)
> to avoid self-conflict, committing per logical area.

### Unit A — Strip the three rules from engine + registry + config + severity + adjusters + atom
- Files: `constraints/registry.py`, `constraints/stages.py`, `constraints/unified.py`,
  `constraints/atoms/_adjusters.py`, `constraints/severity.py`,
  `constraints/atoms/maitland_alternate_home_away.py` (delete),
  `constraints/atoms/__init__.py`, `config/defaults.py`, `config/season_2026.py`
  (penalty weight).
- Test: import smoke (`import constraints.unified`), `validate_solver_stages == []`, registry
  count test, grep-clean for all removed symbols, and a generation test that completes without
  referencing the removed maps.

### Unit B — Tester removal + KEPT-atoms verification
- Files: `analytics/tester.py`.
- Depends on Unit A.
- Test (GWT, hand oracle): a draw with consecutive Maitland home weekends AND many away clubs
  at Maitland in one weekend is FEASIBLE with NO hard failure and NO
  `maitland_alternate_home_away` penalty; the tester reports no `MaxMaitlandHomeWeekends` /
  `AwayAtMaitlandGrouping` violations; but `AwayClubHomeWeekendsCount` totals and
  `_check_fifty_fifty_home_away` balance are still enforced (per DoD 11).

### Unit C — Test-suite sweep
- Files: any test asserting removed slack keys / severity entries / tester methods / the
  alternation penalty (`tests/test_constraint_registry.py`, `tests/test_severity_*`,
  `tests/test_phase4_adjusters.py` maitland/away adjuster cases, `tests/test_tester_*`,
  any `tests/atoms/test_maitland_alternate_home_away*`).
- Depends on Units A+B.
- Test: full suite green.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — remove the three rows + the Phase-6 alias section;
  update §3 count and the severity table.
- `CLAUDE.md` — remove from severity table + constraint-slack list + Friday/grouping notes;
  drop the `MaitlandAlternateHomeAway` references.
- `docs/DRAW_RULES.md` / `seasons/RULES.md` — remove the back-to-back-home, away-clustering,
  and alternation rules (or mark no-longer-enforced).
- `docs/operator-ai/CONSTRAINT_APPLICATION.md` — drop the two adjuster docs.
- `docs/system/COUNT_ADJUSTERS.md` — remove the two adjuster entries.
- `docs/todo/GOALS.md` — update spec-018 row to "delete all three".

## Out of scope

- `AwayClubHomeWeekendsCount` + `AwayClubPerOpponentAndAggregateHomeBalance` (spec-004) —
  KEPT (per-club home-weekend counts + per-pair/aggregate 50/50). These are the "others still
  required."
- Friday-game counts at Maitland/Gosford (`maitland_friday_games`, etc.) — untouched.
- The `home_field_map` mechanism itself (still used by spec-004 atoms and venue filtering).
