<!-- status: ready -->
<!-- owner: session=none claimed=none -->
<!-- depends_on: none (shares config/defaults.py DEFAULT_STAGES, constraints/registry.py, constraints/unified.py with spec-015..018 — rebase + re-run validate_solver_stages before merge) -->

# spec-014 — PHL/2nd adjacency: force same-club adjacency, real-time cross-venue gap, minimal vars

## Why

The convenor's actual goal for PHL/2nd adjacency is **positive**: within a club, that
club's PHL team and 2nd-grade team should play **back-to-back** (adjacent timeslots,
same venue) so spectators/officials see both with no gap. When the two are at
**different venues** there must be a genuine **≥ 3-hour** gap between their **start
times**. Rationale (start-to-start, since that's all we control): game 1 must finish
(~game length) + warm-down (~1.5 h after start is well past the whistle) + travel between
grounds + warm-up before game 2 (~1.5 h). The convenor's figure for a safe start-time
separation across venues is **3 hours**.

The current implementation (`constraints/unified.py::_phl_adjacency_hard`, lines 531-557)
does NOT force adjacency. It only *forbids* the two bad combinations per
`(club, week, day)`:
- within ±`phl_adjacency_window_minutes` (default 180) **and different location** → forbid both,
- outside the window **and same location** → forbid both.

This nudges toward adjacency but never requires it: a club whose PHL and 2nd both play
the same day can land them far apart on the same field with no penalty as long as
neither forbidden pattern triggers (e.g. one early, one late on different fields with a
big gap is allowed). It is also dispatched as a legacy in-engine method on the
`PHLAndSecondGradeAdjacency` engine key — not yet a registered atom, against the
atomization goal in `docs/todo/GOALS.md §2`.

Two further issues to fix while here:
- The legacy window is **±180 min** used symmetrically for a forbid rule. Replace with two
  distinct rules: "same venue ⇒ adjacent slots" (a slot-adjacency rule, no minute threshold)
  and "different venue ⇒ start times ≥ 180 min apart" (a real-minutes travel rule). They
  should not share one magic number even though both happen to be 180 — the same-venue rule
  is about slot adjacency, the cross-venue rule about a 3-hour start-time gap.
- Cost: the current pairwise `O(phl_slots × 2nd_slots)` `Add(sum+sum<=1)` per club/day adds
  **no new variables** but many constraints. The rewrite must not blow up variable count —
  prefer reusing per-(club,week,day,grade) "plays" indicators and a single real-time
  channel rather than per-slot-pair products.

## Definition of Done

1. A new atom `PHLAnd2ndAdjacency` exists in `constraints/atoms/phl_2nd_adjacency.py`,
   subclassing `Atom`, registered in `constraints/registry.py` (severity 1, `atom_group=None`,
   `solver_class_names=['PHLAndSecondGradeAdjacency','PHLAndSecondGradeAdjacencyAI']` taken
   over from the legacy entry, plus any declared helper-var kinds), and dispatched from
   `DEFAULT_STAGES` `critical_feasibility` (replacing the bare `PHLAndSecondGradeAdjacency`
   engine-key entry wired in by the pre-spec cleanup).
2. Behaviour, per `(club, week, day)` where the club fields BOTH a PHL and a 2nd-grade game:
   a. **Same venue:** the two games must be in **adjacent day_slots** on the **same field**
      (back-to-back). Encoded without enumerating every slot-pair product.
   b. **Different venue:** the two start times must differ by **≥ `phl_2nd_cross_venue_min_minutes`
      (default 180 = 3 h)**, using a genuine minutes-since-midnight comparison on the chosen
      start times (NOT day_slot indices — different fields run different clocks).
   c. If the club fields only one (or neither) of the two grades that day, the atom adds
      nothing for that club/day.
3. The 180-minute cross-venue start-time gap and the same-venue-adjacency rule read distinct
   config keys in `CONSTRAINT_DEFAULTS` (`phl_2nd_cross_venue_min_minutes`=180; the legacy
   `phl_adjacency_window_minutes` is removed or repurposed with a migration note).
4. Net decision-variable count added by the atom is **O(clubs × playing-days)** helper
   indicators at most — quantified in the plan's design unit and asserted in a test that
   counts `model.Proto().variables` before/after on a fixture.
5. `tests/atoms/test_phl_2nd_adjacency.py` covers (no mocks, GWT, hand-computed oracle):
   same-venue-non-adjacent → INFEASIBLE; same-venue-adjacent → FEASIBLE; different-venue
   150-min-apart → INFEASIBLE (< 180); different-venue 210-min-apart → FEASIBLE (≥ 180);
   club fields only PHL → no constraints added (n == 0).
6. Legacy `_phl_adjacency_hard` removed from the engine dispatch (or kept only as a
   parity-reference method, clearly marked, not called), and the `PHLAndSecondGradeAdjacency`
   engine-key path in `apply_stage_1_hard` updated/removed accordingly.
7. `analytics/tester.py` has a `_check_phl_2nd_adjacency` post-hoc check matching the new
   semantics (same-venue adjacency + real-time cross-venue gap), wired in the registry.
8. Full test suite green; `validate_solver_stages(DEFAULT_STAGES)` returns `[]`;
   `len(CONSTRAINT_REGISTRY)` updated and its count test adjusted.
9. ⚠️ Locked-week caveat preserved: under `--locked` runs this atom can over-constrain
   Gosford PHL (zero margin). Either the atom self-skips locked weeks for the affected
   club, or the docs/CLAUDE.md note to `--exclude PHLAndSecondGradeAdjacency` is updated to
   the new atom name. Decide in the design unit.

## Open decision (recommendation baked in)

- **Cross-venue threshold:** **180 min (3 h) start-time difference** — confirmed by convenor
  (game length + ~1.5 h warm-down + travel + ~1.5 h warm-up, measured start-to-start). This
  is a real-minutes gap, NOT the legacy ±180 forbid window.
- **Force vs forbid for same-venue:** recommend a *reified* "if both play same venue same
  day ⇒ adjacent slots" rather than an unconditional force, so a club legitimately playing
  only one grade that day is unaffected (DoD 2c). Hard "always adjacent" would be infeasible
  on bye days.

## Implementation units

### Unit A — Design spike: minimal-variable encoding (deliverable: short design note appended here)
- Files: this plan (append "Chosen encoding" section); no production code.
- Evaluate two encodings and pick by added-variable count:
  1. **Real-time channel:** per `(club, week, day, grade)` create one IntVar
     `start_min ∈ [0, 1440]` channeled as `sum(time_minutes(slot) * var)` over that grade's
     candidate vars (valid because NoDoubleBooking ⇒ ≤1 plays), plus a `plays` BoolVar.
     Then: `both_play ⇒ |phl_start - 2nd_start|` constrained by venue. Same-venue adjacency
     becomes "same field AND |slot rank| == 1"; cross-venue becomes `|Δmin| ≥ 120`.
  2. **Indicator-pair:** reuse the existing per-slot vars, build per-(club,day) `plays`
     indicators, and add reified implications only between *candidate adjacent* slot pairs
     (pre-filtered by real time), avoiding the full O(n²) product.
- Output: chosen approach + a hand-counted variable budget (e.g. "≤ 2 IntVars + 2 BoolVars
  per club·day"). This unblocks Unit B.

### Unit B — Implement `PHLAnd2ndAdjacency` atom
- Files: `constraints/atoms/phl_2nd_adjacency.py` (new), `constraints/atoms/__init__.py`
  (import + `__all__`), `constraints/registry.py` (entry; remove/repurpose legacy
  `PHLAndSecondGradeAdjacency` if it becomes atom-owned), `constraints/severity.py`
  (ensure mapping resolves), `config/defaults.py` (`CONSTRAINT_DEFAULTS` new key +
  `DEFAULT_STAGES` critical_feasibility), `constraints/unified.py` (retire/neuter
  `_phl_adjacency_hard` + its dispatch + the `PHLAndSecondGradeAdjacency` skip-key block).
- Depends on Unit A.
- Test: `tests/atoms/test_phl_2nd_adjacency.py` per DoD 5 + a variable-count assertion (DoD 4).

### Unit C — Tester parity + cleanup
- Files: `analytics/tester.py` (`_check_phl_2nd_adjacency`), `constraints/registry.py`
  (tester_check_methods), remove the old `_check_phl_second_grade_*` adjacency portion if
  it conflicts.
- Depends on Unit B.
- Test: post-hoc check flags a hand-built draw with a same-venue non-adjacent PHL/2nd pair
  and a 90-min cross-venue pair; passes a compliant draw.

## Doc registry

- `docs/todo/GOALS.md` — add spec-014 row + short spec blurb; flip to done on ship.
- `CLAUDE.md` — update the "Bastardised constraints" note + locked-week `--exclude` guidance
  to the new atom name; update the constraint/severity table.
- `docs/system/CONSTRAINT_INVENTORY.md` — replace the `PHLAndSecondGradeAdjacency` row with
  the new atom's actual behaviour (same-venue adjacency + 120-min cross-venue), update §3 count.
- `docs/DRAW_RULES.md` — document the convenor-facing rule (back-to-back same venue; 2 h
  apart cross-venue).
- `docs/operator-ai/CONSTRAINT_APPLICATION.md` / `docs/ai/GAME_TIME_DICTIONARIES.md` — note the
  real-time (minutes) comparison and the two distinct config keys.

## Out of scope

- Changing PHL/2nd *concurrency at Broadmeadow* (`PHLConcurrencyAtBroadmeadow`,
  `PHLAnd2ndConcurrencyAtBroadmeadow`) — separate atoms, untouched.
- The Friday-count / Gosford-rounds rules — see spec-015.
- Extending adjacency to other grade pairs (3rd/4th etc.) — not requested.
