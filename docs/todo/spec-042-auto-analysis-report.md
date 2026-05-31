<!-- status: ready -->
<!-- reviewed: adversarial Sonnet review 2026-05-30 — fixes applied inline -->
<!-- severity: S3 -->
<!-- open_questions: 0 -->
<!-- depends_on: spec-039, spec-040, spec-041 -->

# spec-042 — Auto-generated draw analysis report (HTML + JSON + Plotly), per-team/club/rule breakdowns, end-of-draw hook

**Spec source:** convenor request (this session) — "this needs to be included in a report, run
automatically at the end of a draw creation … per team, per club breakdowns of probably each rule,
this will get dense so you need to be smart about layouts … you can choose to use graphs or other
effective means of display within report (which will eventually also be embedded in a UI)." The
convenor chose **HTML + JSON sidecar with Plotly** as the medium. This is the assembly + presentation
spec that ties spec-039/040/041 together into the artefact a human (and later a UI) actually reads.

## Why

Specs 039–041 produce the *data*: a verified atom↔check alignment (`AlignmentReport`), the effective
slack a draw was solved at (`EffectiveConstraintConfig`), hard-constraint violations at that slack
(`DrawTester` / `ViolationReport`), and per-rule soft-outcome scores + raw metrics + per-team/club
rollups (`SoftOutcomeReport`). None of that is presented anywhere today, and none of it runs
automatically — a convenor would have to invoke the tester and the new layers by hand and read raw
objects.

The existing report surface is thin for this purpose: `analytics/reports.py` has
`generate_html_report()` (hand-rolled inline-CSS HTML, tables only, **no charts**, no per-rule
breakdown, no JSON), `ComplianceCertificate` (xlsx pass/fail), and per-club/grade xlsx reports. No
charting library is installed, and the `report --html` CLI flag is **parsed and partially
implemented** (wired in `run_report()` to call `generate_html_report()` in `reports.py`) — but that
existing path produces the _old_ compliance-only HTML, not the new analysis report. The `--analysis`
flag is new and distinct; see DoD-10. (review fix — M1: `--html` is not a dead parse stub;
`run_report` at run.py:1204 calls `generate_html_report()` for it. The plan's claim it is
"unimplemented" is incorrect.)

This spec builds one self-contained HTML report (with interactive Plotly charts) plus a
machine-readable JSON sidecar for the eventual UI, lays out the dense per-team/per-club/per-rule
breakdowns intelligently, and wires it to run automatically at the end of every draw creation (the
natural hook in `DrawVersionManager.save_solver_output()`, right after the existing violation check)
as well as on demand via the CLI.

## Definition of Done

### Report engine

1. **New module `analytics/analysis_report.py` with `AnalysisReport(draw, data)`** exposing
   `generate(output_dir) -> {html_path, json_path}`, producing:
   - `analysis.html` — a single self-contained file (inline CSS + inline/CDN-pinned Plotly) readable
     with no server.
   - `analysis.json` — the machine-readable sidecar: alignment status, slack provenance, hard
     violations, soft outcomes (scores + raw + per-team/club), and all chart data series, under a
     versioned top-level `{"schema_version": "1.0", ...}` so the future UI can consume it stably.
2. **`plotly>=5.x` added to `requirements.txt`** (the only new dependency). No matplotlib/jinja2 added
   — HTML is composed as today (f-string/section helpers); charts are Plotly `to_html(full_html=False,
   include_plotlyjs=...)` fragments embedded in the page. (review fix — L1: confirmed pandas, openpyxl,
   xlsxwriter are present in requirements.txt; plotly and matplotlib/jinja2 are absent — plan's claim
   is accurate.)

### Report content (composed from 039/040/041, no re-computation)

3. **Header / integrity panel.** Draw version + description; the **slack provenance** banner from
   spec-040 ("effective slack/limits used: stored from solve" vs "assumed from current config —
   legacy draw"); and the spec-039 `AlignmentReport` rendered as an "analysis-integrity" badge
   (green = registry↔tester aligned; red lists any drift). If alignment is red, the report says
   plainly that downstream numbers may be unreliable — the report never claims faithfulness the gate
   would reject.
4. **Hard-constraint section.** Per atom: PASS / VIOLATED + count + severity, from `DrawTester` run at
   the spec-040 effective slack. Sourced from the existing `ViolationReport` (reuse
   `ComplianceCertificate`'s data shaping where possible).
5. **Soft-outcome section.** Per soft atom: the 0–100 satisfaction score (headline) + the raw metric
   (detail) from spec-041's `SoftOutcomeReport`, including the Maitland-away-on-NRL-weekends flagship.
   A Plotly horizontal bar of score-by-rule (sorted worst-first) as the at-a-glance view; atoms with
   `score is None` (outcome-free) shown distinctly, not as 0 or 100.
6. **Per-club breakdown.** One collapsible section per club; within it, each rule's status/score *for
   that club* (from the `by_club` rollups). Dense-but-navigable: a club × rule heatmap (Plotly) plus
   the collapsible detail, so the convenor scans the heatmap then drills in.
7. **Per-team breakdown.** A team × rule matrix rendered as a sortable table AND a Plotly heatmap
   (score/­status colour-coded), from spec-041's `by_team` rollup. Smart layout: teams grouped by
   club then grade; rules grouped by severity; sticky headers; the matrix is the primary dense view
   so ~47 teams × ~20 rules stays legible.
8. **Charts present at minimum:** score-by-rule bar (DoD-5), club×rule heatmap (DoD-6), team×rule
   heatmap (DoD-7), and a home/away balance + slot-usage chart reusing `DrawAnalytics`
   (`home_away_analysis`, `weekly_field_usage`). Both methods are confirmed present in
   `analytics/storage.py` (`home_away_analysis` at line 740, `weekly_field_usage` at line 871) and
   return `pd.DataFrame` objects suitable for Plotly. All chart *data* is mirrored into the JSON
   sidecar (DoD-1) so the UI can re-render natively rather than scrape the HTML. (review fix — L2:
   method name verification against real code complete.)

### Auto-run hook + CLI

9. **Auto-run at end of draw creation.** `DrawVersionManager.save_solver_output()` invokes
   `AnalysisReport(...).generate(...)` right after the existing violation-check block (confirmed at
   `versioning.py` lines 631–651 — the `try/except` block that runs the tester and writes
   `current_violations.txt` is the exact insertion point), writing `analysis.html`/`analysis.json`
   into the versioned dir AND copying to `current_analysis.html` / `current_analysis.json`
   (mirroring the `current_violations.txt` pattern — confirmed: versioning.py lines 641–649 write
   `violations_path` then copy to `current_violations`). Wrapped in try/except that logs a WARNING
   through the project logger on failure and **does not fail the solve/save** (the draw is the
   product; the report is a derivative). Note: `save_solver_output` cleans up temp xlsx files at
   the end (lines 654–657); make sure the analysis files are written to `self.versions_path` BEFORE
   the temp-file cleanup runs. (review fix — M2: anchor verified against real versioning.py.)
10. **CLI: `python run.py report <draw> --analysis [--year YYYY] [-o DIR]`** generates the same report
    on demand for any existing draw. Implement this distinctly. The `--html` flag is **already wired**
    in `run_report()` (run.py:1204) to the old `generate_html_report()`; do NOT remove it — that is a
    separate, working path used today. Add `--analysis` as a new, independent argparse flag to the
    `report` subparser (run.py:188–198) and a separate branch in `run_report()`. The forward-only
    rule applies only to genuinely dead code; `--html` is live. (review fix — H1: critical correction
    — the plan's "remove the dead `--html` flag" instruction would break the existing HTML report
    path. `--html` → `generate_html_report()` at run.py:1204–1207 is fully implemented and live.)
11. **The report runs the analysis at the draw's own effective slack** (spec-040 resolver) for both
    the hard check and the soft outcomes — never at ambient live-config slack — so a freshly-saved
    draw and a re-run `report --analysis` on it produce identical numbers.

### Proof

12. **No-mock e2e test (`tests/test_analysis_report.py`)** on a real-data draw fixture (reuse
    spec-034 fixtures):
    - *Given* a real saved draw, *when* `AnalysisReport(...).generate(tmp)` runs, *then* both
      `analysis.html` and `analysis.json` are written; the HTML is non-empty and contains the Plotly
      div(s); the JSON validates against the documented schema (top-level keys: `schema_version`,
      `alignment`, `slack_provenance`, `hard`, `soft`, `per_club`, `per_team`, `charts`).
    - *Given* the same draw, *when* the report's hard/soft numbers are compared to a direct
      `DrawTester` + `measure_soft_outcomes` run at the resolved slack, *then* they are identical
      (DoD-11 — the report re-computes nothing on its own). Note: `measure_soft_outcomes` is defined
      in the new `analytics/soft_outcomes.py` module from spec-041; the function name is
      `measure_soft_outcomes(draw, data) -> SoftOutcomeReport` per spec-041's DoD-1. (review fix —
      M3: function and module name pinned from upstream spec.)
    - *Given* a draw whose alignment is forced red (inject a real broken `ConstraintInfo` as in
      spec-039's test), *then* the report's integrity badge is red and the caveat text is present.
    - *Given* `save_solver_output()` on a fresh solution, *then* `current_analysis.html` /
      `current_analysis.json` exist afterwards, and a forced exception inside report generation logs a
      WARNING and still leaves the draw saved (hook is non-fatal). Note: the test must call
      `save_solver_output` with a real fixture solution dict (keys are 11-tuples with value 1); the
      existing `_temp_schedule_*.xlsx` cleanup at versioning.py:654–657 must have completed before
      the assertion so file-handle timing issues don't mask the `current_analysis.*` files.
13. **≥85% coverage on `analytics/analysis_report.py`** via the e2e fixtures, honestly reported (the
    Plotly-rendering branches are exercised by asserting on the produced HTML/JSON, not stubbed).

## Implementation units

### Unit A — JSON model + report assembly core (no rendering)

- **Files touched:** `analytics/analysis_report.py` (new — `AnalysisReport`, the JSON-sidecar
  schema/dataclass, assembly of `AlignmentReport` + `EffectiveConstraintConfig` + `ViolationReport` +
  `SoftOutcomeReport` into one structured object), `tests/test_analysis_report.py` (new — JSON schema
  + DoD-11 number-equality test).
- **Change summary:** pure assembly + serialisation; pulls from 039/040/041, computes nothing new.
- **Depends on:** spec-039, spec-040, spec-041 (all `done`). Within plan: none.
- **Suggested executor:** Opus (assembling four upstream structures + designing a stable
  UI-facing JSON schema warrants care).
- **No-mock test outline:** DoD-12 JSON-schema + number-equality bullets on a real fixture.

### Unit B — HTML + Plotly rendering + dense per-team/club/rule layouts

- **Files touched:** `analytics/analysis_report.py` (rendering layer — sections, collapsibles, Plotly
  charts, heatmaps; reuse `generate_html_report`'s CSS idiom and `DrawAnalytics` series),
  `requirements.txt` (add `plotly>=5.x`), `tests/test_analysis_report.py` (extend — HTML produced,
  Plotly div present, alignment-red caveat).
- **Change summary:** the human-facing HTML with the four+ charts and the per-team/club/rule layouts.
- **Depends on:** Unit A merged (renders the assembled model).
- **Suggested executor:** Opus (layout density for ~47×20 matrices + chart correctness; on the
  Sonnet/Opus line → Opus).
- **No-mock test outline:** DoD-12 HTML/Plotly + injected-red-alignment bullets.

### Unit C — Auto-run hook + CLI + docs

- **Files touched:** `analytics/versioning.py` (`save_solver_output` — non-fatal auto-run hook after
  the violation-check block; `current_analysis.*` copies), `run.py` (`report --analysis` wiring;
  add `--analysis` flag to `report` subparser — do NOT touch or remove the live `--html` flag),
  `tests/test_analysis_report.py` (extend — hook writes `current_analysis.*`; non-fatal-on-exception).
- **Change summary:** makes the report automatic + on-demand; adds `--analysis` as a new CLI flag
  beside the existing `--html`. (review fix — H1 carried to unit: do not remove `--html`.)
- **Depends on:** Unit B merged.
- **Suggested executor:** Opus (the hook must be genuinely non-fatal and must run at the draw's
  effective slack; CLI + versioning are integration-sensitive).
- **No-mock test outline:** DoD-12 hook bullets driving the real `save_solver_output` path.

## Doc registry

- `docs/system/HARNESS.md` — document the analysis report in the pipeline (auto-run after violation
  check), the `current_analysis.html`/`.json` outputs, and the JSON sidecar schema (for the UI).
- `CLAUDE.md` (final-form copy) — add the `report --analysis` quick command + note the auto-generated
  analysis report in the draw-output description and the Draw Review Checklist.
- `docs/system/TESTING.md` — register `tests/test_analysis_report.py`.
- `requirements.txt` — `plotly>=5.x` (the doc-registry note records the new dependency rationale).
- `docs/todo/00-dependency-tree.md` — add spec-042; mark the plan-set drained when it lands.
- `docs/todo/GOALS.md` — add the spec-042 row. Note: spec-042 is already listed in the GOALS.md
  spec table (row exists at `spec-042`); update status to `done` rather than adding a duplicate row.
  (review fix — L3: GOALS.md already has a spec-042 row at line 175.)

## Out of scope

- **Building the UI itself.** This spec produces the JSON the UI will consume; the UI is a separate
  project/spec. The `schema_version` field is the forward contract.
- **xlsx version of the analysis report.** The convenor chose HTML+JSON; the existing xlsx reports
  (`ComplianceCertificate`, `ClubReport`, `GradeReport`) remain as-is and are not extended here.
- **Computing alignment / slack provenance / soft scores** — those are spec-039/040/041; this spec
  only *presents* them. Any number the report shows is produced upstream.
- **New constraint semantics or weight changes** — none.
- **Backfilling reports for historical draws in bulk** — the `report --analysis` CLI can be run on any
  draw on demand; no batch backfill job is built.

## Dependencies

- `depends_on: spec-039, spec-040, spec-041`. It renders all three; none may be in-flight when this
  starts (all must be `done` and merged on `final-form`). Note: as of 2026-05-30 all three are
  `review_pending` (spec-039, spec-041) or `under_review` (spec-040), NOT `done` — the execution
  protocol step 1 gate will correctly block until they land. (review fix — M4: status snapshot
  recorded to prevent a premature start claim.)
  **Note — spec id precision:** the soft-outcome dependency is `spec-041-soft-constraint-outcome-measurement.md`
  (the `SoftOutcomeReport` / `measure_soft_outcomes` spec). A separate, unrelated plan
  `spec-043-pre-draw-capacity-precheck.md` was authored concurrently by another session and is NOT a
  dependency. (orchestrator correction — the Mode-A reviewer's "two spec-041 files" note was a
  mis-transcription: the unrelated plan is spec-043, not a second spec-041; there is no spec-041 filename collision.)
- Within this plan: A → B → C (shared `analysis_report.py`, serial).

## Risks & blast radius

- **Auto-run hook slowing or breaking the save.** A heavy or throwing report must never cost a
  solved draw. Mitigation: DoD-9's try/except + WARNING log + non-fatal test (DoD-12) make the hook
  strictly best-effort; report generation is post-hoc and fast (no solving).
- **Plotly payload size.** Embedding `plotly.js` inline per report can bloat the HTML (~3 MB).
  Mitigation: use `include_plotlyjs='cdn'` (pinned version) for the HTML, keep the full data in the
  JSON sidecar; document the CDN dependence (offline viewing falls back to the JSON + tables).
  `include_plotlyjs='cdn'` is a real, valid Plotly `to_html()` option — it inserts a `<script src>`
  tag pointing to Plotly's CDN rather than inlining the JS; the alternative `'require'` is for
  Jupyter. Using a CDN URL means reports require internet to render charts (tables are always
  present). This is the right trade-off; document the offline-fallback behaviour explicitly in the
  HTML's `<noscript>` section. (review fix — L4: CDN option verified as sound.)
- **Dense-matrix legibility (~47 teams × ~20 rules).** The team×rule view can become unreadable.
  Mitigation: heatmap-primary with grouped/sticky headers and a sortable table fallback; Mode-B review
  eyeballs the rendered fixture output, not just that it didn't throw. Concrete legibility criteria
  for the DoD-7 gate: (a) the heatmap uses `height` set proportional to row count so rows don't
  collapse below 14px; (b) rule names are truncated at 20 chars with full name in hover text; (c) the
  sortable table is the fallback when JS is unavailable. The Mode-B reviewer MUST open the rendered
  HTML in a real browser and confirm all ~47 team rows are visible without horizontal clipping.
  (review fix — M5: "smart layout" and "Mode-B eyeballs" were vague; concrete criteria added.)
- **Shared-file contention.** `analysis_report.py` is new (no contention); `versioning.py` and
  `run.py` were touched by spec-040 — but spec-040 is `done` before this starts, so Unit C branches
  off the settled source. No concurrent-spec collision given the dependency ordering. Note: spec-038
  (`ClubVsClubStackedWeekends`) is currently `building` and does NOT touch `versioning.py` or
  `run.py`, so no contention from that parallel workstream either. (review fix — L5: checked live
  spec-038 scope.)
- **Schema churn for the UI.** A later layout change could break a UI built on v1.0 JSON. Mitigation:
  `schema_version` is explicit; additive changes bump minor, breaking changes bump major and are
  called out — but no back-compat shim is built now (forward-only; the UI doesn't exist yet).

## Open Questions

0 — medium (HTML+JSON+Plotly), auto-run requirement, per-team/club/rule breakdowns, and "smart dense
layout" are all settled by the convenor; the upstream data contracts come from 039/040/041.

## Execution protocol (self-contained — for whatever agent picks this up)
<!-- Requires an explicit user go-ahead to START (a `ready` plan does not self-start). Once authorised, run the units end-to-end, pausing only on `blocked` or an unrecoverable failure. -->

0. **Do NOT start without an explicit user instruction to implement this plan.** `ready` ≠ "build now".
1. Status must be `ready` (carries a `reviewed:` stamp). Verify `depends_on` (spec-039, spec-040,
   spec-041) are ALL `done` and merged on `final-form` — if any is not, STOP (unsatisfied dependency).
   The soft-outcome dependency is specifically `spec-041-soft-constraint-outcome-measurement.md`
   (the `SoftOutcomeReport` / `measure_soft_outcomes` spec). The concurrently-authored
   `spec-043-pre-draw-capacity-precheck.md` is unrelated and NOT a dependency.
2. Only after the user says to implement: stamp `building`, claim `owner`. Orchestrator = Opus.
3. **Unit A** on `spec042-unitA` worktree off the post-041 `final-form`: delegate to Opus. Gates:
   type-check; AST sweep; `pytest tests/test_analysis_report.py -v` (Unit-A subset); ≥85% on new code;
   `/adversarial` Mode B. Merge → push → tear down.
4. **Unit B** after A on `spec042-unitB`: delegate to Opus. Gates: type-check; the HTML/Plotly tests
   green; `pip install -r requirements.txt` resolves plotly; AST sweep; `/adversarial` Mode B. Merge →
   push → tear down.
5. **Unit C** after B on `spec042-unitC`: delegate to Opus. Gates: type-check; full
   `pytest tests/test_analysis_report.py -v` (incl. the non-fatal hook test); run a real
   `save_solver_output` on a fixture solution and confirm `current_analysis.*` appear; `/adversarial`
   Mode B. Merge → push → tear down.
6. When all three pass: stamp the plan `done`, archive to `docs/todo/done/`, update
   `docs/todo/00-dependency-tree.md` — the analysis-engine plan-set (039–042) is then fully drained.
