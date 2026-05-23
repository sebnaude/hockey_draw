<!-- status: index -->

# Spec dependency tree

Aggregated `depends_on` view of every live (`ready`/`not_ready`/`in_progress`/`delayed`) spec in
`docs/todo/`. Source of truth for *ordering* is each spec's own `depends_on` header; this file is
the cross-spec view so independent work runs in parallel rather than accidentally serialised.
Keep it current: update an edge whenever a spec is created, completed, re-scoped, or unblocked.
**No cycles** — if you find one, two specs are really one unit; collapse or re-cut them.

(Completed work in `docs/todo/done/` is omitted — those edges are satisfied. Done as of this
writing: spec-001…022, **spec-024** (field-spread, replaced the club-balance pair), **spec-025**,
**spec-026**, **spec-028**, and **spec-023** (constraint-groups machinery, 2026-05-24 @083bf5a).)

## DAG

```
spec-021 (done) ──▶ spec-023 (done, constraint-groups) ─┐
                                                        ├─▶ spec-027 (ready, regen-soft) ◀── UNBLOCKED
spec-022 (done, independent)                            │
spec-025 (done, independent) ─▶ spec-026 (done) ────────┘
spec-028 (done, independent)
```

Edges:

- **spec-022** — unify helper-var pathway. Independent (`depends_on: none`). **Done** (2026-05-23).
- **spec-023** — constraint-groups machinery (the redesign that *replaced* the superseded
  `spec-023-atom-hard-soft-phases`). Depended on spec-021 + spec-024 (both `done`). **Done**
  (2026-05-24, merged `083bf5a`). Delivered `ConstraintInfo.groups`, `resolve_group(s)`,
  `DERIVED_GROUPS`, `validate_group_order`, `apply_constraint_set`, `--groups`/`--list-groups`,
  the `soft_only` deletion, and metadata/severity registry read-through. `validate_solver_stages
  (DEFAULT_STAGES)==[]`; DoD-8 staged parity hard=12908 (zero delta).
- **spec-025** — `LOCKED_PAIRINGS` config. Independent; built on already-shipped locked-weeks +
  FORCED machinery. **Done** (2026-05-23, merged `7afc656`).
- **spec-026** — unified regeneration mode. Depends on spec-025 (writes pins into
  `LOCKED_PAIRINGS`). **Done** (2026-05-24, merged `6f39b83`). Group selection (`--groups regen`)
  was guarded with a `TODO(spec-023)` seam in `run.py::run_generate` that falls back to the full
  hard constraint set + warning. spec-023's groups machinery has now landed, so that seam is
  ready to be wired — that wiring is **spec-027's** deliverable (it defines + selects the `regen`
  group), not spec-026's.
- **spec-027** — regeneration soft-constraint group. Depends on spec-023 (groups machinery) +
  spec-026 (regen mode selects the `regen` group); pins via spec-025 indirectly. **All three deps
  are now `done`/landed on final-form** — spec-027 is **startable now** (status `ready`,
  hardened, unclaimed). It defines a `regen` group and wires the `--groups regen` seam left in
  `run.py::run_generate`.
- **spec-028** — per-weekend notes export column. Independent (`depends_on: none`). **Done**
  (2026-05-23, merged `c077c28`).

## Ready to start in parallel right now

- **spec-027** — regeneration soft-constraint group. Its three deps (spec-023, spec-025,
  spec-026) are all `done` and landed on `final-form`. **CLAIMED 2026-05-24** (session
  opus-20260524T093159Z); status `building`.
