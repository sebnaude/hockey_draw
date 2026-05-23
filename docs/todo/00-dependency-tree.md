<!-- status: ready -->

# Spec dependency tree

Aggregated `depends_on` view of every live (`ready`/`not_ready`/`in_progress`/`delayed`) spec in
`docs/todo/`. Source of truth for *ordering* is each spec's own `depends_on` header; this file is
the cross-spec view so independent work runs in parallel rather than accidentally serialised.
Keep it current: update an edge whenever a spec is created, completed, re-scoped, or unblocked.
**No cycles** — if you find one, two specs are really one unit; collapse or re-cut them.

(Completed work in `docs/todo/done/` is omitted — those edges are satisfied.)

## DAG

```
spec-016 (done) ─┐
spec-017 (done) ─┴─▶ spec-021 (in_progress) ─┐
                                              ├─▶ spec-023 (delayed, user hold) ─▶ spec-026 (not_ready)
spec-022 (ready, independent) ────────────────┘                                        ▲
                                                                                       │
spec-024 (ready, independent) ─▶ spec-025 (ready) ─────────────────────────────────────┘
```

Edges:

- **spec-021** depends on spec-016 + spec-017 (both `done`). In progress.
- **spec-022** independent (`depends_on: none`). Ready now.
- **spec-023** depends on spec-021. Currently `delayed` (user hold) AND, on `final-form`, still
  the superseded `spec-023-atom-hard-soft-phases` design; the constraint-groups redesign that
  spec-026 needs lives on the `spec-021-contiguity` branch and is not yet merged to final-form.
- **spec-024** independent (`depends_on: none`). Builds only on already-shipped locked-weeks +
  FORCED machinery. **Ready now — parallelisable immediately.**
- **spec-025** depends on spec-024 (writes into `LOCKED_PAIRINGS`). Ready to start once spec-024
  lands.
- **spec-026** depends on spec-023 (groups machinery) + spec-025 (selects the `regen` group).
  `not_ready`/blocked until the constraint-groups redesign lands on final-form.

## Ready to start in parallel right now

- **spec-022** — independent helper-var pathway cleanup.
- **spec-024** — `LOCKED_PAIRINGS` config (no unmet deps).

(spec-021 is already `in_progress`. spec-025 unblocks the moment spec-024 lands; spec-026
unblocks once both spec-023's groups redesign reaches final-form and spec-025 lands.)
