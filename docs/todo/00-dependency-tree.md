<!-- status: ready -->

# Spec dependency tree

Aggregated `depends_on` view of every live (`ready`/`not_ready`/`in_progress`/`delayed`) spec in
`docs/todo/`. Source of truth for *ordering* is each spec's own `depends_on` header; this file is
the cross-spec view so independent work runs in parallel rather than accidentally serialised.
Keep it current: update an edge whenever a spec is created, completed, re-scoped, or unblocked.
**No cycles** — if you find one, two specs are really one unit; collapse or re-cut them.

(Completed work in `docs/todo/done/` is omitted — those edges are satisfied. Done as of this
writing: spec-001…021 and **spec-024** (field-spread, replaced the club-balance pair).)

## DAG

```
spec-021 (done) ──▶ spec-023 (ready, constraint-groups) ─┐
                                                          ├─▶ spec-027 (ready, regen-soft)
spec-022 (ready, independent)                             │
spec-025 (ready, independent) ─▶ spec-026 (ready) ────────┘
spec-028 (ready, independent)
```

Edges:

- **spec-022** — unify helper-var pathway. Independent (`depends_on: none`). Ready now.
- **spec-023** — constraint-groups machinery (the redesign that *replaced* the superseded
  `spec-023-atom-hard-soft-phases`; now landed on `final-form`). Depends on spec-021 (`done`).
  Ready now.
- **spec-025** — `LOCKED_PAIRINGS` config. Independent (`depends_on: none`); builds only on
  already-shipped locked-weeks + FORCED machinery. Ready now.
- **spec-026** — unified regeneration mode. Depends on spec-025 (writes pins into
  `LOCKED_PAIRINGS`). Ready to start once spec-025 lands.
- **spec-027** — regeneration soft-constraint group. Depends on spec-023 (groups machinery) +
  spec-026 (regen mode selects the `regen` group); pins via spec-025 indirectly. Ready to start
  once spec-023 + spec-026 land. (The earlier hard block — constraint-groups not on final-form —
  is now cleared.)
- **spec-028** — per-weekend notes export column. Independent (`depends_on: none`). Ready now.

## Ready to start in parallel right now

- **spec-022** — helper-var pathway cleanup.
- **spec-023** — constraint-groups machinery (deps `done`).
- **spec-025** — `LOCKED_PAIRINGS` config (no unmet deps).
- **spec-028** — per-weekend notes export column (no unmet deps).

(spec-026 unblocks the moment spec-025 lands; spec-027 unblocks once both spec-023 and spec-026
land.)
