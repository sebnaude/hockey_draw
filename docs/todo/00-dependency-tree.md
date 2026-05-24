<!-- status: index -->

# Spec dependency tree

Aggregated `depends_on` view of every live (`ready`/`not_ready`/`in_progress`/`delayed`) spec in
`docs/todo/`. Source of truth for *ordering* is each spec's own `depends_on` header; this file is
the cross-spec view so independent work runs in parallel rather than accidentally serialised.
Keep it current: update an edge whenever a spec is created, completed, re-scoped, or unblocked.
**No cycles** — if you find one, two specs are really one unit; collapse or re-cut them.

## Live specs

- **spec-029** — club-day weekends in the published-draw Notes column. `depends_on: none`
  (extends spec-028, which is `done`). Single unit. Status: `ready` (adversarial
  Mode A review complete 2026-05-24; critical blast-radius fixes applied to plan).

```
spec-029  ──depends_on──▶  (none; spec-028 already done)
```

When spec-029 lands, mark it done, move it to `docs/todo/done/`, and return this folder to
"no live specs".

Most recently completed:

- **spec-027** — regeneration soft-constraint group. **Done** (2026-05-24, merged `c851f25`).
  Delivered the `regen` constraint group (`core_hard` ∪ `regen_soft` ∪ `soft`), 13 new
  `*RegenSoft` soft-analogue atoms, the `core_hard` tags + `regen` derived group in the registry,
  the `--regen-from` → staged-dispatch wiring (the engine-only `--simple` path can't dispatch the
  non-engine RegenSoft atoms), and the DoD-7 infeasible→feasible witness. Depended on spec-023 +
  spec-025 + spec-026 (all `done` before it started). See
  `docs/system/REGEN_CONSTRAINTS.md` for the full reference.

Earlier (all `done`, in `docs/todo/done/`): spec-001…022, **spec-023** (constraint-groups
machinery, merged `083bf5a`), **spec-024** (field-spread), **spec-025** (`LOCKED_PAIRINGS`,
`7afc656`), **spec-026** (unified regeneration mode, `6f39b83`), **spec-028** (per-weekend notes
export column, `c077c28`).

## Ready to start in parallel right now

- **spec-029** (once `/adversarial` Mode A stamps it `ready`).
