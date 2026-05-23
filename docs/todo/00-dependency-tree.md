<!-- status: ready -->

# Spec dependency tree

Aggregated `depends_on` view of every live (`ready`/`not_ready`/`in_progress`/`delayed`) spec in
`docs/todo/`. Source of truth for *ordering* is each spec's own `depends_on` header; this file is
the cross-spec view so independent work runs in parallel rather than accidentally serialised.
Keep it current: update an edge whenever a spec is created, completed, re-scoped, or unblocked.
**No cycles** — if you find one, two specs are really one unit; collapse or re-cut them.

(Completed work in `docs/todo/done/` is omitted — those edges are satisfied. Done as of this
writing: spec-001…022 and **spec-024** (field-spread, replaced the club-balance pair).)

## DAG

```
spec-021 (done) ──▶ spec-023 (building, constraint-groups) ─┐
                                                            ├─▶ spec-027 (ready, regen-soft)
spec-022 (done, independent)                                │
spec-025 (done, independent) ─▶ spec-026 (ready, startable) ┘
spec-028 (done, independent)
```

Edges:

- **spec-022** — unify helper-var pathway. Independent (`depends_on: none`). **Done** (2026-05-23).
- **spec-023** — constraint-groups machinery (the redesign that *replaced* the superseded
  `spec-023-atom-hard-soft-phases`). Depends on spec-021 + spec-024 (**both `done`** — spec-021
  moved `ClubGameSpread` to the `club_day` hard stage and spec-024 re-scoped it per-field; the
  surviving `_club_game_spread_*` engine methods are intentional, so deleting `soft_only` is
  behaviour-neutral). **`building`** (claimed 2026-05-23 by a concurrent session).
- **spec-025** — `LOCKED_PAIRINGS` config. Independent (`depends_on: none`); builds only on
  already-shipped locked-weeks + FORCED machinery. **Done** (2026-05-23, merged `7afc656`).
- **spec-026** — unified regeneration mode. Depends on spec-025 (writes pins into
  `LOCKED_PAIRINGS`). `ready`; **now startable** — spec-025 has landed. spec-023 is NOT a hard
  dep (DoD 3 guards the `resolve_groups(['regen'])` call and falls back to full hard constraints).
- **spec-027** — regeneration soft-constraint group. Depends on spec-023 (groups machinery) +
  spec-026 (regen mode selects the `regen` group); pins via spec-025 indirectly. Status `ready`
  (hardened) but **NOT startable yet**: spec-025 is `done` but spec-023 is still `building` and
  spec-026 is `ready` (not yet landed). The constraint-groups machinery (`resolve_groups`,
  `DERIVED_GROUPS`, `--groups`, `ConstraintInfo.groups`) and the `soft_only` deletion are NOT on
  `final-form` yet — spec-027 must wait for spec-023 and spec-026 to land.
- **spec-028** — per-weekend notes export column. Independent (`depends_on: none`). **Done**
  (2026-05-23, merged `c077c28`).

## Ready to start in parallel right now

- **spec-026** — unified regeneration mode (its only hard dep, spec-025, has landed). Unclaimed.

(spec-027 unblocks once spec-023 and spec-026 both land; spec-023 is currently `building`.)
