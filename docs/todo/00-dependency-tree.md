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
spec-025 (done, independent) ─▶ spec-026 (done) ────────────┘
spec-028 (done, independent)
```

Edges:

- **spec-022** — unify helper-var pathway. Independent (`depends_on: none`). **Done** (2026-05-23).
- **spec-023** — constraint-groups machinery (the redesign that *replaced* the superseded
  `spec-023-atom-hard-soft-phases`). Depends on spec-021 + spec-024 (**both `done`** — spec-021
  moved `ClubGameSpread` to the `club_day` hard stage and spec-024 re-scoped it per-field; the
  surviving `_club_game_spread_*` engine methods are intentional, so deleting `soft_only` is
  behaviour-neutral). **`building`** (claimed 2026-05-23 by a concurrent session).
  already-shipped locked-weeks + FORCED machinery. **Done** (2026-05-23, merged `7afc656`).
- **spec-026** — unified regeneration mode. Depends on spec-025 (writes pins into
  `LOCKED_PAIRINGS`). **Done** (2026-05-24, merged `6f39b83`). Group selection (`--groups regen`)
  is guarded: until spec-023 lands, regen falls back to the full hard constraint set with a
  warning (the spec-023-landed seam is marked `TODO(spec-023)` in `run.py::run_generate`).
- **spec-027** — regeneration soft-constraint group. Depends on spec-023 (groups machinery) +
  spec-026 (regen mode selects the `regen` group); pins via spec-025 indirectly. Status `ready`
  (hardened) but **NOT startable yet**: spec-025 and spec-026 are now `done`, but spec-023 is
  still `building` (not landed). The constraint-groups machinery (`resolve_groups`,
  `DERIVED_GROUPS`, `--groups`, `ConstraintInfo.groups`) and the `soft_only` deletion are NOT on
  `final-form` yet — spec-027 must wait for spec-023 to land.
- **spec-028** — per-weekend notes export column. Independent (`depends_on: none`). **Done**
  (2026-05-23, merged `c077c28`).

## Ready to start in parallel right now

- (none unclaimed) — spec-023 is `building`; spec-027 unblocks the moment spec-023 lands
  (spec-025 + spec-026 deps already satisfied).
