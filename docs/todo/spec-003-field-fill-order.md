<!-- status: ready -->
<!-- owner: unassigned -->
<!-- depends_on: none -->

# spec-003 — NIHC field-fill ordering (WF → EF → SF) replaces last-game-WF rule

**Spec source:** [`docs/todo/GOALS.md` → spec-003](GOALS.md#spec-003--field-fill-ordering-wf--ef--sf-replaces-last-game-wf-rule)

## Why

The current perennial rule "last game of the day on West Field if only one game" is a special case of a more general principle: at NIHC, **higher-priority fields fill before lower-priority fields**. WF > EF > SF. Generalising fixes partial-slot behaviour throughout the day (not just the last slot) and removes a manual-review footnote from `PERENNIAL_RULES.md`.

## Definition of Done

1. Two new atoms in `constraints/atoms/`:
   - `NIHCFillWFBeforeEF` — per (date, day_slot) at NIHC: `EF_used → WF_used`.
   - `NIHCFillEFBeforeSF` — per (date, day_slot) at NIHC: `SF_used → EF_used`.
2. The third implication (`SF_used → WF_used`) is transitively implied by the two above and is not coded separately.
3. Both atoms registered in `constraints/registry.py` with severity 1 (CRITICAL) — these are structural, not preferences.
4. Both atoms wired into the `critical_feasibility` stage in `DEFAULT_STAGES`.
5. The existing tester rule `_check_west_field_last_slot` (in `analytics/tester.py`) is **deleted** — replaced by `_check_nihc_field_fill_order` covering both implications and reporting per (date, day_slot) violations.
6. Unit tests cover:
   - Given a (date, day_slot) with only an EF game, no WF game, When checked, Then `NIHCFillWFBeforeEF` flags violation.
   - Given a (date, day_slot) with only an SF game, no EF game, When checked, Then `NIHCFillEFBeforeSF` flags violation.
   - Given all three fields used, no violation.
   - Given only WF used, no violation.
7. `docs/system/CONSTRAINT_INVENTORY.md` gets two new rows.
8. `docs/operator-human/PERENNIAL_RULES.md` — last-game-WF section is **replaced** by a field-fill-order section.
9. `CLAUDE.md` Draw Review Checklist updated — remove the WF-last-slot bullet (the atom enforces it now), add a "field-fill order checked by atoms" reference.

## Implementation units

### Unit 1 — Atoms

- **Files touched:** `constraints/atoms/nihc_fill_wf_before_ef.py`, `constraints/atoms/nihc_fill_ef_before_sf.py`, `constraints/registry.py`, `config/defaults.py::DEFAULT_STAGES`.
- **Helper var:** per (date, day_slot, field) "field_used" indicator. Either declare via `HelperVarRegistry` under kind `nihc_field_used` so both atoms share, or compute locally if simpler.
- **Edge case:** SF doesn't exist as a NIHC field on all dates (check `DAY_TIME_MAP`). Atoms must skip (date, day_slot) combinations where the lower-priority field isn't a valid slot — never assert "EF must be used" when EF isn't a real option.

### Unit 2 — Tester replacement

- **Files touched:** `analytics/tester.py` — delete `_check_west_field_last_slot`, add `_check_nihc_field_fill_order`.
- **Update:** any test asserting the old check name.

### Unit 3 — Docs

- **Files touched:** `docs/system/CONSTRAINT_INVENTORY.md`, `docs/operator-human/PERENNIAL_RULES.md`, `CLAUDE.md`.

## Doc registry

- `docs/system/CONSTRAINT_INVENTORY.md` — two new rows
- `docs/operator-human/PERENNIAL_RULES.md` — replace WF-last-slot section
- `CLAUDE.md` — update Draw Review Checklist
- `docs/todo/GOALS.md` — flip spec-003 status to "done"

## Out of scope

- Field-fill order at non-NIHC venues — there are only one or two fields at Maitland Park / Central Coast, doesn't apply.
- Time-of-day priorities within a field (e.g. "prefer 11:30 over 14:00 on WF") — different concern, separate plan if wanted.
