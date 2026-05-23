"""spec-023 Unit D — DoD 9: draw metadata records the resolved group selection.

`DrawVersionManager._build_draw_metadata` (the builder behind `save_solver_output`)
must record:
  * `groups_selected`        — the list of group NAMES the operator chose,
                               from `data['_groups_selected']` (stashed by the
                               simple/staged dispatch in main_staged.py).
  * `applied_constraint_set` — the deduped union of canonical constraint names
                               that selection resolved to, from
                               `data['_constraint_names']`.

GWT, no mocks: a real `DrawVersionManager`, a real minimal solution dict, and the
real metadata builder.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analytics.versioning import DrawVersionManager
from constraints.registry import resolve_groups


def _minimal_solution():
    # One scheduled 11-tuple game key so the stats block has something to chew on.
    key = (
        'Maitland PHL', 'Norths PHL', 'PHL', 'Sunday', 3, '11:30', 1,
        '2026-03-22', 1, 'EF', 'Newcastle International Hockey Centre',
    )
    return {key: 1}


def test_metadata_records_groups_selected_and_applied_set(tmp_path):
    """Given data carrying the Unit-C-stashed _groups_selected/_constraint_names,
    When _build_draw_metadata builds the metadata dict,
    Then groups_selected == the group names and applied_constraint_set == the
    resolved canonical names."""
    names = resolve_groups(['core', 'soft'])
    data = {
        'year': 2026,
        '_groups_selected': ['core', 'soft'],
        '_constraint_names': list(names),
    }
    vm = DrawVersionManager(str(tmp_path), year=2026)
    meta = vm._build_draw_metadata(
        draw=None, solution=_minimal_solution(), data=data,
        mode='staged', timestamp='2026-05-23T00:00:00',
    )
    assert meta['groups_selected'] == ['core', 'soft']
    assert meta['applied_constraint_set'] == list(names)
    # Mirrors constraints_applied — present alongside, not replacing it.
    assert 'constraints_applied' in meta


def test_metadata_defaults_to_default_group_when_unset(tmp_path):
    """Given data with NO _groups_selected (legacy/no --groups run),
    When metadata is built,
    Then groups_selected defaults to ['default'] and applied_constraint_set is
    an empty list (no resolved set was stashed)."""
    data = {'year': 2026}
    vm = DrawVersionManager(str(tmp_path), year=2026)
    meta = vm._build_draw_metadata(
        draw=None, solution=_minimal_solution(), data=data,
        mode='staged', timestamp='2026-05-23T00:00:00',
    )
    assert meta['groups_selected'] == ['default']
    assert meta['applied_constraint_set'] == []
