"""Phase 7c: lockdown test — production code must not import the
pre-atomization combined-constraint modules.

Forbidden in production:
  - `constraints.archived.*` (the new home for the legacy modules)
  - `constraints.original`              (old top-level path)
  - `constraints.ai`                    (old top-level path)
  - `constraints.archived_equalspacing_original` (old top-level path)

The legacy classes live under `constraints/archived/`. New constraint
work goes through the registry / atoms / `constraints.stages` dispatcher.

Tests and the `constraints/archived/` package itself are exempt —
parity tests in `tests/atoms/*_parity.py` and
`tests/test_constraints_equivalence.py` legitimately import the legacy
classes for parity comparisons.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
ALLOWED = (REPO / 'constraints' / 'archived', REPO / 'tests')
SKIP_DIR_NAMES = {'.venv', 'venv', '__pycache__', 'site-packages', '.git'}

# Match every forbidden import shape — both the old top-level paths
# (constraints.original / constraints.ai / constraints.archived_equalspacing_original)
# and the new constraints.archived.* path.
PATTERN = re.compile(
    r'^\s*from\s+constraints\.(archived(\.\w+)?|original|ai|archived_equalspacing_original)\b\s+import\b|'
    r'^\s*import\s+constraints\.(archived(\.\w+)?|original|ai|archived_equalspacing_original)\b',
    re.MULTILINE,
)


def _iter_py_files():
    for path in REPO.rglob('*.py'):
        # Skip allowlist trees.
        try:
            for allowed in ALLOWED:
                path.relative_to(allowed)
                # If we get here, path is under an allowed root.
                break
            else:
                pass
        except ValueError:
            pass

        path_str = str(path)
        if any(seg in path_str for seg in SKIP_DIR_NAMES):
            continue
        # Skip files inside ALLOWED roots.
        skip = False
        for allowed in ALLOWED:
            try:
                path.relative_to(allowed)
                skip = True
                break
            except ValueError:
                continue
        if skip:
            continue
        yield path


def test_no_prod_module_imports_archived():
    offenders = []
    for path in _iter_py_files():
        try:
            text = path.read_text(encoding='utf-8', errors='ignore')
        except OSError:
            continue
        if PATTERN.search(text):
            offenders.append(str(path.relative_to(REPO)))
    assert not offenders, (
        'These prod modules import constraints.archived (use the registry '
        f'instead): {offenders}'
    )


def test_archived_package_marker_present():
    """The archived package itself must exist with __init__ + README."""
    archived = REPO / 'constraints' / 'archived'
    assert archived.is_dir(), f'{archived} missing'
    assert (archived / '__init__.py').is_file()
    assert (archived / 'README.md').is_file()
