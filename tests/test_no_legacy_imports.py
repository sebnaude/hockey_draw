"""Phase 7c: lockdown test — production code must not import from
`constraints/archived/`.

This is the architectural fence around the pre-atomization implementations.
Anything that needs the original combined classes goes through the
registry / atoms / `constraints.stages` dispatcher, NOT a direct import of
`constraints.archived.*`.

Tests are exempt — `tests/test_constraints_equivalence.py` and the
per-cluster parity tests in `tests/atoms/` legitimately import the legacy
classes for parity comparisons.

NOTE on scope: the spec also forbids prod imports of `constraints.original`,
`constraints.ai`, and `constraints.archived_equalspacing_original`. Those
top-level modules still exist as part of the partial Phase 7c migration —
see `docs/ATOMIZATION_HANDOFF.md` for the follow-up move-and-shim plan.
This test enforces the new lockdown around `constraints/archived/` so any
new code that lands gravitates to the registry-driven path.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
ALLOWED = (REPO / 'constraints' / 'archived', REPO / 'tests')
SKIP_DIR_NAMES = {'.venv', 'venv', '__pycache__', 'site-packages', '.git'}

# Match `from constraints.archived.<anything> import ...` and
# `import constraints.archived.<anything>`. The bare
# `from constraints.archived import` is also caught.
PATTERN = re.compile(
    r'^\s*from\s+constraints\.archived(\.\w+)?\s+import\b|'
    r'^\s*import\s+constraints\.archived(\.\w+)?\b',
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
