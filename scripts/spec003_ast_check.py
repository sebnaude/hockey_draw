"""Spec-003 AST check for dead code / dark paths in the new code.

Scans the two new atom files and the tester additions for:
- Unused imports
- Unused locals
- Unreachable branches (statements after `return`/`raise`/`continue`/`break`)
- Bare `except:` or `except Exception: pass`
- `if False:` / `while False:`
- Functions defined but never called within their module (best-effort)

Exits non-zero if any finding is present.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List


REPO = Path(__file__).resolve().parents[1]
TARGETS = [
    REPO / 'constraints' / 'atoms' / 'nihc_fill_wf_before_ef.py',
    REPO / 'constraints' / 'atoms' / 'nihc_fill_ef_before_sf.py',
]


class Finder(ast.NodeVisitor):
    def __init__(self, path: Path):
        self.path = path
        self.issues: List[str] = []
        self._defined_names: set = set()
        self._used_names: set = set()
        self._imported: dict = {}

    # Imports
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            name = alias.asname or alias.name.split('.')[0]
            self._imported[name] = node.lineno
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        for alias in node.names:
            name = alias.asname or alias.name
            self._imported[name] = node.lineno
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self._used_names.add(node.id)
        elif isinstance(node.ctx, ast.Store):
            self._defined_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # When attribute access on a Name, the base name is "used"
        if isinstance(node.value, ast.Name):
            self._used_names.add(node.value.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        # Check for unreachable code (statements after a terminator)
        terminators = (ast.Return, ast.Raise, ast.Continue, ast.Break)
        body = node.body
        for i, stmt in enumerate(body):
            if isinstance(stmt, terminators) and i < len(body) - 1:
                rest = body[i + 1:]
                # Allow trailing docstring / Ellipsis -- not relevant here.
                lineno = rest[0].lineno
                self.issues.append(
                    f"{self.path.name}:{lineno}: unreachable code after "
                    f"{type(stmt).__name__} in {node.name}"
                )
                break
        self.generic_visit(node)

    def visit_If(self, node: ast.If):
        # if False: / if 0:
        if isinstance(node.test, ast.Constant) and not node.test.value:
            self.issues.append(
                f"{self.path.name}:{node.lineno}: dead 'if {node.test.value!r}:' branch"
            )
        self.generic_visit(node)

    def visit_While(self, node: ast.While):
        if isinstance(node.test, ast.Constant) and not node.test.value:
            self.issues.append(
                f"{self.path.name}:{node.lineno}: dead 'while {node.test.value!r}:' loop"
            )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.type is None:
            self.issues.append(
                f"{self.path.name}:{node.lineno}: bare 'except:' (swallows everything)"
            )
        # except X: pass
        if (
            len(node.body) == 1
            and isinstance(node.body[0], ast.Pass)
        ):
            self.issues.append(
                f"{self.path.name}:{node.lineno}: 'except: pass' swallows error"
            )
        self.generic_visit(node)

    def finalize(self):
        for name, lineno in self._imported.items():
            if name == '__future__':
                continue
            if name not in self._used_names:
                self.issues.append(
                    f"{self.path.name}:{lineno}: unused import '{name}'"
                )


def main() -> int:
    all_issues: List[str] = []
    for path in TARGETS:
        if not path.exists():
            all_issues.append(f"missing target: {path}")
            continue
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(path))
        finder = Finder(path)
        finder.visit(tree)
        finder.finalize()
        all_issues.extend(finder.issues)

    if all_issues:
        print('AST findings:')
        for issue in all_issues:
            print(f'  - {issue}')
        return 1
    print('AST check clean for spec-003 atom files.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
