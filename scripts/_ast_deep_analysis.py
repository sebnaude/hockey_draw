"""
Deep AST Analysis - Full call graph, dead code, logic errors, dead islands.
Traces every entry point through to completion.
"""
import ast
import os
import sys
import collections
from pathlib import Path

ROOT = Path(__file__).parent.parent
EXCLUDE_DIRS = {'__pycache__', '.venv', '.git', '.claude'}
EXCLUDE_PREFIXES = ('temp_',)


def get_py_files():
    files = []
    for p in ROOT.rglob('*.py'):
        rel = p.relative_to(ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts):
            continue
        if p.name.startswith(EXCLUDE_PREFIXES):
            continue
        files.append(p)
    return sorted(files)


def rel(path):
    return str(path.relative_to(ROOT)).replace(os.sep, '/')


def parse_file(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            source = f.read()
        return ast.parse(source, filename=str(path)), source
    except SyntaxError as e:
        return None, str(e)


def get_module_name(path):
    r = path.relative_to(ROOT)
    parts = list(r.parts)
    if parts[-1] == '__init__.py':
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1].replace('.py', '')
    return '.'.join(parts)


class CallGraphBuilder(ast.NodeVisitor):
    def __init__(self, filepath):
        self.filepath = filepath
        self.current_scope = None
        self.definitions = {}
        self.calls = []
        self.name_refs = []
        self.imports = {}
        self.import_modules = []
        self.class_bases = {}
        self.class_methods = collections.defaultdict(list)
        self._scope_stack = []

    def _push_scope(self, name):
        self._scope_stack.append(self.current_scope)
        self.current_scope = name

    def _pop_scope(self):
        self.current_scope = self._scope_stack.pop()

    def visit_Import(self, node):
        for alias in node.names:
            local = alias.asname or alias.name
            self.imports[local] = (alias.name, alias.name)
            self.import_modules.append(alias.name)

    def visit_ImportFrom(self, node):
        mod = node.module or ''
        for alias in (node.names or []):
            if alias.name == '*':
                continue
            local = alias.asname or alias.name
            self.imports[local] = (mod, alias.name)
            self.import_modules.append(f"{mod}.{alias.name}")

    def visit_ClassDef(self, node):
        self.definitions[node.name] = (node.lineno, 'class', self.current_scope)
        self.class_bases[node.name] = [self._resolve_name(b) for b in node.bases]
        self._push_scope(node.name)
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.class_methods[node.name].append(item.name)
        self.generic_visit(node)
        self._pop_scope()

    def visit_FunctionDef(self, node):
        parent = self.current_scope
        full_name = f"{parent}.{node.name}" if parent else node.name
        self.definitions[full_name] = (node.lineno, 'function', parent)
        if not parent:
            self.definitions[node.name] = (node.lineno, 'function', None)
        self._push_scope(full_name)
        self.generic_visit(node)
        self._pop_scope()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node):
        name = self._resolve_call(node.func)
        if name:
            self.calls.append((self.current_scope, name, node.lineno))
        self.generic_visit(node)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.name_refs.append((self.current_scope, node.id, node.lineno))

    def visit_Attribute(self, node):
        name = self._resolve_name(node)
        if name:
            self.name_refs.append((self.current_scope, name, node.lineno))
        self.generic_visit(node)

    def _resolve_call(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._resolve_name(node.value)
            if base:
                return f"{base}.{node.attr}"
            return node.attr
        return None

    def _resolve_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            base = self._resolve_name(node.value)
            if base:
                return f"{base}.{node.attr}"
            return node.attr
        return None


class DeepAnalyzer:
    def __init__(self):
        self.files = get_py_files()
        self.parsed = {}
        self.builders = {}
        self.module_map = {}
        self.all_definitions = {}
        self.all_calls = []
        self.all_refs = []
        self.global_symbols = {}

    def parse_all(self):
        print(f"Parsing {len(self.files)} Python files...")
        errors = []
        for f in self.files:
            tree, src = parse_file(f)
            self.parsed[f] = (tree, src)
            self.module_map[get_module_name(f)] = f
            if tree is None:
                errors.append((f, src))
            else:
                builder = CallGraphBuilder(f)
                builder.visit(tree)
                self.builders[f] = builder
                for name, (lineno, dtype, parent) in builder.definitions.items():
                    self.all_definitions[(f, name)] = (lineno, dtype)
                    if parent is None:
                        if name not in self.global_symbols:
                            self.global_symbols[name] = []
                        self.global_symbols[name].append((f, lineno, dtype))
                for caller, callee, lineno in builder.calls:
                    self.all_calls.append((f, caller, callee, lineno))
                for scope, name, lineno in builder.name_refs:
                    self.all_refs.append((f, scope, name, lineno))
        if errors:
            print(f"\n  PARSE ERRORS ({len(errors)}):")
            for f, err in errors:
                print(f"    {rel(f)}: {err}")
        print()

    def trace_entry_points(self):
        print("=" * 80)
        print("1. ENTRY POINT TRACING")
        print("=" * 80)

        # Find run.py subcommands
        run_path = ROOT / 'run.py'
        tree, source = self.parsed.get(run_path, (None, None))
        if not tree or not source:
            print("  ERROR: Cannot parse run.py")
            return

        lines = source.split('\n')
        subcommands = []
        for i, line in enumerate(lines):
            if 'add_parser' in line and "'" in line:
                parts = line.split("'")
                if len(parts) >= 2:
                    subcommands.append((parts[1], i + 1))

        print(f"\n  run.py CLI subcommands: {[c[0] for c in subcommands]}")

        # For each subcommand, find its handler block and trace calls
        for cmd, _ in subcommands:
            print(f"\n  [{cmd}] ->")
            in_handler = False
            handler_indent = 0
            handler_lines = []
            for i, line in enumerate(lines):
                stripped = line.strip()
                if f"'{cmd}'" in stripped and 'args.command' in stripped:
                    in_handler = True
                    handler_indent = len(line) - len(line.lstrip())
                    continue
                if in_handler:
                    if not stripped:
                        continue
                    cur_indent = len(line) - len(line.lstrip())
                    if cur_indent <= handler_indent and not stripped.startswith('#'):
                        break
                    handler_lines.append((i + 1, stripped))

            # Extract function calls from handler
            called_funcs = []
            for lineno, code in handler_lines:
                # Simple extraction of function calls
                for node in ast.walk(ast.parse(f"_ = {code}" if '=' not in code.split('(')[0] else code, mode='exec')):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            called_funcs.append(node.func.id)
                        elif isinstance(node.func, ast.Attribute):
                            called_funcs.append(node.func.attr)

            if handler_lines:
                for lineno, code in handler_lines[:5]:
                    print(f"    L{lineno}: {code[:120]}")
                if len(handler_lines) > 5:
                    print(f"    ... ({len(handler_lines) - 5} more lines)")
            else:
                print(f"    (handler not found via static analysis)")

            if called_funcs:
                print(f"    Calls: {called_funcs}")

        # Script entry points
        print(f"\n  --- Script Entry Points ---")
        for f in self.files:
            if 'scripts/' not in rel(f):
                continue
            tree, source = self.parsed[f]
            if not tree:
                continue
            has_main_guard = False
            for node in ast.walk(tree):
                if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                    if (isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__'):
                        has_main_guard = True
            b = self.builders[f]
            top_funcs = [n for n, (_, dt, p) in b.definitions.items() if dt == 'function' and p is None]
            has_main = 'main' in top_funcs
            if has_main_guard or has_main:
                print(f"    {rel(f)}: entry={'__main__ guard' if has_main_guard else 'main()'}, funcs={top_funcs[:5]}")
            else:
                # Check if file has bare code (not just defs/imports)
                bare_code = False
                for node in ast.iter_child_nodes(tree):
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
                                             ast.Import, ast.ImportFrom, ast.Expr, ast.Assign)):
                        bare_code = True
                        break
                    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                        bare_code = True
                        break
                if bare_code:
                    print(f"    {rel(f)}: has bare executable code (no guard)")

    def find_dead_code(self):
        print("\n" + "=" * 80)
        print("2. DEAD CODE - UNREFERENCED DEFINITIONS")
        print("=" * 80)

        print("\n  --- Top-level functions/classes never imported or referenced from other files ---")
        dead_count = 0
        dead_items = []

        for name, locations in sorted(self.global_symbols.items()):
            if name.startswith('_'):
                continue
            if name in ('main', 'setUp', 'tearDown', 'setUpClass', 'tearDownClass'):
                continue

            for fpath, lineno, dtype in locations:
                rp = rel(fpath)
                if 'test' in rp.lower() or '__init__' in rp or 'scripts/' in rp:
                    continue

                referenced = False
                for other_f in self.files:
                    if other_f == fpath:
                        continue
                    ob = self.builders.get(other_f)
                    if not ob:
                        continue
                    for local, (mod, orig) in ob.imports.items():
                        if orig == name:
                            referenced = True
                            break
                    if referenced:
                        break
                    for _, callee, _ in ob.calls:
                        if callee.split('.')[-1] == name:
                            referenced = True
                            break
                    if referenced:
                        break
                    for _, ref_name, _ in ob.name_refs:
                        if ref_name.split('.')[-1] == name:
                            referenced = True
                            break
                    if referenced:
                        break

                if not referenced:
                    dead_items.append((rp, lineno, dtype, name))
                    dead_count += 1

        for rp, lineno, dtype, name in dead_items:
            print(f"    {rp}:{lineno} - {dtype} \"{name}\"")
        print(f"\n  Total: {dead_count} potentially dead definitions")

    def find_logic_issues(self):
        print("\n" + "=" * 80)
        print("3. LOGIC ISSUES")
        print("=" * 80)

        issues = []

        for f in self.files:
            tree, source = self.parsed[f]
            if tree is None:
                continue
            rp = rel(f)

            # Unreachable code
            for node in ast.walk(tree):
                bodies = []
                if hasattr(node, 'body') and isinstance(node.body, list):
                    bodies.append(node.body)
                if hasattr(node, 'orelse') and isinstance(node.orelse, list):
                    bodies.append(node.orelse)
                if hasattr(node, 'finalbody') and isinstance(node.finalbody, list):
                    bodies.append(node.finalbody)
                if hasattr(node, 'handlers'):
                    for h in node.handlers:
                        if hasattr(h, 'body'):
                            bodies.append(h.body)
                for body in bodies:
                    for i, stmt in enumerate(body):
                        if isinstance(stmt, (ast.Return, ast.Break, ast.Continue, ast.Raise)):
                            if i < len(body) - 1:
                                nxt = body[i + 1]
                                if not isinstance(nxt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                                    issues.append(('UNREACHABLE', rp, nxt.lineno,
                                        f"after {type(stmt).__name__.lower()} at L{stmt.lineno}"))

            # Dead branches
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    if isinstance(node.test, ast.Constant):
                        if node.test.value is False:
                            issues.append(('DEAD_BRANCH', rp, node.lineno, "if False: block"))
                        elif node.test.value is True and node.orelse:
                            issues.append(('DEAD_BRANCH', rp, node.lineno, "if True: else block is dead"))

            # Inconsistent returns
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith('_') or node.name in ('setUp', 'tearDown', 'apply', '__init__'):
                        continue
                    has_return_val = False
                    has_return_none = False
                    ret_count = 0
                    for child in ast.walk(node):
                        if child is node:
                            continue
                        # Skip nested functions
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child is not node:
                            continue
                        if isinstance(child, ast.Return):
                            ret_count += 1
                            if child.value is not None:
                                has_return_val = True
                            else:
                                has_return_none = True
                    if has_return_val and has_return_none and ret_count > 1:
                        issues.append(('INCONSISTENT_RETURN', rp, node.lineno,
                            f"\"{node.name}\" - some paths return value, others return None"))

            # Swallowed errors
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    for handler in node.handlers:
                        if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                            if handler.type is None:
                                issues.append(('SWALLOWED_ERROR', rp, handler.lineno,
                                    "bare except: pass - swallows ALL errors"))
                            elif isinstance(handler.type, ast.Name) and handler.type.id == 'Exception':
                                issues.append(('SWALLOWED_ERROR', rp, handler.lineno,
                                    "except Exception: pass"))

            # Duplicate methods in same class
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = collections.defaultdict(list)
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods[item.name].append(item.lineno)
                    for mname, lns in methods.items():
                        if len(lns) > 1:
                            issues.append(('DUPLICATE_METHOD', rp, lns[-1],
                                f"\"{mname}\" x{len(lns)} in \"{node.name}\" at lines {lns}"))

            # Comparison to None without 'is'
            for node in ast.walk(tree):
                if isinstance(node, ast.Compare):
                    for op, comp in zip(node.ops, node.comparators):
                        if isinstance(comp, ast.Constant) and comp.value is None:
                            if isinstance(op, (ast.Eq, ast.NotEq)):
                                issues.append(('NONE_COMPARISON', rp, node.lineno,
                                    "Use 'is None' / 'is not None' instead of ==/!="))

        # Print grouped
        types_order = ['UNREACHABLE', 'DEAD_BRANCH', 'INCONSISTENT_RETURN',
                       'SWALLOWED_ERROR', 'DUPLICATE_METHOD', 'NONE_COMPARISON']
        issue_map = collections.defaultdict(list)
        for itype, *rest in issues:
            issue_map[itype].append(rest)

        for itype in types_order:
            items = issue_map.get(itype, [])
            print(f"\n  --- {itype} ({len(items)}) ---")
            for rp, lineno, msg in items:
                print(f"    {rp}:{lineno} - {msg}")
            if not items:
                print("    None found.")

    def check_cross_module(self):
        print("\n" + "=" * 80)
        print("4. CROSS-MODULE CONSISTENCY")
        print("=" * 80)

        # models.py vs core/models.py
        print("\n  --- Duplicate Modules ---")
        m1, m2 = ROOT / 'models.py', ROOT / 'core' / 'models.py'
        if m1.exists() and m2.exists():
            with open(m1) as f: s1 = f.read()
            with open(m2) as f: s2 = f.read()
            if s1 == s2:
                print(f"    EXACT DUPLICATE: models.py == core/models.py")
                importers = []
                for f in self.files:
                    b = self.builders.get(f)
                    if not b:
                        continue
                    for mod_path in b.import_modules:
                        if 'core.models' in mod_path or mod_path.startswith('core.'):
                            importers.append(rel(f))
                            break
                if importers:
                    print(f"    Files importing from core/: {importers}")
                else:
                    print(f"    NO files import from core/ -> entire core/ directory is DEAD CODE")
            else:
                print(f"    models.py and core/models.py DIFFER - possible drift!")

        # Constraint registration
        print("\n  --- Constraint Registration in main_staged.py ---")
        main_path = ROOT / 'main_staged.py'
        if main_path in self.builders:
            tree, source = self.parsed[main_path]
            # Find constraint class names in STAGES dicts
            constraint_keywords = {'Constraint', 'Booking', 'Maitland', 'Broadmeadow',
                                   'Spacing', 'Adjacency', 'Alignment', 'FiftyFifty',
                                   'Grouping', 'Timeslot', 'Preferred', 'Spread', 'Symmetry'}
            referenced = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    if any(kw in node.id for kw in constraint_keywords):
                        referenced.add(node.id)

            existing = set()
            for f in self.files:
                if 'constraints/' in rel(f):
                    b = self.builders.get(f)
                    if b:
                        for name, (_, dtype, _) in b.definitions.items():
                            if dtype == 'class':
                                existing.add(name)

            missing = referenced - existing
            extra = set()
            for c in existing:
                if c in ('Constraint', 'ConstraintAI', 'SoftConstraint', 'ABC'):
                    continue
                if c not in referenced and not c.endswith('Soft') and 'Config' not in c:
                    extra.add(c)

            if missing:
                print(f"    MISSING: {sorted(missing)}")
            else:
                print(f"    All {len(referenced)} constraint refs resolve OK")
            if extra:
                print(f"    UNUSED constraint classes (defined but not in STAGES): {sorted(extra)}")

        # Duplicate implementations
        print("\n  --- Duplicate Function Implementations ---")
        for name, locations in sorted(self.global_symbols.items()):
            if name.startswith('_') or name.startswith('test'):
                continue
            prod = [(f, ln, dt) for f, ln, dt in locations
                    if 'test' not in rel(f).lower() and '__init__' not in rel(f)]
            if len(prod) > 1:
                dirs = set(f.parent for f, _, _ in prod)
                if len(dirs) > 1:
                    print(f"    \"{name}\":")
                    for fp, ln, dt in prod:
                        print(f"      {rel(fp)}:{ln} ({dt})")

    def check_data_flow(self):
        print("\n" + "=" * 80)
        print("5. DATA FLOW")
        print("=" * 80)

        data_written = collections.defaultdict(list)
        data_read = collections.defaultdict(list)

        for f in self.files:
            tree, _ = self.parsed[f]
            if tree is None:
                continue
            rp = rel(f)
            for node in ast.walk(tree):
                if isinstance(node, ast.Subscript):
                    if (isinstance(node.value, ast.Name) and node.value.id == 'data' and
                        isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
                        key = node.slice.value
                        if isinstance(node.ctx, ast.Store):
                            data_written[key].append((rp, node.lineno))
                        else:
                            data_read[key].append((rp, node.lineno))
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and
                    node.func.attr == 'get' and isinstance(node.func.value, ast.Name) and
                    node.func.value.id == 'data' and node.args and
                    isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                    data_read[node.args[0].value].append((rp, node.lineno))

        write_only = set(data_written) - set(data_read)
        read_only = set(data_read) - set(data_written)

        print(f"\n  data dict: {len(data_read)} keys read, {len(data_written)} keys written")
        print(f"  {len(read_only)} read-only (from config loading), {len(write_only)} write-only")

        if write_only:
            print(f"\n  --- Keys WRITTEN but never READ (dead data?) ---")
            for k in sorted(write_only):
                locs = data_written[k]
                print(f"    '{k}': {', '.join(f'{r}:{l}' for r, l in locs[:3])}")

        # Show all keys for completeness
        print(f"\n  --- All data keys used across codebase ---")
        all_keys = sorted(set(data_read) | set(data_written))
        for k in all_keys:
            r = len(data_read.get(k, []))
            w = len(data_written.get(k, []))
            status = "OK" if r > 0 and w > 0 else ("READ-ONLY" if r > 0 else "WRITE-ONLY")
            files_r = set(loc[0].split('/')[0] for loc in data_read.get(k, []))
            files_w = set(loc[0].split('/')[0] for loc in data_written.get(k, []))
            print(f"    '{k}': {w}W/{r}R [{status}]")

    def find_islands(self):
        print("\n" + "=" * 80)
        print("6. DEAD ISLAND DETECTION")
        print("=" * 80)

        # File-level dependency graph
        file_deps = collections.defaultdict(set)
        for f in self.files:
            b = self.builders.get(f)
            if not b:
                continue
            for local, (mod, orig) in b.imports.items():
                for mod_name, mod_file in self.module_map.items():
                    if mod and (mod_name == mod or mod_name.endswith('.' + mod.split('.')[0]) or
                               mod.startswith(mod_name)):
                        file_deps[f].add(mod_file)

        # BFS from all entry points
        entry_files = set()
        entry_files.add(ROOT / 'run.py')
        for f in self.files:
            rp = rel(f)
            if 'test' in f.name.lower() or f.name == 'conftest.py' or 'scripts/' in rp:
                entry_files.add(f)

        reachable = set(entry_files)
        queue = list(entry_files)
        while queue:
            current = queue.pop(0)
            for dep in file_deps.get(current, set()):
                if dep not in reachable:
                    reachable.add(dep)
                    queue.append(dep)

        # __init__.py -> all siblings reachable
        for f in list(reachable):
            if f.name == '__init__.py':
                for other in self.files:
                    if other.parent == f.parent and other not in reachable:
                        reachable.add(other)
                        queue.append(other)
        while queue:
            current = queue.pop(0)
            for dep in file_deps.get(current, set()):
                if dep not in reachable:
                    reachable.add(dep)
                    queue.append(dep)

        unreachable = sorted(set(self.files) - reachable)
        if unreachable:
            print(f"\n  UNREACHABLE FILES ({len(unreachable)}):")
            for f in unreachable:
                print(f"    {rel(f)}")
        else:
            print("\n  All files reachable from entry points. No dead islands.")

    def run_all(self):
        self.parse_all()
        self.trace_entry_points()
        self.find_dead_code()
        self.find_logic_issues()
        self.check_cross_module()
        self.check_data_flow()
        self.find_islands()
        print("\n" + "=" * 80)
        print("DEEP ANALYSIS COMPLETE")
        print("=" * 80)


if __name__ == '__main__':
    analyzer = DeepAnalyzer()
    analyzer.run_all()
