"""AST-based scan for remaining dummy slot/variable references in a codebase."""
import ast
import os
import sys

TARGET_DIR = sys.argv[1] if len(sys.argv) > 1 else "."

findings = []

def add(filepath, lineno, node_type, description):
    findings.append((filepath, lineno, node_type, description))

class DummyVisitor(ast.NodeVisitor):
    def __init__(self, filepath):
        self.filepath = filepath

    # 1. Variable/attribute names containing "dummy" (case insensitive)
    def visit_Name(self, node):
        if "dummy" in node.id.lower():
            add(self.filepath, node.lineno, "Name", f"Name '{node.id}'")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if "dummy" in node.attr.lower():
            add(self.filepath, node.lineno, "Attribute", f"Attribute '.{node.attr}'")
        # Check for self.Y
        if node.attr == "Y" and isinstance(node.value, ast.Name) and node.value.id == "self":
            add(self.filepath, node.lineno, "Attribute", "self.Y reference")
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        if "dummy" in node.name.lower():
            add(self.filepath, node.lineno, "FunctionDef", f"Function '{node.name}'")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if "dummy" in node.name.lower():
            add(self.filepath, node.lineno, "AsyncFunctionDef", f"Async function '{node.name}'")
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        if "dummy" in node.name.lower():
            add(self.filepath, node.lineno, "ClassDef", f"Class '{node.name}'")
        self.generic_visit(node)

    # 2. String literals containing "dummy" (case insensitive)
    def visit_Constant(self, node):
        if isinstance(node.value, str) and "dummy" in node.value.lower():
            snippet = node.value[:80].replace('\n', '\\n')
            add(self.filepath, node.lineno, "StringLiteral", f"String containing 'dummy': \"{snippet}\"")
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        # f-strings: check the Constant parts
        for val in node.values:
            if isinstance(val, ast.Constant) and isinstance(val.value, str) and "dummy" in val.value.lower():
                snippet = val.value[:80].replace('\n', '\\n')
                add(self.filepath, node.lineno, "FString", f"F-string part containing 'dummy': \"{snippet}\"")
        self.generic_visit(node)

    # 3. Dict key access for 'num_dummy_timeslots' or 'dummy_slots'
    def visit_Subscript(self, node):
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            if node.slice.value in ('num_dummy_timeslots', 'dummy_slots'):
                add(self.filepath, node.lineno, "Subscript", f"Dict key access ['{node.slice.value}']")
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check .get('num_dummy_timeslots') or .get('dummy_slots')
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            for arg in node.args:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    if arg.value in ('num_dummy_timeslots', 'dummy_slots'):
                        add(self.filepath, node.lineno, "Call(.get)", f".get('{arg.value}')")

        # 4. Tuple unpacking from generate_X — 3-element target
        # Handled in visit_Assign instead

        self.generic_visit(node)

    def visit_Assign(self, node):
        # Check for 3-element tuple unpacking from generate_X
        if isinstance(node.value, ast.Call):
            func = node.value.func
            func_name = None
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name == "generate_X":
                if isinstance(node.targets[0], ast.Tuple):
                    n = len(node.targets[0].elts)
                    if n == 3:
                        names = []
                        for e in node.targets[0].elts:
                            if isinstance(e, ast.Name):
                                names.append(e.id)
                            else:
                                names.append("?")
                        add(self.filepath, node.lineno, "Assign(Tuple3)", f"3-element unpack from generate_X: {', '.join(names)}")

        self.generic_visit(node)

    # 5. len(key) == 4 comparisons
    def visit_Compare(self, node):
        # Check for len(key) == 4 or len(key) < 5 patterns
        if isinstance(node.left, ast.Call):
            func = node.left.func
            if isinstance(func, ast.Name) and func.id == "len":
                if node.left.args and isinstance(node.left.args[0], ast.Name) and node.left.args[0].id == "key":
                    for op, comparator in zip(node.ops, node.comparators):
                        if isinstance(comparator, ast.Constant) and comparator.value == 4:
                            op_str = type(op).__name__
                            add(self.filepath, node.lineno, "Compare", f"len(key) {op_str} 4")
        self.generic_visit(node)


def scan_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
        visitor = DummyVisitor(filepath)
        visitor.visit(tree)
    except SyntaxError as e:
        print(f"  [SKIP] SyntaxError in {filepath}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"  [SKIP] Error in {filepath}: {e}", file=sys.stderr)


def main():
    py_files = []
    for root, dirs, files in os.walk(TARGET_DIR):
        # Skip common non-source dirs
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", ".venv", "venv", "node_modules", ".tox")]
        for f in files:
            if f.endswith(".py"):
                py_files.append(os.path.join(root, f))

    print(f"Scanning {len(py_files)} Python files in {TARGET_DIR}\n")

    for pf in sorted(py_files):
        scan_file(pf)

    if not findings:
        print("NO FINDINGS — codebase is clean of dummy references.")
        return

    print(f"{'='*100}")
    print(f"FOUND {len(findings)} items:")
    print(f"{'='*100}")
    for filepath, lineno, node_type, desc in sorted(findings):
        rel = os.path.relpath(filepath, TARGET_DIR)
        print(f"  {rel}:{lineno}  [{node_type}]  {desc}")

    print(f"\n{'='*100}")
    print("Summary by category:")
    from collections import Counter
    cats = Counter(nt for _, _, nt, _ in findings)
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
