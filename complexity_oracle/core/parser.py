from __future__ import annotations

import ast
import builtins

from complexity_oracle.models.analysis import Complexity, ParseResult, UnresolvedCall

_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))

# Builtins that have non-trivial complexity — worth flagging even though they
# are not "unknown".  Keyed by the bare function/method name.
_EXPENSIVE_BUILTINS: dict[str, str] = {
    "sorted":   "O(n log n) — Timsort; calling inside a loop raises overall complexity",
    "sort":     "O(n log n) — list.sort; calling inside a loop raises overall complexity",
    "max":      "O(n) — linear scan; calling inside a loop raises overall complexity",
    "min":      "O(n) — linear scan; calling inside a loop raises overall complexity",
    "sum":      "O(n) — linear scan; calling inside a loop raises overall complexity",
    "reversed": "O(n) — iterates full sequence",
    "index":    "O(n) — linear search on list; use a dict or set for O(1) lookup",
    "count":    "O(n) — linear scan; repeated calls inside a loop are O(n²)",
}


class _ComplexityVisitor(ast.NodeVisitor):
    """Walks a Python AST to extract complexity-relevant information."""

    def __init__(self) -> None:
        self.functions: list[str] = []
        self.max_loop_depth: int = 0
        self.unresolved_calls: list[UnresolvedCall] = []
        self.flagged_lines: dict[int, str] = {}

        self._current_loop_depth: int = 0
        self._defined_names: set[str] = set()
        self._imported_names: set[str] = set()

    # -- functions ----------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self._defined_names.add(node.name)
        # Track all parameter names so method calls on them aren't flagged.
        for arg in (
            node.args.posonlyargs
            + node.args.args
            + node.args.kwonlyargs
            + ([node.args.vararg] if node.args.vararg else [])
            + ([node.args.kwarg] if node.args.kwarg else [])
        ):
            self._defined_names.add(arg.arg)
        self._check_recursion(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._defined_names.add(node.name)
        self.generic_visit(node)

    # -- assignments --------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        """Track assigned names so method calls on local vars aren't flagged."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._defined_names.add(target.id)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._defined_names.add(node.target.id)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Name):
            self._defined_names.add(node.target.id)
        self.generic_visit(node)

    # -- loops --------------------------------------------------------------

    def _enter_loop(self) -> None:
        self._current_loop_depth += 1
        if self._current_loop_depth > self.max_loop_depth:
            self.max_loop_depth = self._current_loop_depth

    def _exit_loop(self) -> None:
        self._current_loop_depth -= 1

    def visit_For(self, node: ast.For) -> None:
        # Track loop variable(s) as locally defined.
        if isinstance(node.target, ast.Name):
            self._defined_names.add(node.target.id)
        elif isinstance(node.target, ast.Tuple):
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    self._defined_names.add(elt.id)
        self._enter_loop()
        self.generic_visit(node)
        self._exit_loop()

    def visit_While(self, node: ast.While) -> None:
        self._enter_loop()
        self.generic_visit(node)
        self._exit_loop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        for generator in node.generators:
            self._enter_loop()
        self.generic_visit(node)
        for _ in node.generators:
            self._exit_loop()

    visit_SetComp = visit_ListComp  # type: ignore[assignment]
    visit_GeneratorExp = visit_ListComp  # type: ignore[assignment]

    def visit_DictComp(self, node: ast.DictComp) -> None:
        for generator in node.generators:
            self._enter_loop()
        self.generic_visit(node)
        for _ in node.generators:
            self._exit_loop()

    # -- imports ------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._imported_names.add(alias.asname or alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self._imported_names.add(alias.asname or alias.name)

    # -- calls --------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        name = self._call_name(node.func)
        if name is not None:
            # The "leaf" name for method calls: `seen.append` → "append"
            leaf = name.split(".")[-1]
            root = name.split(".")[0]

            if root not in _BUILTIN_NAMES:
                if root in self._imported_names:
                    self.unresolved_calls.append(
                        UnresolvedCall(name, node.lineno, "External import (complexity unknown)")
                    )
                elif root not in self._defined_names:
                    self.unresolved_calls.append(
                        UnresolvedCall(name, node.lineno, "Undefined function (not imported or defined locally)")
                    )
                # Method on a locally-defined variable with known expensive complexity
                # e.g. haystack.index(x) inside a loop → O(n) per call → O(n²) overall.
                elif leaf in _EXPENSIVE_BUILTINS and self._current_loop_depth > 0:
                    hint = _EXPENSIVE_BUILTINS[leaf]
                    self.unresolved_calls.append(
                        UnresolvedCall(name, node.lineno, f"Expensive builtin inside loop: {hint}")
                    )
            # Flag expensive top-level builtins (sorted/max/min/sum) inside loops.
            elif (name in _EXPENSIVE_BUILTINS or leaf in _EXPENSIVE_BUILTINS) and self._current_loop_depth > 0:
                hint = _EXPENSIVE_BUILTINS.get(name) or _EXPENSIVE_BUILTINS.get(leaf, "")
                self.unresolved_calls.append(
                    UnresolvedCall(name, node.lineno, f"Expensive builtin inside loop: {hint}")
                )
        self.generic_visit(node)

    # -- helpers ------------------------------------------------------------

    def _check_recursion(self, func_node: ast.FunctionDef) -> None:
        """Flag direct recursive calls inside *func_node*."""
        for child in ast.walk(func_node):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == func_node.name
            ):
                self.flagged_lines[child.lineno] = f"Recursive call to {func_node.name}"

    @staticmethod
    def _call_name(node: ast.expr) -> str | None:
        """Extract a dotted name from a Call's func node, or None."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = _ComplexityVisitor._call_name(node.value)
            if base is not None:
                return f"{base}.{node.attr}"
            return node.attr
        return None


def _estimate_complexity(visitor: _ComplexityVisitor) -> Complexity:
    """Heuristic: derive a static complexity class from AST observations."""
    if visitor.flagged_lines:
        for reason in visitor.flagged_lines.values():
            if "Recursive call" in reason:
                return Complexity.UNKNOWN

    if visitor.max_loop_depth == 0:
        return Complexity.O_1
    if visitor.max_loop_depth == 1:
        return Complexity.O_N
    return Complexity.O_N2


def parse_file(file_path: str) -> ParseResult:
    """Parse a Python file and extract complexity information.

    Raises FileNotFoundError if the file does not exist.
    Raises SyntaxError if the file contains invalid Python.
    """
    with open(file_path, encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source, filename=file_path)
    visitor = _ComplexityVisitor()
    visitor.visit(tree)

    return ParseResult(
        functions=visitor.functions,
        max_loop_depth=visitor.max_loop_depth,
        static_complexity=_estimate_complexity(visitor),
        unresolved_calls=visitor.unresolved_calls,
        flagged_lines=visitor.flagged_lines,
    )
