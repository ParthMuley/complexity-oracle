from __future__ import annotations

import ast
import builtins

from complexity_oracle.models.analysis import Complexity, ParseResult, UnresolvedCall

_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))


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
        self._check_recursion(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._defined_names.add(node.name)
        self.generic_visit(node)

    # -- loops --------------------------------------------------------------

    def _enter_loop(self) -> None:
        self._current_loop_depth += 1
        if self._current_loop_depth > self.max_loop_depth:
            self.max_loop_depth = self._current_loop_depth

    def _exit_loop(self) -> None:
        self._current_loop_depth -= 1

    def visit_For(self, node: ast.For) -> None:
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
