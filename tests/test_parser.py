import pytest

from complexity_oracle.models.analysis import Complexity, UnresolvedCall
from complexity_oracle.core.parser import parse_file


def _write(tmp_path, code: str) -> str:
    """Write code to a temp .py file and return its path."""
    p = tmp_path / "test_input.py"
    p.write_text(code)
    return str(p)


# -- basic parsing ----------------------------------------------------------


class TestBasicParsing:
    def test_empty_file(self, tmp_path):
        result = parse_file(_write(tmp_path, ""))
        assert result.functions == []
        assert result.max_loop_depth == 0
        assert result.static_complexity == Complexity.O_1
        assert result.unresolved_calls == []
        assert result.flagged_lines == {}

    def test_single_function(self, tmp_path):
        result = parse_file(_write(tmp_path, "def hello():\n    return 1\n"))
        assert result.functions == ["hello"]
        assert result.static_complexity == Complexity.O_1

    def test_multiple_functions(self, tmp_path):
        code = (
            "def a(): pass\n"
            "def b(): pass\n"
            "class C:\n"
            "    def m(self): pass\n"
        )
        result = parse_file(_write(tmp_path, code))
        assert "a" in result.functions
        assert "b" in result.functions
        assert "m" in result.functions


# -- loop depth -------------------------------------------------------------


class TestLoopDepth:
    def test_no_loops(self, tmp_path):
        result = parse_file(_write(tmp_path, "x = 1\n"))
        assert result.max_loop_depth == 0
        assert result.static_complexity == Complexity.O_1

    def test_single_for(self, tmp_path):
        code = "for i in range(10):\n    pass\n"
        result = parse_file(_write(tmp_path, code))
        assert result.max_loop_depth == 1
        assert result.static_complexity == Complexity.O_N

    def test_single_while(self, tmp_path):
        code = "n = 10\nwhile n > 0:\n    n -= 1\n"
        result = parse_file(_write(tmp_path, code))
        assert result.max_loop_depth == 1
        assert result.static_complexity == Complexity.O_N

    def test_nested_loops(self, tmp_path):
        code = "for i in range(10):\n    for j in range(10):\n        pass\n"
        result = parse_file(_write(tmp_path, code))
        assert result.max_loop_depth == 2
        assert result.static_complexity == Complexity.O_N2

    def test_triple_nested(self, tmp_path):
        code = (
            "for i in range(10):\n"
            "    for j in range(10):\n"
            "        for k in range(10):\n"
            "            pass\n"
        )
        result = parse_file(_write(tmp_path, code))
        assert result.max_loop_depth == 3
        assert result.static_complexity == Complexity.O_N2

    def test_sequential_loops_not_nested(self, tmp_path):
        code = "for i in range(10):\n    pass\nfor j in range(10):\n    pass\n"
        result = parse_file(_write(tmp_path, code))
        assert result.max_loop_depth == 1
        assert result.static_complexity == Complexity.O_N

    def test_list_comprehension(self, tmp_path):
        code = "x = [i for i in range(10)]\n"
        result = parse_file(_write(tmp_path, code))
        assert result.max_loop_depth == 1
        assert result.static_complexity == Complexity.O_N

    def test_nested_comprehension(self, tmp_path):
        code = "x = [[i*j for j in range(10)] for i in range(10)]\n"
        result = parse_file(_write(tmp_path, code))
        assert result.max_loop_depth == 2
        assert result.static_complexity == Complexity.O_N2

    def test_comprehension_with_multiple_fors(self, tmp_path):
        """[x*y for x in a for y in b] is a flat double loop."""
        code = "x = [i*j for i in range(10) for j in range(10)]\n"
        result = parse_file(_write(tmp_path, code))
        assert result.max_loop_depth == 2


# -- recursion --------------------------------------------------------------


class TestRecursion:
    def test_direct_recursion(self, tmp_path):
        code = (
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
        )
        result = parse_file(_write(tmp_path, code))
        assert result.static_complexity == Complexity.UNKNOWN
        assert any("Recursive call" in msg for msg in result.flagged_lines.values())

    def test_non_recursive_same_name_call(self, tmp_path):
        """Calling a different function with the same name as a local is NOT recursion."""
        code = "def foo():\n    pass\ndef bar():\n    foo()\n"
        result = parse_file(_write(tmp_path, code))
        # foo() inside bar() is not recursion
        assert not any("Recursive call" in msg for msg in result.flagged_lines.values())


# -- unresolved calls -------------------------------------------------------


class TestUnresolvedCalls:
    def test_builtins_not_flagged(self, tmp_path):
        code = "x = len([1, 2, 3])\ny = sorted([3, 1, 2])\nz = range(10)\n"
        result = parse_file(_write(tmp_path, code))
        assert result.unresolved_calls == []

    def test_local_function_not_flagged(self, tmp_path):
        code = "def helper(x):\n    return x\nresult = helper(5)\n"
        result = parse_file(_write(tmp_path, code))
        assert result.unresolved_calls == []

    def test_imported_flagged_as_external(self, tmp_path):
        code = "import json\nresult = json.loads('{}')\n"
        result = parse_file(_write(tmp_path, code))
        assert len(result.unresolved_calls) == 1
        assert result.unresolved_calls[0].name == "json.loads"
        assert "External import" in result.unresolved_calls[0].reason

    def test_from_import_flagged(self, tmp_path):
        code = "from os.path import exists\nexists('/tmp')\n"
        result = parse_file(_write(tmp_path, code))
        assert len(result.unresolved_calls) == 1
        assert result.unresolved_calls[0].name == "exists"
        assert "External import" in result.unresolved_calls[0].reason

    def test_undefined_function_flagged(self, tmp_path):
        code = "result = mystery_func(42)\n"
        result = parse_file(_write(tmp_path, code))
        assert len(result.unresolved_calls) == 1
        assert result.unresolved_calls[0].name == "mystery_func"
        assert "Undefined" in result.unresolved_calls[0].reason


# -- local variable tracking ------------------------------------------------


class TestLocalVariableTracking:
    """Method calls on local variables must NOT be flagged as unresolved."""

    def test_method_on_assigned_variable_not_flagged(self, tmp_path):
        code = (
            "def remove_dups(items):\n"
            "    seen = []\n"
            "    for x in items:\n"
            "        if x not in seen:\n"
            "            seen.append(x)\n"
            "    return seen\n"
        )
        result = parse_file(_write(tmp_path, code))
        names = [c.name for c in result.unresolved_calls]
        assert "seen.append" not in names

    def test_method_on_function_param_not_flagged(self, tmp_path):
        code = (
            "def process(data):\n"
            "    result = []\n"
            "    for x in data:\n"
            "        result.append(x * 2)\n"
            "    return result\n"
        )
        result = parse_file(_write(tmp_path, code))
        names = [c.name for c in result.unresolved_calls]
        assert "data.append" not in names
        assert "result.append" not in names

    def test_loop_variable_not_flagged(self, tmp_path):
        code = (
            "def process(items):\n"
            "    for item in items:\n"
            "        item.strip()\n"
        )
        result = parse_file(_write(tmp_path, code))
        names = [c.name for c in result.unresolved_calls]
        assert "item.strip" not in names

    def test_annotated_assignment_not_flagged(self, tmp_path):
        code = (
            "def process():\n"
            "    result: list = []\n"
            "    result.append(1)\n"
            "    return result\n"
        )
        result = parse_file(_write(tmp_path, code))
        names = [c.name for c in result.unresolved_calls]
        assert "result.append" not in names

    def test_tuple_unpack_loop_var_not_flagged(self, tmp_path):
        code = (
            "def process(pairs):\n"
            "    for k, v in pairs:\n"
            "        k.upper()\n"
        )
        result = parse_file(_write(tmp_path, code))
        names = [c.name for c in result.unresolved_calls]
        assert "k.upper" not in names


# -- expensive builtins -------------------------------------------------------


class TestExpensiveBuiltins:
    """sorted/max/min/sum inside loops should be flagged as expensive."""

    def test_sorted_inside_loop_flagged(self, tmp_path):
        code = (
            "def f(matrix):\n"
            "    for row in matrix:\n"
            "        sorted(row)\n"
        )
        result = parse_file(_write(tmp_path, code))
        names = [c.name for c in result.unresolved_calls]
        assert "sorted" in names

    def test_sorted_outside_loop_not_flagged(self, tmp_path):
        code = "result = sorted([3, 1, 2])\n"
        result = parse_file(_write(tmp_path, code))
        assert result.unresolved_calls == []

    def test_max_inside_loop_flagged(self, tmp_path):
        code = (
            "def f(rows):\n"
            "    for r in rows:\n"
            "        max(r)\n"
        )
        result = parse_file(_write(tmp_path, code))
        names = [c.name for c in result.unresolved_calls]
        assert "max" in names

    def test_list_index_inside_loop_flagged(self, tmp_path):
        code = (
            "def f(data):\n"
            "    haystack = list(data)\n"
            "    for x in data:\n"
            "        haystack.index(x)\n"
        )
        result = parse_file(_write(tmp_path, code))
        names = [c.name for c in result.unresolved_calls]
        assert "haystack.index" in names

    def test_expensive_builtin_reason_mentions_loop(self, tmp_path):
        code = (
            "def f(matrix):\n"
            "    for row in matrix:\n"
            "        sorted(row)\n"
        )
        result = parse_file(_write(tmp_path, code))
        reasons = [c.reason for c in result.unresolved_calls if c.name == "sorted"]
        assert len(reasons) == 1
        assert "loop" in reasons[0].lower()


# -- async -------------------------------------------------------------------


class TestAsync:
    def test_async_function_detected(self, tmp_path):
        code = "async def fetch():\n    pass\n"
        result = parse_file(_write(tmp_path, code))
        assert "fetch" in result.functions

    def test_loop_inside_async(self, tmp_path):
        code = "async def fetch(items):\n    for i in items:\n        pass\n"
        result = parse_file(_write(tmp_path, code))
        assert result.max_loop_depth == 1


# -- error handling ----------------------------------------------------------


class TestErrorHandling:
    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            parse_file("/no/such/file.py")

    def test_invalid_syntax(self, tmp_path):
        result_path = _write(tmp_path, "def broken(\n")
        with pytest.raises(SyntaxError):
            parse_file(result_path)


# -- contract ----------------------------------------------------------------


class TestParseResultContract:
    def test_field_types(self, tmp_path):
        code = "import os\ndef f(n):\n    for i in range(n):\n        os.getcwd()\n"
        result = parse_file(_write(tmp_path, code))

        assert isinstance(result.functions, list)
        assert isinstance(result.max_loop_depth, int)
        assert isinstance(result.static_complexity, Complexity)
        assert isinstance(result.unresolved_calls, list)
        assert all(isinstance(c, UnresolvedCall) for c in result.unresolved_calls)
        assert isinstance(result.flagged_lines, dict)
