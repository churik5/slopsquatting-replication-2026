"""Python package extractor — ast first, tree-sitter fallback.

Given a generated response, returns the set of top-level import names.
Fence-aware: if the whole response contains markdown fenced blocks, parses each
block in turn; otherwise tries the whole response as Python.

Output is DEDUPED per generation but preserves first-seen order.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    _TS_LANG = Language(tspython.language())
    _TS_PARSER: Parser | None = Parser(_TS_LANG)
except Exception:  # noqa: BLE001 — optional dep
    _TS_PARSER = None


_FENCE_RE = re.compile(r"```(?:python|py|)\s*\n?([\s\S]+?)```", re.IGNORECASE)


@lru_cache(maxsize=1)
def _stdlib_names() -> frozenset[str]:
    """Return names in the Python stdlib for 3.11 (and common aliases)."""
    try:
        from stdlib_list import stdlib_list  # noqa: PLC0415

        names = set(stdlib_list("3.11"))
    except Exception:  # noqa: BLE001
        names = set()
    # Add a few common aliases / submodules top-levels that stdlib_list can miss.
    names.update(
        {
            "__future__",
            "typing",
            "collections",
            "importlib",
            "urllib",
            "xml",
            "email",
            "http",
            "concurrent",
            "asyncio",
            "logging",
            "multiprocessing",
            "_thread",
            "itertools",
            "functools",
        }
    )
    return frozenset(names)


def _is_stdlib(top_level: str) -> bool:
    return top_level in _stdlib_names()


def _extract_imports_from_tree(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top:
                    out.append(top)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                top = node.module.split(".")[0]
                if top:
                    out.append(top)
    return out


def _extract_via_tree_sitter(code: str) -> list[str]:
    if _TS_PARSER is None:
        return []
    try:
        tree = _TS_PARSER.parse(code.encode("utf-8", errors="ignore"))
    except Exception:  # noqa: BLE001
        return []
    out: list[str] = []
    # Walk the syntax tree looking for import_statement / import_from_statement.
    cursor = tree.walk()
    stack = [cursor.node]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        t = node.type
        if t == "import_statement":
            # Children: import <dotted_name>, <dotted_name>, ...
            for child in node.children:
                if child.type in {"dotted_name", "aliased_import"}:
                    first = (
                        child.children[0] if child.children else child
                    )  # handle aliased
                    text = code[first.start_byte : first.end_byte]
                    top = text.split(".")[0].strip()
                    if top:
                        out.append(top)
        elif t == "import_from_statement":
            # First `dotted_name` is the module
            for child in node.children:
                if child.type in {"dotted_name", "relative_import"}:
                    text = code[child.start_byte : child.end_byte]
                    top = text.split(".")[0].strip()
                    if top and not top.startswith("."):
                        out.append(top)
                    break
        stack.extend(node.children)
    return out


def _extract_from_text(text: str) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    # First: try whole response as Python.
    try:
        tree = ast.parse(text)
        return _extract_imports_from_tree(tree)
    except SyntaxError:
        pass
    # Fallback 1: try each fenced code block separately.
    fences = _FENCE_RE.findall(text)
    out: list[str] = []
    any_ok = False
    for block in fences:
        try:
            tree = ast.parse(block)
            any_ok = True
            out.extend(_extract_imports_from_tree(tree))
        except SyntaxError:
            # Fallback 2: tree-sitter on the raw block.
            out.extend(_extract_via_tree_sitter(block))
    if not any_ok and not fences:
        # Fallback 3: tree-sitter on whole response.
        out.extend(_extract_via_tree_sitter(text))
    return out


def extract_imports(code: str, *, include_stdlib: bool = False) -> list[str]:
    """Extract top-level import names from a Python response.

    Parameters
    ----------
    code : str
        The generated response text (may contain markdown fences).
    include_stdlib : bool, default False
        If False, standard-library modules are filtered out (they are never
        hallucinated packages and shouldn't inflate the denominator).

    Returns
    -------
    list[str]
        Deduped list of top-level import names in first-seen order.
    """
    raw = _extract_from_text(code)
    seen: set[str] = set()
    out: list[str] = []
    stdlib = _stdlib_names() if not include_stdlib else frozenset()
    for name in raw:
        if name in seen:
            continue
        seen.add(name)
        if not include_stdlib and name in stdlib:
            continue
        out.append(name)
    return out
