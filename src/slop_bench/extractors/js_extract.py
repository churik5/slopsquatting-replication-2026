"""JavaScript package extractor via tree-sitter-javascript.

Extracts package specifiers from `require(...)` and `import ... from '...'`
statements. Filters out relative paths (./, ../) and Node core modules.

TODO: currently a basic implementation; will be expanded in P0.2 for JS parity.
"""
from __future__ import annotations

import re

try:
    import tree_sitter_javascript as tsjs
    from tree_sitter import Language, Parser

    _TS_LANG = Language(tsjs.language())
    _TS_PARSER: Parser | None = Parser(_TS_LANG)
except Exception:  # noqa: BLE001
    _TS_PARSER = None


_FENCE_RE = re.compile(r"```(?:javascript|js|typescript|ts|jsx|tsx|)\s*\n?([\s\S]+?)```", re.IGNORECASE)
# Regex fallback (Spracklen-style) — more robust than only tree-sitter.
_REQUIRE_RE = re.compile(r"""\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)""")
_IMPORT_FROM_RE = re.compile(r"""\bfrom\s+['"]([^'"]+)['"]""")
_SIDE_EFFECT_IMPORT_RE = re.compile(r"""\bimport\s+['"]([^'"]+)['"]""")


def _normalize_specifier(spec: str) -> str | None:
    """Map a JS import specifier like 'lodash/fp' or '@scope/pkg/sub' to the
    package name that npm actually ships ('lodash' or '@scope/pkg').

    Returns None for relative/absolute paths and URLs.
    """
    spec = spec.strip()
    if not spec:
        return None
    if spec.startswith((".", "/")) or spec.startswith(("http://", "https://")):
        return None
    if spec.startswith("@"):
        # Scoped package: @scope/name[/subpath]
        parts = spec.split("/")
        if len(parts) >= 2:
            return "/".join(parts[:2])
        return None
    return spec.split("/")[0]


def _extract_with_regex(block: str) -> list[str]:
    specs: list[str] = []
    specs.extend(_REQUIRE_RE.findall(block))
    specs.extend(_IMPORT_FROM_RE.findall(block))
    specs.extend(_SIDE_EFFECT_IMPORT_RE.findall(block))
    out: list[str] = []
    for s in specs:
        name = _normalize_specifier(s)
        if name:
            out.append(name)
    return out


def extract_imports(code: str, *, include_core: bool = False) -> list[str]:
    if not code or not isinstance(code, str):
        return []
    fences = _FENCE_RE.findall(code)
    if not fences:
        # For plain text responses, still try regex on the whole thing
        blocks = [code]
    else:
        blocks = fences
    raw: list[str] = []
    for block in blocks:
        raw.extend(_extract_with_regex(block))
    # Dedupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for name in raw:
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    if not include_core:
        try:
            from ..validators.npm import _load_core_modules  # noqa: PLC0415

            core = _load_core_modules()
            out = [n for n in out if n not in core]
        except Exception:  # noqa: BLE001
            pass
    return out
