"""H1 heuristics (pip/npm install regex) + Spracklen normalisation.

H1 extractors return the list of raw package names as they appear in the
generated code. Normalisation is a separate step (so we can log raw + norm to
the DB and keep a clean audit trail).
"""
from __future__ import annotations

import re

# ── H1: Python pip install ───────────────────────────────────────────────────
# Spracklen applies this regex to the FULL response (package_detection.parse_pip_install),
# not just inside fenced code blocks.
PIP_INSTALL_RE = re.compile(r"pip\s+install\s+(?P<package>\S+)")

# Strip trailing version specifier / bracket extras before membership check.
_VERSION_SPEC_RE = re.compile(r"([^=<>!~]+)([=<>!~]{1,2}[\d.]+)?")
_PUNCT_STRIP = str.maketrans("", "", "()[]`")


def h1_extract_python(code: str) -> list[str]:
    """Return raw package tokens extracted from `pip install ...` lines."""
    if not code:
        return []
    matches = PIP_INSTALL_RE.findall(code)
    out: list[str] = []
    for raw in matches:
        if raw.startswith("-"):
            continue
        out.append(raw)
    return out


# ── H1: JavaScript npm install / yarn add / pnpm add ────────────────────────
# Spracklen only looks inside fenced ```...``` blocks for JS.
_FENCE_RE = re.compile(r"```([\s\S]+?)```")
# Faithful port of Spracklen's `custom_parse_javascript.extract_npm_install`.
_NPM_INSTALL_RE = re.compile(
    r"npm install\s+((?!@)(?:-g|--save-dev|-D)?[\s]+)?([\w\-/@.]+)",
    re.IGNORECASE,
)


def h1_extract_javascript(code: str) -> list[str]:
    if not code or not isinstance(code, str):
        return []
    fences = _FENCE_RE.findall(code)
    if not fences:
        return []
    out: list[str] = []
    for sample in fences:
        for _flag, name in _NPM_INSTALL_RE.findall(sample):
            if not name:
                continue
            if name.startswith("./") or name.startswith("-"):
                continue
            out.append(name)
    return out


def h1_extract(code: str, language: str) -> list[str]:
    lang = language.lower()
    if lang in {"python", "py"}:
        return h1_extract_python(code)
    if lang in {"javascript", "js"}:
        return h1_extract_javascript(code)
    raise ValueError(f"unsupported language: {language!r}")


# ── Normalisation (mirrors Spracklen's package_detection.py) ─────────────────
def normalize_python(name: str) -> str:
    """Strip extras/versions, collapse -_. → -, lowercase.

    Mirrors Spracklen's `normalize_pip` + `normalize_python` combo so the
    result is comparable against the master-list set.
    """
    if not name:
        return ""
    name = name.translate(_PUNCT_STRIP)
    if re.search(r"[+@:\"',{}/\*]", name):
        return ""
    # Handle version-spec like "package>=1.2.3"
    m = _VERSION_SPEC_RE.match(name)
    if m:
        name = m.group(1).strip()
    if name.startswith("--") or "requirements" in name:
        return ""
    name = re.sub(r"\d\. ", "", name)
    name = re.sub(r"[-_.]+", "-", name).strip(' "`.-').lower()
    return name


def normalize_javascript(name: str) -> str:
    if not name:
        return ""
    return re.sub(r"[`]", "", name).strip()


def normalize(name: str, language: str) -> str:
    lang = language.lower()
    if lang in {"python", "py"}:
        return normalize_python(name)
    if lang in {"javascript", "js"}:
        return normalize_javascript(name)
    raise ValueError(f"unsupported language: {language!r}")
