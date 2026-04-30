import pytest

from slop_bench.heuristics import (
    h1_extract_javascript,
    h1_extract_python,
    normalize_javascript,
    normalize_python,
)


# ── Python H1 ───────────────────────────────────────────────────────────────
def test_pip_install_basic():
    code = "pip install requests\nimport requests\n"
    assert h1_extract_python(code) == ["requests"]


def test_pip_install_with_version():
    code = "pip install requests>=2.28.0\n"
    assert h1_extract_python(code) == ["requests>=2.28.0"]


def test_pip_install_multiple_lines():
    code = """
    # install first
    pip install numpy
    pip install pandas==2.0
    import numpy; import pandas
    """
    assert h1_extract_python(code) == ["numpy", "pandas==2.0"]


def test_pip_install_skips_flags():
    code = "pip install --upgrade pip\npip install -r requirements.txt\n"
    # Spracklen's regex matches both, but package_detection filters '-' prefixes;
    # we do that in-function.
    out = h1_extract_python(code)
    assert "--upgrade" not in out
    assert "-r" not in out


def test_pip_install_empty():
    assert h1_extract_python("") == []
    assert h1_extract_python("no install here") == []


# ── Python normalisation ────────────────────────────────────────────────────
def test_normalize_python_strips_version():
    assert normalize_python("requests>=2.28.0") == "requests"
    assert normalize_python("Pandas==2.0") == "pandas"
    assert normalize_python("scikit_learn") == "scikit-learn"
    assert normalize_python("Scikit-Learn.") == "scikit-learn"


def test_normalize_python_garbage():
    # names with forbidden chars are dropped
    assert normalize_python("weird@name") == ""
    assert normalize_python("") == ""
    assert normalize_python("--upgrade") == ""


# ── JavaScript H1 ────────────────────────────────────────────────────────────
def test_js_npm_install_inside_fence():
    md = """
Run the install:
```bash
npm install express
npm install lodash@4.17.0
```
Then:
```javascript
const express = require('express');
```
"""
    out = h1_extract_javascript(md)
    # Faithful Spracklen behaviour: regex [\w\-/@.]+ keeps the version spec.
    # Normalisation does NOT strip versions for JS — this is a known source of
    # false-hallucinated flags for npm and we report it verbatim for fidelity.
    assert "express" in out
    assert "lodash@4.17.0" in out


def test_js_skips_flags():
    md = """```bash
npm install -g typescript
```"""
    out = h1_extract_javascript(md)
    # The regex may match "-g typescript" or "typescript"; we filter local and flags
    # Spracklen's regex group(2) captures [\w\-/@.]+ so will grab "typescript"
    assert "typescript" in out


def test_js_no_fence_returns_empty():
    assert h1_extract_javascript("npm install foo") == []


def test_js_empty_input():
    assert h1_extract_javascript("") == []
    assert h1_extract_javascript(None) == []  # type: ignore[arg-type]


# ── JS normalisation ────────────────────────────────────────────────────────
def test_normalize_js():
    assert normalize_javascript("`lodash`") == "lodash"
    assert normalize_javascript("express ") == "express"
    assert normalize_javascript("") == ""
