from slop_bench.extractors.js_extract import extract_imports as extract_js
from slop_bench.extractors.python_extract import extract_imports as extract_py


# ── Python extractor ─────────────────────────────────────────────────────────
def test_py_simple_imports():
    code = """
import os
import json as j
import numpy.linalg
from pandas import DataFrame
from matplotlib.pyplot import plot
"""
    out = extract_py(code)
    # stdlib filtered
    assert "os" not in out
    assert "json" not in out
    # third-party kept
    assert "numpy" in out
    assert "pandas" in out
    assert "matplotlib" in out


def test_py_dedupes():
    code = "import numpy\nimport numpy.linalg\nfrom numpy import array\n"
    assert extract_py(code) == ["numpy"]


def test_py_relative_imports_skipped():
    code = "from . import foo\nfrom ..bar import baz\n"
    # relative imports have level > 0 and are excluded
    assert extract_py(code) == []


def test_py_fenced_block():
    code = """Here's how to use it:

```python
import requests
r = requests.get("https://example.com")
```

That's the basic idea."""
    out = extract_py(code)
    assert "requests" in out


def test_py_invalid_syntax_falls_back():
    # Code with prose around fences that wouldn't parse as a whole
    code = """
Step 1: install it with `pip install foo`.
```python
import sklearn
```
Step 2: done.
"""
    out = extract_py(code)
    assert "sklearn" in out


def test_py_include_stdlib():
    code = "import os\nimport sys\n"
    out = extract_py(code, include_stdlib=True)
    assert "os" in out
    assert "sys" in out


def test_py_none_response():
    assert extract_py("None") == []
    assert extract_py("") == []


# ── JavaScript extractor ─────────────────────────────────────────────────────
def test_js_require():
    code = "```js\nconst express = require('express');\nconst fs = require('fs');\n```"
    out = extract_js(code)
    assert "express" in out
    assert "fs" not in out  # core module filtered


def test_js_import_from():
    code = """```javascript
import React from 'react';
import { useState } from 'react';
import lodash from 'lodash/fp';
import './local.js';
```"""
    out = extract_js(code)
    assert "react" in out
    assert "lodash" in out     # subpath stripped
    assert "./local.js" not in out


def test_js_scoped_package():
    code = "```js\nimport { x } from '@angular/core';\nimport pkg from '@scope/pkg/sub';\n```"
    out = extract_js(code)
    assert "@angular/core" in out
    assert "@scope/pkg" in out


def test_js_side_effect_import():
    code = "```js\nimport 'normalize.css';\n```"
    out = extract_js(code)
    assert "normalize.css" in out
