"""Offline pass: for every Python generation in the DB, run ast/tree-sitter
extraction and insert rows into heuristic_hits with heuristic='IMPORTS'.

Idempotent: skips (generation_id, 'IMPORTS', package_raw) that already exist.
Takes no API calls, so no budget impact.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slop_bench.db import connect  # noqa: E402
from slop_bench.extractors.python_extract import extract_imports as extract_py  # noqa: E402
from slop_bench.extractors.js_extract import extract_imports as extract_js  # noqa: E402
from slop_bench.heuristics import normalize_python, normalize_javascript  # noqa: E402


def main() -> None:
    conn = connect()
    cur = conn.cursor()

    rows = cur.execute(
        "SELECT id, language, generated_code FROM generations"
    ).fetchall()

    already = {
        (r["generation_id"], r["package_raw"])
        for r in cur.execute(
            "SELECT generation_id, package_raw FROM heuristic_hits WHERE heuristic='IMPORTS'"
        ).fetchall()
    }

    new_inserts = 0
    total_gens = 0
    for row in rows:
        total_gens += 1
        gid = row["id"]
        code = row["generated_code"] or ""
        lang = row["language"]
        if lang == "python":
            names = extract_py(code)
            normalise = normalize_python
        elif lang == "javascript":
            names = extract_js(code)
            normalise = normalize_javascript
        else:
            continue
        for raw in names:
            if (gid, raw) in already:
                continue
            norm = normalise(raw)
            cur.execute(
                "INSERT INTO heuristic_hits (generation_id, heuristic, package_raw, package_norm) "
                "VALUES (?, 'IMPORTS', ?, ?)",
                (gid, raw, norm),
            )
            new_inserts += 1

    print(f"Processed {total_gens} generations, inserted {new_inserts} new IMPORTS rows.")


if __name__ == "__main__":
    main()
