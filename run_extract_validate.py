"""Extract packages from all generations + validate against PyPI/npm master lists."""
import sqlite3
from pathlib import Path
from slop_bench.extractors.python_extract import extract_imports as py_extract
from slop_bench.extractors.js_extract import extract_imports as js_extract
from slop_bench.heuristics import h1_extract_python, h1_extract_javascript, normalize_python, normalize_javascript
from slop_bench.validators.pypi import _load_pypi_names, _load_py_false_positives
from slop_bench.validators.npm import _load_npm_names, _load_js_false_positives

DB = Path("data/runs.db")

def main():
    # Load master lists once
    print("Loading master lists...")
    pypi = _load_pypi_names()
    npm = _load_npm_names()
    py_fp = _load_py_false_positives()
    js_fp = _load_js_false_positives()
    # Load core modules
    core_modules = set()
    with open("data/master_lists/core_modules.csv") as f:
        for line in f:
            core_modules.add(line.strip())
    print(f"PyPI: {len(pypi)}, npm: {len(npm)}, core: {len(core_modules)}")

    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    cur = con.cursor()

    # Clear old heuristic_hits to rebuild (keep validations as cache)
    print("Clearing old heuristic_hits...")
    cur.execute("DELETE FROM heuristic_hits")
    con.commit()

    print("Fetching generations...")
    cur.execute("SELECT id, language, generated_code FROM generations")
    rows = cur.fetchall()
    print(f"Processing {len(rows)} generations...")

    hits_batch = []
    validation_cache = {}  # (pkg_norm, registry) -> exists_in_master
    
    for i, (gen_id, lang, code) in enumerate(rows):
        if i % 2000 == 0:
            print(f"  [{i}/{len(rows)}]")
        if not code:
            continue
        
        # Extract packages
        if lang == "python":
            h1_raw = h1_extract_python(code)
            try:
                imports_raw = py_extract(code)
            except Exception:
                imports_raw = []
            all_raw = [(p, "H1") for p in h1_raw] + [(p, "IMPORTS") for p in imports_raw]
            normalize = normalize_python
            master = pypi
            fp = py_fp
            registry = "pypi"
        else:
            h1_raw = h1_extract_javascript(code)
            try:
                imports_raw = js_extract(code)
            except Exception:
                imports_raw = []
            all_raw = [(p, "H1") for p in h1_raw] + [(p, "IMPORTS") for p in imports_raw]
            normalize = normalize_javascript
            master = npm
            fp = js_fp
            registry = "npm"
        
        for raw, heur in all_raw:
            norm = normalize(raw)
            if not norm or norm in fp:
                continue
            if lang == "python" and norm in core_modules:
                continue
            hits_batch.append((gen_id, heur, raw, norm))
            # Cache validation
            key = (norm, registry)
            if key not in validation_cache:
                validation_cache[key] = 1 if norm in master else 0

    print(f"Inserting {len(hits_batch)} hits...")
    cur.executemany(
        "INSERT INTO heuristic_hits (generation_id, heuristic, package_raw, package_norm) VALUES (?,?,?,?)",
        hits_batch
    )
    
    print(f"Inserting {len(validation_cache)} validations...")
    cur.executemany(
        "INSERT OR IGNORE INTO validations (package_norm, registry, exists_in_master) VALUES (?,?,?)",
        [(k[0], k[1], v) for k, v in validation_cache.items()]
    )
    con.commit()
    con.close()
    print("Done!")

if __name__ == "__main__":
    main()
