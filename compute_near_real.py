"""
Compute edit-distance between top-10 hallucinated package names per model
and real PyPI package names. Counts how many top-10 are within distance ≤2
(typosquat potential).

Outputs NUM_NEAR_REAL_PY for the paper.
"""
import sqlite3
import csv
from pathlib import Path

DB = "data/runs.db"
PYPI_NAMES_PATH = Path("data/master_lists/pypi_package_names.csv")
NPM_NAMES_PATH = Path("data/master_lists/npm_package_names.csv")
MODELS = ["haiku-4-5", "deepseek-v3.2", "sonnet-4-6", "gpt-5.4-mini", "gemini-2.5-pro"]
TOP_N = 10
MAX_DISTANCE = 2

NOISE_FILTER = """
    h.package_norm NOT LIKE 'package:%'
    AND h.package_norm NOT LIKE 'node:%'
    AND h.package_norm NOT LIKE 'dart:%'
    AND h.package_norm NOT LIKE '@/%'
    AND h.package_norm NOT LIKE 'your-%'
    AND h.package_norm NOT LIKE 'whatever%'
    AND h.package_norm NOT LIKE 'sample%'
    AND h.package_norm NOT LIKE '$%' ESCAPE '\\'
    AND h.package_norm NOT LIKE '%@latest'
    AND h.package_norm NOT LIKE '%.jpg%'
    AND h.package_norm NOT LIKE '%.png%'
    AND h.package_norm != '~'
    AND h.package_norm != ','
    AND h.package_norm != '//'
    AND LENGTH(h.package_norm) <= 50
    AND h.package_norm NOT LIKE '% %'
"""


def levenshtein(s1: str, s2: str, max_dist: int = 3) -> int:
    """Compute Levenshtein distance with early termination above max_dist."""
    if abs(len(s1) - len(s2)) > max_dist:
        return max_dist + 1
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        min_in_row = curr_row[0]
        for j, c2 in enumerate(s2):
            ins = prev_row[j + 1] + 1
            dele = curr_row[j] + 1
            sub = prev_row[j] + (c1 != c2)
            cell = min(ins, dele, sub)
            curr_row.append(cell)
            if cell < min_in_row:
                min_in_row = cell
        if min_in_row > max_dist:
            return max_dist + 1
        prev_row = curr_row
    return prev_row[-1]


def find_nearest(name: str, real_names_by_len: dict, max_dist: int):
    """Find the nearest real package within max_dist using length-bucketing."""
    best = (None, max_dist + 1)
    name_len = len(name)
    for L in range(name_len - max_dist, name_len + max_dist + 1):
        if L < 1:
            continue
        for real in real_names_by_len.get(L, ()):
            d = levenshtein(name, real, max_dist)
            if d < best[1]:
                best = (real, d)
                if d == 0:
                    return best
    return best


def load_names_bucketed(path: Path) -> dict:
    """Load names into a dict keyed by length for fast filtering."""
    print(f"Loading {path.name}...")
    by_len = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            n = line.strip().lower()
            if n:
                by_len.setdefault(len(n), []).append(n)
    total = sum(len(v) for v in by_len.values())
    print(f"  Loaded {total} package names")
    return by_len


def get_top_halluc(cur, model: str, language: str, registry: str) -> list:
    """Get top-N hallucinated package names for a model+language."""
    cur.execute(f"""
        SELECT h.package_norm, COUNT(*) as cnt
        FROM generations g
        JOIN heuristic_hits h ON h.generation_id = g.id
        JOIN validations v
            ON v.package_norm = h.package_norm
            AND v.registry = ?
        WHERE g.model_key = ?
          AND g.language = ?
          AND v.exists_in_master = 0
          AND ({NOISE_FILTER})
        GROUP BY h.package_norm
        ORDER BY cnt DESC
        LIMIT ?
    """, (registry, model, language, TOP_N))
    return cur.fetchall()


print("=" * 70)
print(f"EDIT-DISTANCE ANALYSIS: top-{TOP_N} halluc within distance ≤{MAX_DISTANCE}")
print("=" * 70)

pypi_by_len = load_names_bucketed(PYPI_NAMES_PATH)
npm_by_len = load_names_bucketed(NPM_NAMES_PATH)

conn = sqlite3.connect(DB)
cur = conn.cursor()

print(f"\n{'='*70}")
print("PYTHON top-10 hallucinations — nearest real pypi package")
print("=" * 70)

py_total_near = 0
py_per_model = {}

for model in MODELS:
    top = get_top_halluc(cur, model, "python", "pypi")
    print(f"\n[{model}]")
    near_count = 0
    for halluc, cnt in top:
        nearest, dist = find_nearest(halluc.lower(), pypi_by_len, MAX_DISTANCE)
        is_near = dist <= MAX_DISTANCE
        if is_near:
            near_count += 1
            py_total_near += 1
        marker = "★ NEAR" if is_near else "      "
        print(f"  {marker}  {halluc:30s} ({cnt:3d}×)  → {nearest} (dist={dist})")
    py_per_model[model] = (near_count, len(top))
    print(f"  → {near_count}/{len(top)} within distance ≤{MAX_DISTANCE}")

print(f"\n{'='*70}")
print("JAVASCRIPT top-10 hallucinations — nearest real npm package")
print("=" * 70)

js_total_near = 0
js_per_model = {}

for model in MODELS:
    top = get_top_halluc(cur, model, "javascript", "npm")
    print(f"\n[{model}]")
    near_count = 0
    for halluc, cnt in top:
        nearest, dist = find_nearest(halluc.lower(), npm_by_len, MAX_DISTANCE)
        is_near = dist <= MAX_DISTANCE
        if is_near:
            near_count += 1
            js_total_near += 1
        marker = "★ NEAR" if is_near else "      "
        print(f"  {marker}  {halluc:30s} ({cnt:3d}×)  → {nearest} (dist={dist})")
    js_per_model[model] = (near_count, len(top))
    print(f"  → {near_count}/{len(top)} within distance ≤{MAX_DISTANCE}")

conn.close()

print(f"\n{'='*70}")
print("SUMMARY")
print("=" * 70)
print(f"\nPython per-model (near/top10):")
for m, (near, total) in py_per_model.items():
    print(f"  {m:20s}  {near}/{total}")

print(f"\nJavaScript per-model (near/top10):")
for m, (near, total) in js_per_model.items():
    print(f"  {m:20s}  {near}/{total}")

print(f"\nNUM_NEAR_REAL_PY = {py_total_near}  (total across 5 models, distance ≤{MAX_DISTANCE})")
print(f"NUM_NEAR_REAL_JS = {js_total_near}  (bonus: same for JS)")
