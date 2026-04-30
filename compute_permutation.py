"""
Permutation test for pairwise hallucination rate differences.
Complements chi² + Holm: provides distribution-free p-values.

For each pair of models, randomly shuffle the model labels 10,000 times
and count how often the observed delta is exceeded.
"""
import sqlite3
import numpy as np
from itertools import combinations

DB = "data/runs.db"
MODELS = ["haiku-4-5", "deepseek-v3.2", "sonnet-4-6", "gpt-5.4-mini", "gemini-2.5-pro"]
N_PERMUTATIONS = 10000
SEED = 42

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

conn = sqlite3.connect(DB)
cur = conn.cursor()

# Load each mention as a (model, is_halluc) pair
print(f"Loading mentions for permutation test (seed={SEED}, N={N_PERMUTATIONS})...")
data = {}  # model -> array of 0/1
for m in MODELS:
    cur.execute(f"""
        SELECT
            CASE WHEN v.exists_in_master = 0 THEN 1 ELSE 0 END
        FROM generations g
        JOIN heuristic_hits h ON h.generation_id = g.id
        JOIN validations v
            ON v.package_norm = h.package_norm
            AND v.registry = (CASE WHEN g.language='python' THEN 'pypi' ELSE 'npm' END)
        WHERE g.model_key = ?
          AND ({NOISE_FILTER})
    """, (m,))
    data[m] = np.array([r[0] for r in cur.fetchall()], dtype=np.int8)
    print(f"  {m}: {len(data[m])} mentions, {data[m].sum()} halluc ({100*data[m].mean():.2f}%)")

conn.close()

# Permutation test for each pair
print(f"\nRunning {N_PERMUTATIONS} permutations per pair...")
rng = np.random.default_rng(SEED)
pairs = list(combinations(MODELS, 2))
results = []

for m1, m2 in pairs:
    a = data[m1]
    b = data[m2]
    observed_delta = abs(a.mean() - b.mean())

    combined = np.concatenate([a, b])
    n_a = len(a)
    n_total = len(combined)

    count_exceed = 0
    for _ in range(N_PERMUTATIONS):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_delta = abs(perm_a.mean() - perm_b.mean())
        if perm_delta >= observed_delta:
            count_exceed += 1

    p_perm = count_exceed / N_PERMUTATIONS
    results.append((m1, m2, 100 * observed_delta, p_perm))
    print(f"  {m1:18s} vs {m2:18s}  Δ={100*observed_delta:5.2f}pp  p_perm={p_perm:.4f}")

print(f"\n=== Summary ===")
print(f"NUM_PERMUTATIONS = {N_PERMUTATIONS}")
print(f"SEED = {SEED}")
print(f"Pairs with p_perm < 0.05: {sum(1 for _,_,_,p in results if p < 0.05)} / {len(results)}")
print(f"Largest p_perm: {max(p for _,_,_,p in results):.4f}")
