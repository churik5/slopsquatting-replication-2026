"""
Compute pairwise chi-square + Holm correction + Jaccard similarity.

Filtered for known noise patterns:
- path aliases (@/components, etc)
- placeholders (your-module, whatever-pkg, sample-*)
- Dart syntax (package:flutter, dart:io)
- Node.js builtins prefixed with node: (node:fs, node:path)
- template variables ($var)
- asset paths (.jpg, .png)
- has-tag suffixes (@latest)
- single symbols (~, ,)
- multi-word strings
- strings longer than 50 chars
"""
import sqlite3
from itertools import combinations
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

DB = "data/runs.db"
MODELS = ["haiku-4-5", "deepseek-v3.2", "sonnet-4-6", "gpt-5.4-mini", "gemini-2.5-pro"]

# Reusable noise filter — applied wherever we count hallucinations.
# Keeps only "looks like a real package name" strings.
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

# ============================================================
# 1) Per-model counts (for chi² test)
# ============================================================
print("=" * 60)
print("PER-MODEL HALLUCINATION RATES (filtered)")
print("=" * 60)

counts = {}
for m in MODELS:
    cur.execute(f"""
        SELECT
            COUNT(h.id),
            SUM(CASE WHEN v.exists_in_master=0 THEN 1 ELSE 0 END)
        FROM generations g
        LEFT JOIN heuristic_hits h ON h.generation_id = g.id
        LEFT JOIN validations v
            ON v.package_norm = h.package_norm
            AND v.registry = (CASE WHEN g.language='python' THEN 'pypi' ELSE 'npm' END)
        WHERE g.model_key = ?
          AND (h.id IS NULL OR ({NOISE_FILTER}))
    """, (m,))
    total, halluc = cur.fetchone()
    counts[m] = (halluc, total - halluc)
    pct = 100 * halluc / total if total else 0
    print(f"{m:18s}: {halluc:5d} halluc / {total:6d} mentions = {pct:.2f}%")

# ============================================================
# 2) Pairwise chi-square + Holm correction
# ============================================================
print()
print("=" * 60)
print("PAIRWISE CHI-SQUARE TEST (Holm-corrected)")
print("=" * 60)

pairs = list(combinations(MODELS, 2))
pvals = []
chi2s = []
deltas = []

for m1, m2 in pairs:
    h1, n1 = counts[m1]
    h2, n2 = counts[m2]
    table = [[h1, n1], [h2, n2]]
    chi2, p, _, _ = chi2_contingency(table)
    pct1 = 100 * h1 / (h1 + n1)
    pct2 = 100 * h2 / (h2 + n2)
    delta = abs(pct1 - pct2)
    pvals.append(p)
    chi2s.append(chi2)
    deltas.append(delta)

reject, p_adj, _, _ = multipletests(pvals, alpha=0.05, method='holm')

results = []
for (m1, m2), p, p_h, chi2, d, sig in zip(pairs, pvals, p_adj, chi2s, deltas, reject):
    results.append({
        'pair': f"{m1} vs {m2}",
        'm1': m1, 'm2': m2,
        'delta': d, 'chi2': chi2,
        'p_raw': p, 'p_holm': p_h, 'sig': sig
    })
    flag = '***' if sig else 'ns'
    print(f"{m1:18s} vs {m2:18s}  Δ={d:5.2f}pp  χ²={chi2:7.2f}  p_raw={p:.2e}  p_holm={p_h:.2e}  {flag}")

num_sig = sum(reject)
print(f"\nNUM_SIG_PAIRS = {num_sig} / {len(pairs)}")

sig_results = [r for r in results if r['sig']]
ns_results = [r for r in results if not r['sig']]

if sig_results:
    largest = max(sig_results, key=lambda x: x['delta'])
    print(f"LARGEST_PAIR = {largest['pair']}")
    print(f"LARGEST_DELTA = {largest['delta']:.2f}pp")
    print(f"CHI2 = {largest['chi2']:.2f}")
    print(f"PHOLM = {largest['p_holm']:.2e}")

    max_sig_p = max(sig_results, key=lambda x: x['p_holm'])
    print(f"PVALUE (largest among sig) = {max_sig_p['p_holm']:.2e}")

if ns_results:
    closest_ns = min(ns_results, key=lambda x: x['delta'])
    print(f"NS_PAIR = {closest_ns['pair']} (Δ={closest_ns['delta']:.2f}pp)")

# ============================================================
# 3) Jaccard similarity of hallucinated package sets
# ============================================================
print()
print("=" * 60)
print("JACCARD SIMILARITY (filtered hallucinated package sets)")
print("=" * 60)

jaccards = []
for m1, m2 in pairs:
    cur.execute(f"""
        SELECT DISTINCT h.package_norm FROM generations g
        JOIN heuristic_hits h ON h.generation_id = g.id
        JOIN validations v
            ON v.package_norm = h.package_norm
            AND v.registry = (CASE WHEN g.language='python' THEN 'pypi' ELSE 'npm' END)
        WHERE g.model_key = ? AND v.exists_in_master = 0
          AND ({NOISE_FILTER})
    """, (m1,))
    set1 = set(r[0] for r in cur.fetchall())

    cur.execute(f"""
        SELECT DISTINCT h.package_norm FROM generations g
        JOIN heuristic_hits h ON h.generation_id = g.id
        JOIN validations v
            ON v.package_norm = h.package_norm
            AND v.registry = (CASE WHEN g.language='python' THEN 'pypi' ELSE 'npm' END)
        WHERE g.model_key = ? AND v.exists_in_master = 0
          AND ({NOISE_FILTER})
    """, (m2,))
    set2 = set(r[0] for r in cur.fetchall())

    j = len(set1 & set2) / len(set1 | set2) if (set1 | set2) else 0
    jaccards.append(j)
    print(f"{m1:18s} ∩ {m2:18s}  J={j:.3f}  (|∩|={len(set1 & set2):4d}, |∪|={len(set1 | set2):5d})")

print(f"\nMEAN_JACCARD = {sum(jaccards)/len(jaccards):.3f}")
print(f"JACCARD_LO   = {min(jaccards):.3f}")
print(f"JACCARD_HI   = {max(jaccards):.3f}")

# ============================================================
# 4) Universal hallucinations — packages all 5 models invented
# ============================================================
print()
print("=" * 60)
print("UNIVERSAL HALLUCINATIONS (invented by ALL 5 models)")
print("=" * 60)

cur.execute(f"""
    WITH model_halluc AS (
        SELECT DISTINCT g.model_key, h.package_norm, g.language
        FROM generations g
        JOIN heuristic_hits h ON h.generation_id = g.id
        JOIN validations v
            ON v.package_norm = h.package_norm
            AND v.registry = (CASE WHEN g.language='python' THEN 'pypi' ELSE 'npm' END)
        WHERE v.exists_in_master = 0
          AND ({NOISE_FILTER})
    )
    SELECT package_norm, language, COUNT(DISTINCT model_key) as model_count
    FROM model_halluc
    GROUP BY package_norm, language
    HAVING model_count = 5
    ORDER BY package_norm
    LIMIT 30;
""")

universals = cur.fetchall()
print(f"Total packages hallucinated by all 5 models (top 30 shown): {len(universals)}+")
for pkg, lang, cnt in universals:
    print(f"  [{lang:10s}] {pkg}")

conn.close()
