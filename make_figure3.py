"""
Figure 3: Jaccard similarity heatmap of hallucinated package sets.

Shows pairwise overlap between models' unique hallucinations.
High similarity → models share training data biases or
hallucinate from same underlying patterns.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sqlite3
from itertools import combinations

DB = "data/runs.db"
MODELS = ["Haiku 4.5", "Sonnet 4.6", "DeepSeek V3.2", "Gemini 2.5 Pro", "GPT-5.4-mini"]
KEYS = ["haiku-4-5", "sonnet-4-6", "deepseek-v3.2", "gemini-2.5-pro", "gpt-5.4-mini"]

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


def get_halluc_set(cur, model_key: str) -> set:
    cur.execute(f"""
        SELECT DISTINCT h.package_norm
        FROM generations g
        JOIN heuristic_hits h ON h.generation_id = g.id
        JOIN validations v
            ON v.package_norm = h.package_norm
            AND v.registry = (CASE WHEN g.language='python' THEN 'pypi' ELSE 'npm' END)
        WHERE g.model_key = ? AND v.exists_in_master = 0
          AND ({NOISE_FILTER})
    """, (model_key,))
    return set(r[0] for r in cur.fetchall())


# Load all sets
print("Loading hallucinated sets...")
conn = sqlite3.connect(DB)
cur = conn.cursor()
sets = {}
for key in KEYS:
    sets[key] = get_halluc_set(cur, key)
    print(f"  {key}: {len(sets[key])} unique halluc names")
conn.close()

# Build Jaccard matrix
n = len(MODELS)
matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        a = sets[KEYS[i]]
        b = sets[KEYS[j]]
        if i == j:
            matrix[i][j] = 1.0
        elif a or b:
            matrix[i][j] = len(a & b) / len(a | b)

# Style
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
})

# Plot
fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)

# Use viridis-like academic palette
im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=0.4, aspect="equal")

# Annotate cells with values
for i in range(n):
    for j in range(n):
        val = matrix[i][j]
        # Skip diagonal text or use bold
        if i == j:
            text = "—"
            color = "#999"
        else:
            text = f"{val:.3f}"
            # Choose text color for readability
            color = "white" if val > 0.25 else "#1a1a1a"
        ax.text(j, i, text, ha="center", va="center",
                color=color, fontsize=10,
                fontweight="bold" if i != j and val > 0.30 else "normal")

# Ticks
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(MODELS, rotation=30, ha="right", fontsize=10)
ax.set_yticklabels(MODELS, fontsize=10)

# Hide spines
for spine in ax.spines.values():
    spine.set_visible(False)

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Jaccard similarity", fontsize=10)
cbar.ax.tick_params(labelsize=9)

ax.set_title(
    "Pairwise Jaccard similarity of hallucinated package sets",
    fontsize=11, pad=12,
)

plt.tight_layout()
plt.savefig("figure3_jaccard.png", dpi=300, bbox_inches="tight")
plt.savefig("figure3_jaccard.pdf", bbox_inches="tight")
print("Saved: figure3_jaccard.png and figure3_jaccard.pdf")

# Print numbers for caption
print("\nFor reference:")
pairs = list(combinations(range(n), 2))
js = [(MODELS[i], MODELS[j], matrix[i][j]) for i, j in pairs]
js.sort(key=lambda x: x[2], reverse=True)
print("Highest overlap:")
for m1, m2, j in js[:3]:
    print(f"  {m1:18s} ∩ {m2:18s} = {j:.3f}")
print("Lowest overlap:")
for m1, m2, j in js[-3:]:
    print(f"  {m1:18s} ∩ {m2:18s} = {j:.3f}")
