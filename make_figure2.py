"""
Figure 2: Headline hallucination rate bar chart with 95% CIs.

Wilson score interval used for confidence intervals
(standard for proportions, more accurate than normal approximation
for small p or extreme rates).
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import binom
import math

# Final filtered numbers from compute_stats.py
DATA = [
    # (display_name, halluc, total)
    ("Haiku 4.5",       1399, 30303),
    ("Sonnet 4.6",      1875, 34633),
    ("Gemini 2.5 Pro",  1983, 34168),
    ("DeepSeek V3.2",   1537, 26116),
    ("GPT-5.4-mini",    1194, 19589),
]


def wilson_ci(halluc: int, total: int, alpha: float = 0.05) -> tuple[float, float, float]:
    """Wilson score interval. Returns (rate, ci_low, ci_high) as percentages."""
    if total == 0:
        return 0.0, 0.0, 0.0
    z = 1.959963984540054  # 95% CI z-score
    p = halluc / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return p * 100, max(0, (center - margin)) * 100, min(1, (center + margin)) * 100


# Style — clean academic
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.grid": False,
})

# Compute rates and CIs
rows = []
for name, h, t in DATA:
    rate, lo, hi = wilson_ci(h, t)
    rows.append((name, rate, lo, hi))

# Sort ascending by rate (best first at bottom)
rows.sort(key=lambda r: r[1])

names = [r[0] for r in rows]
rates = [r[1] for r in rows]
errs_low = [r[1] - r[2] for r in rows]
errs_high = [r[3] - r[1] for r in rows]
errs = [errs_low, errs_high]

# Plot
fig, ax = plt.subplots(figsize=(7, 3.8), dpi=150)

# Bars
y_pos = list(range(len(names)))
bars = ax.barh(
    y_pos, rates,
    xerr=errs,
    height=0.55,
    color="#4a6b8a",
    edgecolor="#2c4257",
    linewidth=0.8,
    error_kw={"ecolor": "#1a1a1a", "elinewidth": 1.2, "capsize": 4, "capthick": 1.2},
)

# Annotations
for i, (rate, lo, hi) in enumerate([(r[1], r[2], r[3]) for r in rows]):
    ax.text(
        hi + 0.15, i,
        f"{rate:.2f}% [{lo:.2f}–{hi:.2f}]",
        va="center", ha="left",
        fontsize=10, color="#2c2c2c",
    )

# Highlight best
ax.get_children()[0].set_color("#2d8a4f")
ax.get_children()[0].set_edgecolor("#1a5a32")

# Labels
ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.set_xlabel("Hallucination rate (%)", fontsize=11)
ax.set_xlim(0, max(r[3] for r in rows) + 2.0)
ax.tick_params(axis="both", which="both", length=4)

# Light vertical reference lines (Spracklen comparison)
ax.axvline(3.6, color="#999", linestyle=":", linewidth=0.8, alpha=0.7, zorder=0)
ax.text(3.6, len(names) - 0.3, "  Spracklen 2025\n  best (3.6%)",
        fontsize=8, color="#666", va="top")

plt.title(
    "Hallucination rates across 5 frontier 2026 models  (n≈40k prompts/model, 95% Wilson CIs)",
    fontsize=11, pad=12, loc="left",
)

plt.tight_layout()
plt.savefig("figure2_headline.png", dpi=300, bbox_inches="tight")
plt.savefig("figure2_headline.pdf", bbox_inches="tight")
print("Saved: figure2_headline.png and figure2_headline.pdf")

# Print numbers for caption
print("\nFor caption:")
for name, rate, lo, hi in rows:
    print(f"  {name:18s}: {rate:.2f}% [{lo:.2f}, {hi:.2f}]")
