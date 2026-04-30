"""Preliminary H1 + IMPORTS hallucination rates per model.

Reports three metrics per (model, language):
  * H1 only: pip/npm install regex (Spracklen's baseline)
  * IMPORTS only: ast/tree-sitter static import extraction
  * Union H1 ∪ IMPORTS: faithful analogue to Spracklen's H1+H2+H3 union

Package-level rate = #hallucinated / #packages
Generation-level rate = #gens_with_any_hallucinated_pkg / #gens
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from slop_bench.db import connect  # noqa: E402
from slop_bench.validators.npm import NpmValidator  # noqa: E402
from slop_bench.validators.pypi import PyPIValidator  # noqa: E402


def compute_rates(language: str) -> None:
    conn = connect()
    if language == "python":
        validator = PyPIValidator()
    else:
        validator = NpmValidator()

    rows = conn.execute(
        """SELECT g.id AS gen_id, g.model_key, h.heuristic, h.package_raw, h.package_norm
           FROM generations g
           LEFT JOIN heuristic_hits h ON h.generation_id = g.id
           WHERE g.language = ?""",
        (language,),
    ).fetchall()

    # Group by (model_key, heuristic) and accumulate package-level stats
    per_model: dict[str, dict] = defaultdict(
        lambda: {
            "gens": set(),
            "pkgs_H1": [],
            "pkgs_IMPORTS": [],
            "halluc_H1": [],
            "halluc_IMPORTS": [],
            "gens_halluc_H1": set(),
            "gens_halluc_IMPORTS": set(),
            "gens_halluc_any": set(),
        }
    )

    for r in rows:
        m = r["model_key"]
        per_model[m]["gens"].add(r["gen_id"])
        if r["heuristic"] is None:
            continue
        raw = r["package_raw"]
        _, is_halluc = validator.classify(raw)
        if r["heuristic"] == "H1":
            per_model[m]["pkgs_H1"].append(raw)
            if is_halluc:
                per_model[m]["halluc_H1"].append(raw)
                per_model[m]["gens_halluc_H1"].add(r["gen_id"])
                per_model[m]["gens_halluc_any"].add(r["gen_id"])
        elif r["heuristic"] == "IMPORTS":
            per_model[m]["pkgs_IMPORTS"].append(raw)
            if is_halluc:
                per_model[m]["halluc_IMPORTS"].append(raw)
                per_model[m]["gens_halluc_IMPORTS"].add(r["gen_id"])
                per_model[m]["gens_halluc_any"].add(r["gen_id"])

    # Header
    print(f"\n=== {language.upper()} — preliminary rates ===\n")
    cols = [
        "Model",
        "Gens",
        "H1#",
        "H1halluc",
        "H1%",
        "IMP#",
        "IMPhalluc",
        "IMP%",
        "GenHallucAny%",
    ]
    widths = [16, 6, 6, 9, 6, 6, 10, 6, 14]
    print(" ".join(f"{c:<{w}}" for c, w in zip(cols, widths)))
    print("-" * (sum(widths) + len(widths)))

    for model_key in sorted(per_model):
        s = per_model[model_key]
        n_gens = len(s["gens"])
        h1_tot = len(s["pkgs_H1"])
        h1_hal = len(s["halluc_H1"])
        imp_tot = len(s["pkgs_IMPORTS"])
        imp_hal = len(s["halluc_IMPORTS"])
        h1_rate = 100 * h1_hal / h1_tot if h1_tot else float("nan")
        imp_rate = 100 * imp_hal / imp_tot if imp_tot else float("nan")
        any_rate = 100 * len(s["gens_halluc_any"]) / n_gens if n_gens else 0.0

        def fmt(x, decimals=2):
            if isinstance(x, float) and x != x:
                return "n/a"
            return f"{x:.{decimals}f}" if isinstance(x, float) else str(x)

        row = [
            model_key,
            n_gens,
            h1_tot,
            h1_hal,
            fmt(h1_rate),
            imp_tot,
            imp_hal,
            fmt(imp_rate),
            fmt(any_rate),
        ]
        print(" ".join(f"{str(v):<{w}}" for v, w in zip(row, widths)))

    # Dump the hallucinated names for inspection
    print("\n=== hallucinated package names per model (IMPORTS) ===")
    for model_key in sorted(per_model):
        halluc = per_model[model_key]["halluc_IMPORTS"]
        if halluc:
            from collections import Counter

            c = Counter(halluc)
            print(f"\n{model_key}:")
            for name, n in c.most_common(10):
                print(f"  {name!r}  ×{n}")


def main() -> None:
    compute_rates("python")
    compute_rates("javascript")


if __name__ == "__main__":
    main()
