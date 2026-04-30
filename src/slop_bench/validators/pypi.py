"""PyPI local-master-list validator.

Lazy-loads the Spracklen-shipped pypi_package_names.csv (500k names) and
false_positive_packages.csv. Returns a cached singleton. Live pypi.org HEAD
requests live in `slop_bench.spot_check`.
"""
from __future__ import annotations

import csv
import functools
from pathlib import Path

from ..config import MASTER_LIST_PATHS
from ..heuristics import normalize_python


@functools.lru_cache(maxsize=1)
def _load_pypi_names() -> frozenset[str]:
    path = MASTER_LIST_PATHS["python"]["pypi"]
    names: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            names.add(normalize_python(line.strip()))
    names.discard("")
    return frozenset(names)


@functools.lru_cache(maxsize=1)
def _load_py_false_positives() -> frozenset[str]:
    path = MASTER_LIST_PATHS["python"]["false_positives"]
    fps: set[str] = set()
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                fps.add(row[1])
    return frozenset(fps)


class PyPIValidator:
    """Thin wrapper so callers can inject custom paths in tests."""

    def __init__(
        self,
        *,
        names: frozenset[str] | None = None,
        false_positives: frozenset[str] | None = None,
    ) -> None:
        self._names = names if names is not None else _load_pypi_names()
        self._false_positives = (
            false_positives if false_positives is not None else _load_py_false_positives()
        )

    def is_known(self, name_norm: str) -> bool:
        return bool(name_norm) and name_norm in self._names

    def is_false_positive(self, name_raw: str) -> bool:
        return name_raw in self._false_positives

    def classify(self, name_raw: str) -> tuple[str, bool]:
        """Return (normalised_name, is_hallucinated).

        Mirrors Spracklen's `check_packages`: a name is hallucinated iff
        normalised is non-empty AND not in the pypi set AND raw not in
        the false-positive set.
        """
        norm = normalize_python(name_raw)
        if not norm or " " in norm or norm in {"none", "nan"}:
            return norm, False
        if self.is_known(norm):
            return norm, False
        if self.is_false_positive(name_raw):
            return norm, False
        return norm, True

    def __len__(self) -> int:
        return len(self._names)
