"""npm local-master-list validator."""
from __future__ import annotations

import csv
import functools
from pathlib import Path

from ..config import MASTER_LIST_PATHS
from ..heuristics import normalize_javascript


@functools.lru_cache(maxsize=1)
def _load_npm_names() -> frozenset[str]:
    path = MASTER_LIST_PATHS["javascript"]["npm"]
    names: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            names.add(line.strip())
    names.discard("")
    return frozenset(names)


@functools.lru_cache(maxsize=1)
def _load_core_modules() -> frozenset[str]:
    path = MASTER_LIST_PATHS["javascript"]["core_modules"]
    with path.open(encoding="utf-8") as f:
        return frozenset(
            tok.strip() for tok in next(csv.reader(f), []) if tok.strip()
        )


@functools.lru_cache(maxsize=1)
def _load_js_false_positives() -> frozenset[str]:
    path = MASTER_LIST_PATHS["javascript"]["false_positives"]
    fps: set[str] = set()
    with path.open(encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                fps.add(row[1])
    return frozenset(fps)


class NpmValidator:
    def __init__(
        self,
        *,
        names: frozenset[str] | None = None,
        core_modules: frozenset[str] | None = None,
        false_positives: frozenset[str] | None = None,
    ) -> None:
        self._names = names if names is not None else _load_npm_names()
        self._core_modules = core_modules if core_modules is not None else _load_core_modules()
        self._false_positives = (
            false_positives if false_positives is not None else _load_js_false_positives()
        )

    def is_known(self, name_norm: str) -> bool:
        return bool(name_norm) and name_norm in self._names

    def is_core(self, name_norm: str) -> bool:
        return name_norm in self._core_modules

    def is_false_positive(self, name_raw: str) -> bool:
        return name_raw in self._false_positives

    def classify(self, name_raw: str) -> tuple[str, bool]:
        norm = normalize_javascript(name_raw)
        if not norm or " " in norm or norm in {"none", "nan"}:
            return norm, False
        if self.is_core(norm):
            return norm, False
        if self.is_known(norm):
            return norm, False
        if self.is_false_positive(name_raw):
            return norm, False
        return norm, True

    def __len__(self) -> int:
        return len(self._names)
