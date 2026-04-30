"""Budget enforcement — hard cap across providers.

Usage
-----
>>> b = Budget.load()
>>> b.charge("openai", input_tokens=400, output_tokens=600,
...          input_price=0.75, output_price=4.50, batch=False)
>>> b.remaining
159.xx
>>> b.save()   # atomic write to data/spend.json

Idempotence: every call to `charge()` must be paired with a DB row that records
the run, model, prompt_id. On restart we reconstruct cumulative spend from the DB,
not from the JSON — the JSON is a cache for human-readable status only.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .config import BUDGET_CAP, BUDGET_WARN, SPEND_JSON


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    cap: float = BUDGET_CAP
    warn_at: float = BUDGET_WARN
    per_provider: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    path: Path = SPEND_JSON

    # ------------------------------------------------------------------
    @property
    def total(self) -> float:
        return sum(self.per_provider.values())

    @property
    def remaining(self) -> float:
        return self.cap - self.total

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path = SPEND_JSON) -> "Budget":
        b = cls(path=path)
        if path.exists():
            data = json.loads(path.read_text())
            b.per_provider = dict(data.get("per_provider", {}))
        return b

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "total_usd": round(self.total, 6),
                    "remaining_usd": round(self.remaining, 6),
                    "cap_usd": self.cap,
                    "per_provider": {k: round(v, 6) for k, v in self.per_provider.items()},
                    "updated_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        os.replace(tmp, self.path)

    # ------------------------------------------------------------------
    def project(
        self,
        *,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        input_price: float,
        output_price: float,
        batch: bool = False,
        discount: float = 0.0,
    ) -> float:
        factor = (1 - discount) if batch else 1.0
        cost = (input_tokens / 1_000_000 * input_price + output_tokens / 1_000_000 * output_price) * factor
        return cost

    def charge(self, *, provider: str, cost_usd: float) -> None:
        with self._lock:
            new_total = self.total + cost_usd
            if new_total > self.cap:
                raise BudgetExceeded(
                    f"Budget cap ${self.cap:.2f} would be exceeded by a "
                    f"${cost_usd:.4f} charge (would reach ${new_total:.2f}). "
                    f"Aborting."
                )
            self.per_provider[provider] = self.per_provider.get(provider, 0.0) + cost_usd
            if self.total > self.warn_at:
                # Caller is expected to check warn flag + pause; we just print.
                print(
                    f"[BUDGET WARNING] ${self.total:.2f} of ${self.cap:.2f} spent "
                    f"(${self.remaining:.2f} left)."
                )
            self.save()
