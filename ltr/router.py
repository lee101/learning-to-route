import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embed import Embedder


@dataclass
class ModelSpec:
    id: str
    input_price_per_1m: float
    output_price_per_1m: float
    prior_pass: float = 0.5
    prior_n: float = 1.0

    def cost(self, in_tokens: int = 2000, out_tokens: int = 1000) -> float:
        return (
            in_tokens * self.input_price_per_1m + out_tokens * self.output_price_per_1m
        ) / 1e6


@dataclass
class Anchor:
    text: str
    vec: np.ndarray
    stats: dict = field(default_factory=dict)

    def update(self, model_id: str, passed: bool, cost: float, alpha: float = 0.3):
        s = self.stats.setdefault(model_id, {"n": 0, "pass": 0.5, "cost": cost})
        s["n"] += 1
        a = max(alpha, 1.0 / s["n"])
        s["pass"] = (1 - a) * s["pass"] + a * (1.0 if passed else 0.0)
        s["cost"] = (1 - a) * s["cost"] + a * cost


class Router:
    def __init__(self, models: list[ModelSpec], embedder: Embedder | None = None,
                 k: int = 8, quality_floor: float = 0.6, cost_weight: float = 0.15):
        self.models = {m.id: m for m in models}
        self.order = sorted(models, key=lambda m: m.cost())
        self.embedder = embedder or Embedder()
        self.k = k
        self.quality_floor = quality_floor
        self.cost_weight = cost_weight
        self.anchors: list[Anchor] = []
        self._matrix: np.ndarray | None = None

    def _rebuild(self):
        self._matrix = (
            np.stack([a.vec for a in self.anchors]) if self.anchors else None
        )

    def neighbors(self, text: str, k: int | None = None):
        if self._matrix is None:
            return []
        q = self.embedder.encode([text])[0]
        sims = self._matrix @ q
        k = min(k or self.k, len(self.anchors))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.anchors[i], float(sims[i])) for i in idx]

    def expected_pass(self, text: str) -> dict[str, float]:
        nbrs = self.neighbors(text)
        out = {}
        for m in self.order:
            num = m.prior_pass * m.prior_n
            den = m.prior_n
            for anchor, sim in nbrs:
                s = anchor.stats.get(m.id)
                if s is None:
                    continue
                w = max(sim, 0.0) * min(s["n"], 10)
                num += w * s["pass"]
                den += w
            out[m.id] = num / den
        return out

    def route(self, text: str, quality_floor: float | None = None) -> dict:
        floor = quality_floor if quality_floor is not None else self.quality_floor
        ep = self.expected_pass(text)
        max_cost = max(m.cost() for m in self.order)
        for m in self.order:
            if ep[m.id] >= floor:
                return {"model": m.id, "expected_pass": ep[m.id], "policy": "floor", "all": ep}
        best = max(
            self.order,
            key=lambda m: ep[m.id] - self.cost_weight * m.cost() / max_cost,
        )
        return {"model": best.id, "expected_pass": ep[best.id], "policy": "utility", "all": ep}

    def update(self, text: str, model_id: str, passed: bool, cost: float,
               new_anchor_sim: float = 0.92):
        nbrs = self.neighbors(text, k=1)
        if nbrs and nbrs[0][1] >= new_anchor_sim:
            nbrs[0][0].update(model_id, passed, cost)
            return nbrs[0][0]
        vec = self.embedder.encode([text])[0]
        anchor = Anchor(text=text, vec=vec)
        anchor.update(model_id, passed, cost)
        self.anchors.append(anchor)
        self._rebuild()
        return anchor

    def save(self, path: str | Path):
        data = {
            "models": [
                {"id": m.id, "input_price_per_1m": m.input_price_per_1m,
                 "output_price_per_1m": m.output_price_per_1m,
                 "prior_pass": m.prior_pass, "prior_n": m.prior_n}
                for m in self.order
            ],
            "k": self.k,
            "quality_floor": self.quality_floor,
            "cost_weight": self.cost_weight,
            "anchors": [
                {"text": a.text, "vec": a.vec.tolist(), "stats": a.stats}
                for a in self.anchors
            ],
        }
        Path(path).write_text(json.dumps(data))

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder | None = None) -> "Router":
        data = json.loads(Path(path).read_text())
        r = cls(
            [ModelSpec(**m) for m in data["models"]],
            embedder=embedder,
            k=data.get("k", 8),
            quality_floor=data.get("quality_floor", 0.6),
            cost_weight=data.get("cost_weight", 0.15),
        )
        for a in data["anchors"]:
            r.anchors.append(
                Anchor(text=a["text"], vec=np.asarray(a["vec"], dtype=np.float32),
                       stats=a["stats"])
            )
        r._rebuild()
        return r
