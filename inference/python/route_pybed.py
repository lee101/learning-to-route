"""Routing as pure embedding search with pybed's CAGRA-style index.

The router table is just (vectors, payloads). Any ANN index works; here the
anchor set is indexed with pybed's CagraIndex so routing a task is one static
embed (a token-row lookup, ~0.15ms CPU) plus one graph walk.

Usage: python route_pybed.py "refactor this async handler" [model_dir]
"""
import json
import sys
from pathlib import Path

import numpy as np
from pybed import CagraIndex, EmbedModel

ROUTER = Path(__file__).parents[2] / "experiments" / "router.json"


def main():
    query = sys.argv[1]
    model_dir = (
        Path(sys.argv[2]) if len(sys.argv) > 2
        else Path(__import__("pybed").__file__).resolve().parents[1] / "model"
    )
    model = EmbedModel.from_dir(model_dir)
    data = json.loads(ROUTER.read_text())
    anchors = data["anchors"]

    embs = np.stack([model.embed_quantized(a["text"])[0] for a in anchors])
    index = CagraIndex(embs, degree=min(16, len(anchors) - 1))

    q, _, qnorm = model.embed_quantized(query)
    hits = index.search(q, qnorm, top_k=min(8, len(anchors)))

    scores: dict[str, tuple[float, float]] = {}
    for h in hits:
        for m, s in anchors[h.doc_idx]["stats"].items():
            w = max(h.score, 0.0) * min(s["n"], 10)
            num, den = scores.get(m, (0.0, 0.0))
            scores[m] = (num + w * s["pass"], den + w)
    ranked = sorted(
        ((m, round(num / den, 4)) for m, (num, den) in scores.items() if den > 0),
        key=lambda x: -x[1],
    )
    print(json.dumps({"query": query, "expected_pass": ranked}, indent=2))


if __name__ == "__main__":
    main()
