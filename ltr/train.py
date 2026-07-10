import argparse
import json
from pathlib import Path

from .embed import Embedder
from .router import ModelSpec, Router

DEFAULT_MODELS = [
    ModelSpec("deepseek-v4-flash", 0.14, 0.28),
    ModelSpec("gpt-5.4-nano", 0.20, 1.25),
    ModelSpec("gpt-5.4-mini", 0.75, 4.50),
    ModelSpec("gpt-5.6-luna", 1.00, 6.00),
    ModelSpec("gpt-5.6-terra", 2.50, 15.00),
    ModelSpec("gpt-5.6-sol", 5.00, 30.00),
]


def train(results_path: str, out_path: str, models: list[ModelSpec] | None = None,
          backend: str = "auto") -> Router:
    rows = [json.loads(l) for l in Path(results_path).read_text().splitlines() if l.strip()]
    seen = sorted({r["model"] for r in rows})
    if models is None:
        by_id = {m.id: m for m in DEFAULT_MODELS}
        models = [by_id.get(m, ModelSpec(m, 1.0, 4.0)) for m in seen]
    router = Router(models, embedder=Embedder(backend))
    for r in rows:
        router.update(r["task"], r["model"], r.get("score", float(r["passed"])), r.get("cost", 0.0))
    router.save(out_path)
    return router


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()
    router = train(args.results, args.out, backend=args.backend)
    print(f"trained router: {len(router.anchors)} anchors, "
          f"{len(router.models)} models, backend={router.embedder.backend} -> {args.out}")


if __name__ == "__main__":
    main()
