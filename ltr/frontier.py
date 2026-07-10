import argparse
import json
from collections import defaultdict
from pathlib import Path

from .embed import Embedder
from .router import Router


def evaluate(results_path: str, router_path: str | None = None,
             floors: list[float] | None = None, backend: str = "auto") -> dict:
    rows = [json.loads(l) for l in Path(results_path).read_text().splitlines() if l.strip()]
    by_task = defaultdict(dict)
    for r in rows:
        by_task[r["task"]][r["model"]] = r
    out = {"single_model": {}, "routed": {}, "oracle": None}
    models = sorted({r["model"] for r in rows})
    for m in models:
        runs = [t[m] for t in by_task.values() if m in t]
        if not runs:
            continue
        out["single_model"][m] = {
            "pass_rate": sum(r["passed"] for r in runs) / len(runs),
            "total_cost": sum(r.get("cost", 0) for r in runs),
            "n": len(runs),
        }
    oracle_cost, oracle_pass = 0.0, 0
    for t, runs in by_task.items():
        solved = [r for r in runs.values() if r["passed"]]
        if solved:
            best = min(solved, key=lambda r: r.get("cost", 0))
            oracle_cost += best.get("cost", 0)
            oracle_pass += 1
        else:
            oracle_cost += max(r.get("cost", 0) for r in runs.values())
    out["oracle"] = {"pass_rate": oracle_pass / len(by_task), "total_cost": oracle_cost}
    if router_path:
        for floor in floors or [0.5, 0.6, 0.7, 0.8, 0.9]:
            router = Router.load(router_path, embedder=Embedder(backend))
            cost, passed, n = 0.0, 0, 0
            for t, runs in by_task.items():
                router_lo = _clone_without(router, t)
                choice = router_lo.route(t, quality_floor=floor)["model"]
                r = runs.get(choice)
                if r is None:
                    continue
                n += 1
                cost += r.get("cost", 0)
                passed += 1 if r["passed"] else 0
            if n:
                out["routed"][str(floor)] = {
                    "pass_rate": passed / n, "total_cost": cost, "n": n,
                }
    return out


def _clone_without(router: Router, task_text: str) -> Router:
    clone = Router(list(router.order), embedder=router.embedder, k=router.k,
                   quality_floor=router.quality_floor, cost_weight=router.cost_weight)
    clone.anchors = [a for a in router.anchors if a.text != task_text]
    clone._rebuild()
    return clone


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--router")
    ap.add_argument("--backend", default="auto")
    args = ap.parse_args()
    print(json.dumps(evaluate(args.results, args.router, backend=args.backend), indent=2))


if __name__ == "__main__":
    main()
