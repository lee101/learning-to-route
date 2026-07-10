import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from ltr.frontier import evaluate
from ltr.train import train

RESULTS = Path(__file__).parents[1] / "experiments" / "results.jsonl"
ROUTER = Path(__file__).parents[1] / "experiments" / "router.json"


def main():
    train(str(RESULTS), str(ROUTER))
    rep = evaluate(str(RESULTS), str(ROUTER), floors=[0.5, 0.6, 0.7, 0.8, 0.9])
    print(json.dumps(rep, indent=2))
    best_pass = max(v["pass_rate"] for v in rep["single_model"].values())
    best_model_cost = max(
        v["total_cost"] for v in rep["single_model"].values()
        if v["pass_rate"] == best_pass
    )
    rows = []
    for m, v in sorted(rep["single_model"].items(), key=lambda x: x[1]["total_cost"]):
        rows.append((m, v["pass_rate"], v["total_cost"]))
    for f, v in rep["routed"].items():
        rows.append((f"Router (t={f}, LOTO)", v["pass_rate"], v["total_cost"]))
    o = rep["oracle"]
    rows.append(("Oracle (cheapest passing)", o["pass_rate"], o["total_cost"]))
    print("\n| Policy | Pass rate | Total cost | Cost vs best model |")
    print("|---|---|---|---|")
    for name, pr, cost in rows:
        rel = f"{cost / best_model_cost * 100:.0f}%" if best_model_cost else "-"
        print(f"| {name} | {pr:.1%} | ${cost:.4f} | {rel} |")


if __name__ == "__main__":
    main()
