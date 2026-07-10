import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from ltr.embed import Embedder
from ltr.router import ModelSpec, Router
from ltr.train import DEFAULT_MODELS

EXP = ROOT / "experiments"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(exist_ok=True)

MODELS = ["deepseek-v4-flash", "gpt-5.4-nano", "gpt-5.4-mini", "gemini-3.5-flash"]
SPECS = {m.id: m for m in DEFAULT_MODELS}
SPECS["gemini-3.5-flash"] = ModelSpec("gemini-3.5-flash", 1.50, 9.00)

C = {"deepseek-v4-flash": "#2a9d8f", "gpt-5.4-nano": "#457b9d", "gpt-5.4-mini": "#e9c46a",
     "gemini-3.5-flash": "#e76f51", "router": "#9b5de5", "oracle": "#222222", "cascade": "#f15bb5"}


def load():
    rows = []
    for m in MODELS:
        p = EXP / f"results.{m}.jsonl"
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    (EXP / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_task = defaultdict(dict)
    for r in rows:
        by_task[r["task_id"]][r["model"]] = r
    return rows, by_task


def build_router(rows, backend="pybed", k=8):
    specs = [SPECS[m] for m in MODELS]
    router = Router(specs, embedder=Embedder(backend), k=k)
    for r in rows:
        router.update(r["task"], r["model"], r["passed"], r["cost"])
    return router


def loto_route(router, task_text, floor):
    keep = router.anchors
    router.anchors = [a for a in keep if a.text != task_text]
    router._rebuild()
    choice = router.route(task_text, quality_floor=floor)["model"]
    router.anchors = keep
    router._rebuild()
    return choice


def eval_policy_router(router, by_task, floor):
    cost = 0.0
    passed = 0
    for tid, runs in by_task.items():
        text = next(iter(runs.values()))["task"]
        m = loto_route(router, text, floor)
        r = runs[m]
        cost += r["cost"]
        passed += r["passed"]
    return passed / len(by_task), cost


def eval_cascade(by_task, order):
    cost = 0.0
    passed = 0
    for runs in by_task.values():
        for m in order:
            r = runs[m]
            cost += r["cost"]
            if r["passed"]:
                passed += 1
                break
    return passed / len(by_task), cost


def eval_router_cascade(router, by_task, floor):
    order = sorted(MODELS, key=lambda m: SPECS[m].cost())
    cost = 0.0
    passed = 0
    for runs in by_task.values():
        text = next(iter(runs.values()))["task"]
        start = loto_route(router, text, floor)
        chain = order[order.index(start):]
        for m in chain:
            r = runs[m]
            cost += r["cost"]
            if r["passed"]:
                passed += 1
                break
    return passed / len(by_task), cost


def main():
    rows, by_task = load()
    n = len(by_task)
    report = {"n_tasks": n, "single": {}, "router": {}, "k_ablation": {},
              "backend": {}, "cascade": {}, "oracle": {}}

    for m in MODELS:
        runs = [t[m] for t in by_task.values()]
        report["single"][m] = {
            "pass": sum(r["passed"] for r in runs) / n,
            "cost": sum(r["cost"] for r in runs),
            "latency_p50": float(np.median([r["latency"] for r in runs])),
        }

    ocost, opass = 0.0, 0
    for runs in by_task.values():
        solved = [r for r in runs.values() if r["passed"]]
        if solved:
            ocost += min(r["cost"] for r in solved)
            opass += 1
        else:
            ocost += min(r["cost"] for r in runs.values())
    report["oracle"] = {"pass": opass / n, "cost": ocost}

    router = build_router(rows, "pybed")
    floors = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for f in floors:
        p, c = eval_policy_router(router, by_task, f)
        report["router"][str(f)] = {"pass": p, "cost": c}

    for k in [1, 2, 4, 8, 12, 16]:
        rk = build_router(rows, "pybed", k=k)
        p, c = eval_policy_router(rk, by_task, 0.6)
        report["k_ablation"][str(k)] = {"pass": p, "cost": c}

    for backend in ["pybed", "hash"]:
        rb = build_router(rows, backend)
        p, c = eval_policy_router(rb, by_task, 0.6)
        report["backend"][backend] = {"pass": p, "cost": c}

    order = sorted(MODELS, key=lambda m: SPECS[m].cost())
    p, c = eval_cascade(by_task, order)
    report["cascade"]["price_order"] = {"pass": p, "cost": c}
    for f in [0.5, 0.6, 0.7]:
        p, c = eval_router_cascade(router, by_task, f)
        report["cascade"][f"router_start_{f}"] = {"pass": p, "cost": c}

    (EXP / "report.json").write_text(json.dumps(report, indent=2))

    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for m in MODELS:
        s = report["single"][m]
        ax.scatter(s["cost"], s["pass"], s=90, color=C[m], zorder=3)
        ax.annotate(m, (s["cost"], s["pass"]), textcoords="offset points",
                    xytext=(8, -4), fontsize=9)
    rc = [(v["cost"], v["pass"]) for v in report["router"].values()]
    rc.sort()
    ax.plot([x for x, _ in rc], [y for _, y in rc], "o-", color=C["router"],
            label="router (LOTO, floor sweep)", zorder=4, markersize=5)
    o = report["oracle"]
    ax.scatter(o["cost"], o["pass"], marker="*", s=260, color=C["oracle"],
               label="oracle (cheapest passing)", zorder=5)
    cc = report["cascade"]["price_order"]
    ax.scatter(cc["cost"], cc["pass"], marker="D", s=80, color=C["cascade"],
               label="cascade (verify + escalate)", zorder=5)
    rcx = report["cascade"]["router_start_0.6"]
    ax.scatter(rcx["cost"], rcx["pass"], marker="D", s=80, color=C["router"],
               label="router-start cascade", zorder=5)
    ax.set_xlabel("total cost over benchmark (USD)")
    ax.set_ylabel("pass rate")
    ax.set_title("Cost/quality frontier: 17 tasks, 4 cheap models")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "frontier.png")

    tids = sorted(by_task)
    mat = np.array([[1 if by_task[t][m]["passed"] else 0 for m in MODELS] for t in tids])
    fig, ax = plt.subplots(figsize=(6, 6.5))
    ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MODELS)), [m.replace("-", "-\n", 1) for m in MODELS], fontsize=8)
    ax.set_yticks(range(len(tids)), tids, fontsize=8)
    for i in range(len(tids)):
        for j in range(len(MODELS)):
            ax.text(j, i, "pass" if mat[i, j] else "fail", ha="center", va="center", fontsize=7)
    ax.set_title("Per-task outcomes (routing signal)")
    fig.tight_layout()
    fig.savefig(FIGS / "heatmap.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    ks = sorted(int(k) for k in report["k_ablation"])
    ax.plot(ks, [report["k_ablation"][str(k)]["pass"] for k in ks], "o-",
            color=C["router"], label="pass rate")
    ax2 = ax.twinx()
    ax2.plot(ks, [report["k_ablation"][str(k)]["cost"] for k in ks], "s--",
             color=C["gemini-3.5-flash"], label="cost")
    ax.set_xlabel("k (neighbors)")
    ax.set_ylabel("pass rate")
    ax2.set_ylabel("total cost (USD)")
    ax.set_title("Neighborhood size ablation (floor 0.6, LOTO)")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "k_ablation.png")

    fig, ax = plt.subplots(figsize=(6.5, 4))
    names, passes, costs = [], [], []
    for m in MODELS:
        names.append(m)
        passes.append(report["single"][m]["pass"])
        costs.append(report["single"][m]["cost"])
    names += ["router f=0.6", "cascade", "router cascade", "oracle"]
    passes += [report["router"]["0.6"]["pass"], report["cascade"]["price_order"]["pass"],
               report["cascade"]["router_start_0.6"]["pass"], report["oracle"]["pass"]]
    costs += [report["router"]["0.6"]["cost"], report["cascade"]["price_order"]["cost"],
              report["cascade"]["router_start_0.6"]["cost"], report["oracle"]["cost"]]
    x = np.arange(len(names))
    ax.bar(x - 0.2, passes, 0.4, color=C["router"], label="pass rate")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, costs, 0.4, color="#adb5bd", label="cost (USD)")
    ax.set_xticks(x, names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("pass rate")
    ax2.set_ylabel("total cost (USD)")
    ax.set_title("Policies compared")
    fig.tight_layout()
    fig.savefig(FIGS / "policies.png")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
