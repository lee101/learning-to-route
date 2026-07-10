import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from ltr.embed import Embedder
from ltr.router import ModelSpec, Router

EXP = ROOT / "experiments"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(exist_ok=True)

MODELS = ["deepseek-v4-flash", "gpt-5.4-nano", "gpt-5.4-mini", "gemini-3.5-flash", "gpt-5.5"]
SPECS = {
    "deepseek-v4-flash": ModelSpec("deepseek-v4-flash", 0.14, 0.28),
    "gpt-5.4-nano": ModelSpec("gpt-5.4-nano", 0.20, 1.25),
    "gpt-5.4-mini": ModelSpec("gpt-5.4-mini", 0.75, 4.50),
    "gemini-3.5-flash": ModelSpec("gemini-3.5-flash", 1.50, 9.00),
    "gpt-5.5": ModelSpec("gpt-5.5", 5.00, 30.00),
}
SHORT = {
    "deepseek-v4-flash": "deepseek\nv4-flash",
    "gpt-5.4-nano": "gpt-5.4\nnano",
    "gpt-5.4-mini": "gpt-5.4\nmini",
    "gemini-3.5-flash": "gemini-3.5\nflash",
    "gpt-5.5": "gpt-5.5",
}
C = {"deepseek-v4-flash": "#2a9d8f", "gpt-5.4-nano": "#457b9d", "gpt-5.4-mini": "#e9c46a",
     "gemini-3.5-flash": "#e76f51", "gpt-5.5": "#6d597a",
     "router": "#9b5de5", "oracle": "#111111", "cascade": "#d90368"}


def sc(r):
    return r.get("score", 1.0 if r["passed"] else 0.0)


def load():
    rows = []
    for m in MODELS:
        p = EXP / f"results.{m}.jsonl"
        rows += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    (EXP / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    by_task = defaultdict(dict)
    for r in rows:
        by_task[r["task_id"]][r["model"]] = r
    by_task = {t: v for t, v in by_task.items() if len(v) == len(MODELS)}
    return rows, by_task


def build_router(rows, backend="pybed", k=8):
    router = Router([SPECS[m] for m in MODELS], embedder=Embedder(backend), k=k)
    for r in rows:
        router.update(r["task"], r["model"], sc(r), r["cost"])
    return router


def loto_route(router, task_text, floor):
    keep = router.anchors
    router.anchors = [a for a in keep if a.text != task_text]
    router._rebuild()
    choice = router.route(task_text, quality_floor=floor)["model"]
    router.anchors = keep
    router._rebuild()
    return choice


def eval_router(router, by_task, floor):
    cost = passed = score = 0.0
    for runs in by_task.values():
        text = next(iter(runs.values()))["task"]
        r = runs[loto_route(router, text, floor)]
        cost += r["cost"]
        passed += r["passed"]
        score += sc(r)
    n = len(by_task)
    return {"pass": passed / n, "score": score / n, "cost": cost}


def eval_cascade(by_task, order, router=None, floor=0.5):
    cost = passed = score = 0.0
    for runs in by_task.values():
        chain = order
        if router is not None:
            text = next(iter(runs.values()))["task"]
            start = loto_route(router, text, floor)
            chain = order[order.index(start):]
        best = 0.0
        for m in chain:
            r = runs[m]
            cost += r["cost"]
            best = max(best, sc(r))
            if r["passed"]:
                passed += 1
                break
        score += best
    n = len(by_task)
    return {"pass": passed / n, "score": score / n, "cost": cost}


def main():
    rows, by_task = load()
    n = len(by_task)
    order = sorted(MODELS, key=lambda m: SPECS[m].cost())
    rep = {"n_tasks": n, "single": {}, "router": {}, "cascade": {},
           "k_ablation": {}, "backend": {}, "oracle": {}}

    for m in MODELS:
        runs = [t[m] for t in by_task.values()]
        rep["single"][m] = {
            "pass": sum(r["passed"] for r in runs) / n,
            "score": sum(sc(r) for r in runs) / n,
            "cost": sum(r["cost"] for r in runs),
            "latency_p50": float(np.median([r["latency"] for r in runs])),
        }

    ocost = opass = oscore = 0.0
    for runs in by_task.values():
        solved = [r for r in runs.values() if r["passed"]]
        if solved:
            ocost += min(r["cost"] for r in solved)
            opass += 1
            oscore += 1.0
        else:
            best = max(runs.values(), key=sc)
            ocost += min(r["cost"] for r in runs.values())
            oscore += sc(best)
    rep["oracle"] = {"pass": opass / n, "score": oscore / n, "cost": ocost}

    router = build_router(rows)
    for f in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        rep["router"][str(f)] = eval_router(router, by_task, f)

    rep["cascade"]["price_order"] = eval_cascade(by_task, order)
    for f in [0.5, 0.6, 0.7]:
        rep["cascade"][f"router_start_{f}"] = eval_cascade(by_task, order, router, f)

    for k in [1, 2, 4, 8, 12, 16]:
        rep["k_ablation"][str(k)] = eval_router(build_router(rows, k=k), by_task, 0.6)
    for backend in ["pybed", "hash"]:
        rep["backend"][backend] = eval_router(build_router(rows, backend), by_task, 0.6)

    (EXP / "report.json").write_text(json.dumps(rep, indent=2))

    plt.rcParams.update({
        "font.size": 11, "figure.dpi": 170, "axes.spines.top": False,
        "axes.spines.right": False, "font.family": "DejaVu Sans",
    })

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    ax.set_xscale("log")
    label_off = {
        "deepseek-v4-flash": (0, 16), "gpt-5.4-nano": (0, -22),
        "gpt-5.4-mini": (0, 14), "gemini-3.5-flash": (0, 14), "gpt-5.5": (0, 14),
    }
    for m in MODELS:
        s = rep["single"][m]
        ax.scatter(s["cost"], s["pass"], s=150, color=C[m], zorder=4,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(m, (s["cost"], s["pass"]), textcoords="offset points",
                    xytext=label_off[m], ha="center", fontsize=9.5, fontweight="bold",
                    color=C[m])
    rc = sorted((v["cost"], v["pass"], f) for f, v in rep["router"].items())
    ax.plot([x for x, _, _ in rc], [y for _, y, _ in rc], "-", color=C["router"],
            lw=2, alpha=0.8, zorder=3)
    ax.scatter([x for x, _, _ in rc], [y for _, y, _ in rc], s=40, color=C["router"],
               zorder=4, label="router alone (floor sweep, LOTO)")
    o = rep["oracle"]
    ax.scatter(o["cost"], o["pass"], marker="*", s=420, color=C["oracle"], zorder=5,
               edgecolors="white", linewidths=1)
    ax.annotate("oracle", (o["cost"], o["pass"]), textcoords="offset points",
                xytext=(0, 15), ha="center", fontsize=9.5, fontweight="bold")
    cc = rep["cascade"]["price_order"]
    ax.scatter(cc["cost"], cc["pass"], marker="D", s=140, color=C["cascade"], zorder=5,
               edgecolors="white", linewidths=1.5)
    ax.annotate("cascade\n(verify + escalate)", (cc["cost"], cc["pass"]),
                textcoords="offset points", xytext=(0, -34), ha="center",
                fontsize=9.5, fontweight="bold", color=C["cascade"])
    rcx = rep["cascade"]["router_start_0.5"]
    ax.scatter(rcx["cost"], rcx["pass"], marker="D", s=140, color=C["router"], zorder=5,
               edgecolors="white", linewidths=1.5, label="router-start cascade")
    ax.set_xlabel("total benchmark cost, USD (log scale)")
    ax.set_ylabel("pass rate")
    ax.set_ylim(top=1.06)
    ax.set_title(f"Cost vs quality: {n} coding tasks, cheap models + one frontier tier",
                 pad=14)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%g"))
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    ax.grid(alpha=0.2, which="both")
    fig.tight_layout()
    fig.savefig(FIGS / "frontier.png", bbox_inches="tight")

    tids = sorted(by_task, key=lambda t: sum(sc(r) for r in by_task[t].values()))
    mat = np.array([[sc(by_task[t][m]) for m in order] for t in tids])
    fig, ax = plt.subplots(figsize=(7.2, 0.36 * len(tids) + 1.6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(order)), [SHORT[m] for m in order], fontsize=9)
    ax.set_yticks(range(len(tids)), tids, fontsize=8.5)
    for i in range(len(tids)):
        for j in range(len(order)):
            v = mat[i, j]
            kind = by_task[tids[i]][order[j]].get("kind", "exact")
            txt = f"{v:.2f}" if kind == "optimize" else ("✓" if v >= 1 else "✗")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if 0.25 < v < 0.9 else ("white" if v <= 0.25 else "#1a4301"))
    ax.set_title("Per-task scores (models ordered by price)", pad=12)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "heatmap.png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    pols = []
    for m in order:
        s = rep["single"][m]
        pols.append((m, s["pass"], s["cost"], C[m]))
    pols.append(("router f=0.6", rep["router"]["0.6"]["pass"], rep["router"]["0.6"]["cost"], C["router"]))
    pols.append(("cascade", cc["pass"], cc["cost"], C["cascade"]))
    pols.append(("router cascade", rcx["pass"], rcx["cost"], C["router"]))
    pols.append(("oracle", o["pass"], o["cost"], C["oracle"]))
    pols = [(name, p, c / max(p * n, 1e-9), col) for name, p, c, col in pols]
    pols.sort(key=lambda x: x[2])
    y = np.arange(len(pols))
    ax.barh(y, [c for _, _, c, _ in pols], color=[col for *_, col in pols], height=0.62)
    ax.set_yticks(y, [f"{name}  ({p:.0%} pass)" for name, p, _, _ in pols], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("cost per solved task, USD")
    ax.set_title("Cost per solved task (label shows pass rate)", pad=12)
    for yi, (_, _, c, _) in zip(y, pols):
        ax.text(c, yi, f"  ${c:.4f}", va="center", fontsize=9)
    ax.grid(alpha=0.2, axis="x")
    fig.tight_layout()
    fig.savefig(FIGS / "cost_per_solve.png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(7, 4))
    ks = sorted(int(k) for k in rep["k_ablation"])
    ax.plot(ks, [rep["k_ablation"][str(k)]["pass"] for k in ks], "o-",
            color=C["router"], lw=2, label="pass rate")
    ax.plot(ks, [rep["k_ablation"][str(k)]["score"] for k in ks], "s--",
            color=C["gpt-5.4-nano"], lw=2, label="mean score")
    ax.set_xlabel("k (neighbours consulted)")
    ax.set_ylabel("quality (LOTO, floor 0.6)")
    ax.set_title("Neighbourhood size ablation", pad=12)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGS / "k_ablation.png", bbox_inches="tight")

    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
