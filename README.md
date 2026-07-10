# learning-to-route

**Learn to route coding tasks to the cheapest capable LLM with tiny static embedding models.**

Frontier models cost 5x to 55x more than small models that solve most coding tasks just as well (gpt-5.6-sol $5.00/1M input vs deepseek-v4-flash $0.09/1M spot). A router that sends each task to the cheapest model likely to solve it moves the cost/speed/quality frontier without training a single LLM.

This repo is the open research codebase for the whitepaper [**Learning to Route**](paper/learning-to-route.md). Headline result on the included 27-task coding benchmark (medium-hard exact tasks plus quality-scored optimization tasks): a verify-and-escalate cascade over four cheap models solves **100% of tasks at 26% of the cost of the best single model and 4% of the cost of a frontier model (gpt-5.5)**, which ties a $0.03 model at 77.8% pass. Per solved task: cascade $0.0041 vs frontier $0.1213, a 30x gap. Total research spend: about $4.

![frontier](paper/figs/frontier.png)

- **Routing is embedding search.** A task is embedded with a static embedding model (a token lookup plus mean pool, no transformer forward pass, about 0.15ms on CPU). Its k nearest anchor tasks vote on which model to use, weighted by observed pass rates. The router is literally an ANN index, the same CAGRA-style index you'd use for retrieval.
- **Online learning from coding agents.** Every agent run reports back (task, model, passed, cost, latency). Anchors update via EWMA, so the task-to-model map improves as agents work. No training loop.
- **Researched on cheap models only.** The routing table is built by benchmarking sub-dollar LLMs on a medium-to-hard coding benchmark. Escalation to expensive tiers (gpt-5.6-luna/terra/sol, Claude Fable) is the learned exception, not the default.

## Quickstart

```bash
uv venv && uv pip install -e .
uv run python -m ltr.train --results experiments/results.jsonl --out experiments/router.json
uv run python -m ltr.route "write a rate limiter with sliding window log"
```

Run the benchmark against any OpenAI-compatible endpoint (we use [openpaths](https://openpaths.io)):

```bash
export LTR_BASE_URL=https://openpaths.io/v1 LTR_API_KEY=...
uv run python benchmark/harness.py --models deepseek-v4-flash,gpt-5.4-nano,gpt-5.4-mini,gemini-3.5-flash --out experiments/results.jsonl
uv run python scripts/experiments.py
```

## Static embedders

Any of these produce the router's task vectors. All serve `sentence-transformers/static-retrieval-mrl-en-v1`-family static models (int8, 512-dim, about 16MB):

- [pybed](https://github.com/lee101/pybed): pure Python, FlatIndex plus CAGRA-style graph index, optional CUDA kernels
- [gobed](https://github.com/lee101/gobed): Go, 0.15ms per embedding
- [zbed](https://github.com/lee101/zbed): Zig, SIMD
- [public-static-modern-bert](https://github.com/lee101/public-static-modern-bert): recipe for distilling your own static embedder (e.g. from ModernBERT)
- fallback: `sentence-transformers` static models

Embedding is a memory read and search is a graph walk, so routing overhead is microseconds to milliseconds and runs anywhere: edge, gateway, or inside the agent binary. See the [CAGRA](https://arxiv.org/abs/2308.15136) build-on-GPU serve-on-CPU pattern for large anchor tables.

## Layout

- `ltr/`: router package (embed, route, train, online update, cost/quality frontier)
- `benchmark/`: medium-to-hard coding tasks plus an OpenAI-compatible harness with sandboxed test execution
- `inference/`: routing at inference time from Python (pybed) and Go (gobed)
- `paper/`: the whitepaper and figures
- `experiments/`: results, routing tables, report
- `scripts/`: experiments and reporting

## Related work

RouteLLM, FrugalGPT, Hybrid LLM, RouterBench, Arch-Router, [Supra-Router-51M](https://huggingface.co/SupraLabs/Supra-Router-51M), kNN routers. See the [paper](paper/learning-to-route.md) for the comparison. Short version: simple kNN over good embeddings beats learned routers ([arXiv:2505.12601](https://arxiv.org/abs/2505.12601)); this repo makes the embeddings about 400x cheaper and the table self-updating.

Mirrored at [huggingface.co/openpaths/learning-to-route](https://huggingface.co/openpaths/learning-to-route). Improvements deploy into the production auto-router at [openpaths.io](https://openpaths.io).

MIT license.
