# learning-to-route

**Learn to route coding tasks to the cheapest capable LLM with tiny static embedding models.**

Frontier models are 10–55x more expensive than small models that solve most coding tasks just as well (gpt-5.6-sol $5/1M in vs deepseek-v4-flash $0.09/1M in). A router that sends each task to the cheapest model likely to solve it pushes the cost/speed/quality frontier without training a single LLM.

This repo is the open research codebase for the whitepaper [**Learning to Route**](paper/learning-to-route.md):

- **Routing = embedding search.** A task is embedded with a static embedding model (a token lookup + mean-pool, no transformer forward pass, ~0.1–0.3ms). Its k nearest anchor tasks vote on which model to use, weighted by observed pass rates and cost. The router *is* an ANN index — the same CAGRA-style index you'd use for retrieval.
- **Online learning from coding agents.** Every agent run reports back (task, model, passed, cost, latency). Anchors update via EWMA, so the task→model map improves as agents work. No gradient training loop needed in the inner loop; the embedder can be periodically re-distilled.
- **Trained on cheap models only.** The routing table is built by benchmarking small/cheap LLMs (deepseek-v4-flash, gpt-5.4-nano/mini, gpt-5.6-luna at low reasoning) on a medium→hard coding benchmark. Escalation to expensive tiers is the learned exception, not the default.

## Quickstart

```bash
uv venv && uv pip install -e .
uv run python -m ltr.train --results experiments/results.jsonl --out experiments/router.json
uv run python -m ltr.route "write a rate limiter with sliding window log"
```

Run the benchmark against any OpenAI-compatible endpoint (we use [openpaths](https://openpaths.co)):

```bash
export LTR_BASE_URL=https://api.openpaths.co/v1 LTR_API_KEY=...
uv run python benchmark/harness.py --models deepseek-v4-flash,gpt-5.4-nano,gpt-5.4-mini --out experiments/results.jsonl
```

## Static embedders

Any of these produce the router's task vectors — all use `sentence-transformers/static-retrieval-mrl-en-v1`-family static models (int8, 512-dim, ~16MB):

- [pybed](https://github.com/lee101/pybed) — pure Python, FlatIndex + CAGRA-style graph index, optional CUDA kernels
- [gobed](https://github.com/lee101/gobed) — Go, 0.15ms/embedding
- [zbed](https://github.com/lee101/zbed) — Zig, SIMD
- [public-static-modern-bert](https://github.com/lee101/public-static-modern-bert) — recipe for distilling your own static embedder (e.g. from ModernBERT)
- fallback: `sentence-transformers` static models

Because embedding is a memory read and search is a graph walk, routing overhead is microseconds-to-milliseconds and runs anywhere (edge, gateway, agent-local) — see the [CAGRA](https://arxiv.org/abs/2308.15136) build-on-GPU/serve-on-CPU pattern.

## Layout

- `ltr/` — router package: embed, route, train, online update, cost/quality frontier
- `benchmark/` — medium→hard coding tasks + OpenAI-compatible harness with sandboxed test execution
- `inference/` — routing at inference time from Python (pybed) and Go (gobed)
- `paper/` — the whitepaper
- `experiments/` — results, routing tables, frontier plots

## Related work

RouteLLM, FrugalGPT, Hybrid LLM, RouterBench, Arch-Router, [Supra-Router-51M](https://huggingface.co/SupraLabs/Supra-Router-51M), kNN routers — see the [paper](paper/learning-to-route.md) for the full comparison. Short version: simple kNN over good embeddings beats learned routers ([arXiv:2505.12601](https://arxiv.org/abs/2505.12601)); we make the embeddings ~400x cheaper and the table self-updating.

MIT license.
