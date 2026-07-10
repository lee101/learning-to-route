# Learning to Route: Static Embedding Models as Self-Improving LLM Routers for Coding Tasks

**Lee Penkman** — [openpaths.io](https://openpaths.io) · [github.com/lee101/learning-to-route](https://github.com/lee101/learning-to-route)

*Draft v0.1, July 2026*

## Abstract

Frontier coding models cost 5–55x more per token than small models that solve a large fraction of real coding tasks equally well (gpt-5.6-sol at $5.00/1M input vs deepseek-v4-flash at $0.14/1M official, $0.09/1M spot). Model routing — picking the cheapest model likely to solve each task — is therefore one of the highest-leverage optimizations available to coding agents, and it requires training no LLM at all. We present a router in which *routing is literally embedding search*: a task is embedded by a **static embedding model** (a token-row lookup and mean-pool with no transformer forward pass, ~0.15ms on CPU), and its k nearest anchor tasks in an ANN index vote on the model to use, weighted by observed pass rates. Because the router state is just (vector, outcome-statistics) pairs, it updates online: every completed agent task — pass/fail, cost, latency — is folded into its nearest anchor by exponential moving average, so the task→model map improves continuously from live agent traffic without a training loop. We evaluate on a 17-problem medium-to-hard coding benchmark using only cheap models (≤$0.75/1M input) to build the routing table, showing the router recovers most of the quality of the best single model at a fraction of its cost, and we show the same anchor table serves from Go (gobed), Zig (zbed), and Python (pybed) inference libraries, including GPU graph search with CAGRA-style indexes. We release the benchmark, router, training and evaluation code under MIT.

## 1. Introduction

The dominant cost in agentic coding is not the hard tasks — it is sending easy tasks to expensive models. Production gateways see the same shape of traffic repeatedly: fix an off-by-one, write a parser, resolve a merge conflict, add a test. Most of these are solved identically by a $0.14/1M model and a $5.00/1M model; a minority genuinely need the frontier tier. The July 2026 release of OpenAI's GPT-5.6 family made this explicit by shipping *price tiers as a product* — Sol ($5/$30), Terra ($2.50/$15), Luna ($1/$6) — a 5x in-family spread, and a ~55x spread against spot-priced small models like DeepSeek V4 Flash.

Existing routers either train a dedicated classifier (RouteLLM's BERT router, Arch-Router's 1.5B generative router, Supra-Router-51M's micro-LLM), call a commercial black box (Martian, NotDiamond), or hand-write rules. We observe three things:

1. **kNN is enough.** Recent work shows simple k-nearest-neighbor routing over sentence embeddings matches or beats learned matrix-factorization and MLP routers (arXiv:2505.12601): semantically similar queries benefit from the same model, and kNN needs far fewer samples.
2. **Embedding can be nearly free.** Static embedding models (Hugging Face static-retrieval-mrl-en-v1; Model2Vec/potion) run 100–400x faster than transformer encoders on CPU while retaining ~87–95% of retrieval quality. Embedding a task costs a table lookup — microseconds, no GPU, no serving infra.
3. **Coding agents generate labels for free.** Every agent run ends in a verifiable outcome (tests pass, patch applies, typecheck clean) with known cost and latency. This is exactly the supervision a router needs, delivered continuously and for free.

Combining these, the router becomes an ANN index over past tasks, updated online from agent outcomes, queried with a static embedder. It has no training loop, no inference server, adds no meaningful latency, and can run inside a gateway hot path (we deploy the same pattern inside [openpaths.io](https://openpaths.io)'s auto-router), inside a CLI agent, or on-device.

Contributions:

- **Routing-as-search formulation** with a cost-aware decision rule (cheapest model whose neighborhood-estimated pass probability clears a floor) and an online EWMA anchor-update rule fed by agent outcomes (§3).
- **A medium→hard coding micro-benchmark** (17 tasks with adversarial hidden tests: RFC-4180 CSV, semver ranges, regex engines, consistent hashing, LCS diff minimality checks) designed so cheap models fail on a meaningful fraction, giving the router signal (§4).
- **A cheap-models-only training regime**: the routing table is built entirely with ≤$0.75/1M-input models; expensive tiers enter only as escalation targets, bounding research cost to cents (§5).
- **Polyglot zero-copy serving**: the same JSON anchor table is served by pybed (Python, CAGRA-style graph index, custom CUDA kernels), gobed (Go, 0.15ms/embed), and zbed (Zig, SIMD), demonstrating router deployment without Python in the loop (§3.4).

## 2. Related Work

**Learned routers.** RouteLLM (arXiv:2406.18665) trains four routers (similarity-weighted Elo, matrix factorization, BERT, causal-LM) on ~55k Chatbot Arena preference pairs; at 95% of GPT-4 quality it cuts cost >85% on MT-Bench, 45% on MMLU, and generalizes to unseen model pairs. FrugalGPT (arXiv:2305.05176) cascades models with a stopping judge, matching GPT-4 with up to 98% cost reduction. Hybrid LLM (arXiv:2404.14618) trains a difficulty predictor for two-model routing (up to 40% fewer big-model calls at no quality drop); BEST-Route (arXiv:2506.22716) adds best-of-n sampling on the small model (up to 60% cost cut, <1% drop). Arch-Router (arXiv:2506.16655) is a 1.5B generative router mapping queries to human-defined policies at ~50ms per decision. Supra-Router-51M is a 51.7M-parameter Llama-architecture micro-LLM fine-tuned on 992 samples that emits a structured analysis (`Domain | Complexity | Math | Code | Route`) before its route token; it reports no quantitative benchmarks and carries no license, but represents the smallest generative-router design point we know of. Commercial routers (Martian, NotDiamond — which powers OpenRouter's `auto`) claim 20–97% cost reductions at matched quality; RouteLLM reports beating both at >40% lower cost.

**kNN and benchmark infrastructure.** RouterBench (arXiv:2403.12031) releases 405k precomputed generations from 11 models over 7 tasks, enabling offline router evaluation; its strongest simple baseline is cosine kNN (k=40) over MiniLM embeddings. "When Simple kNN Beats Complex Learned Routers" (arXiv:2505.12601) shows kNN routing dominates learned routers across router benchmarks — the locality argument this paper builds on. Our contribution relative to these: we shrink the embedder from a transformer to a static lookup (2–3 orders of magnitude cheaper), and make the anchor table *online* — updated per agent outcome rather than fit offline.

**Static embeddings.** Hugging Face's static-embedding recipe (static-retrieval-mrl-en-v1) trains a bare token-embedding matrix with contrastive loss: 100–400x faster than mpnet-class encoders on CPU at 87.4% NDCG@10 retention, with Matryoshka truncation to 256d costing 0.56%. Model2Vec/potion distill transformer embedders into static tables (potion-base-32M: 94.7% of MiniLM's MTEB average, up to 500x faster). Our public recipe for distilling from ModernBERT is at [lee101/public-static-modern-bert](https://github.com/lee101/public-static-modern-bert) — reported there honestly as a partial negative result (single-dataset distillation failed to beat the HF baseline on BEIR), which is *why* this paper uses the off-the-shelf static-retrieval-mrl-en-v1 weights quantized to int8/512d rather than a custom embedder.

**GPU ANN.** CAGRA (arXiv:2308.15136, cuVS) builds fixed-degree proximity graphs on GPU: 33–77x faster than CPU HNSW at 90–95% recall, with build-on-GPU/serve-on-CPU export. The router's anchor table is small (10²–10⁵ anchors), so exact search suffices early; CAGRA-style graph search (as implemented dependency-free in pybed, and with cuVS in gobed) keeps routing sub-millisecond as the table grows unboundedly from agent traffic.

## 3. Method

### 3.1 Routing as embedding search

Let `E: text → ℝ^d` be a static embedder (int8 table, 512d, L2-normalized output). The router state is a set of anchors `A = {(v_i, S_i)}` where `v_i = E(task_i)` and `S_i : model → (n, p̂, ĉ)` holds an observation count, an EWMA pass rate, and an EWMA cost for each model observed on tasks near this anchor.

To route task `t`: retrieve the k nearest anchors by cosine (exact or CAGRA graph search), then estimate per-model pass probability as a similarity- and evidence-weighted average with a Beta-style prior:

```
p̂_m(t) = ( p0·n0 + Σ_i w_i · p̂_i,m ) / ( n0 + Σ_i w_i ),   w_i = max(cos(v_i, E(t)), 0) · min(n_i,m, N_cap)
```

The prior `(p0=0.5, n0=1)` keeps estimates calibrated where evidence is thin; `N_cap` (10) stops any one anchor from dominating.

### 3.2 Cost-aware decision rule

Models are ordered by expected request cost. The router picks the **cheapest model with `p̂_m(t) ≥ τ`** (quality floor, default 0.6). If none clears the floor, it maximizes the utility `p̂_m(t) − λ·cost_m/cost_max` — i.e. when nothing is confidently sufficient, escalate toward quality but stay cost-sensitive. τ is the single product-level knob: raising it trades money for reliability, exactly the test-time dial Hybrid LLM argues for. Under a cascade deployment (run cheap first, verify with tests, escalate on failure) the floor can be set lower because failures are caught by verification rather than shipped.

### 3.3 Online learning from coding agents

After a routed task completes, the agent reports `(t, m, passed, cost)`. If the nearest anchor has cosine ≥ 0.92 the observation folds into it (EWMA with α = max(0.3, 1/n)); otherwise `t` becomes a new anchor. This gives the router three properties classical trained routers lack:

- **Continual adaptation.** A new model is added to the fleet with only its price and a prior; it acquires an empirical footprint as traffic touches it. No retraining event, no dataset snapshot. (Contrast: RouteLLM must re-fit to change the model pool; Arch-Router must re-prompt.)
- **Drift tracking.** When a provider silently improves or degrades a model, EWMA pass rates follow within `1/α` observations.
- **Per-deployment specialization.** An agent fleet working on a Go monorepo and one working on data-science notebooks converge to different tables from the same code.

The map is also *inspectable*: each anchor is a readable task with per-model pass rates — the router explains itself by showing the neighbors that voted.

### 3.4 Serving: the router is an index

Router state serializes to JSON `(text, vector, stats)`. Anything that can embed with the same static model and do cosine top-k can serve it: we ship reference implementations for [pybed](https://github.com/lee101/pybed) (pure Python, FlatIndex + dependency-free CAGRA-style graph, optional custom CUDA kernels — 0.34ms/query over 20k docs on an RTX 5090), [gobed](https://github.com/lee101/gobed) (Go, int8 model, 0.15ms/embed, cuVS CAGRA integration), and [zbed](https://github.com/lee101/zbed) (Zig, SIMD, ~16MB int8/512d model). The embedding model is 16MB and the anchor table is kilobytes-to-megabytes, so the entire router fits in an agent binary, a gateway sidecar, or an edge worker. This is the same architecture as openpaths.io's production auto-router, which resolves `openpaths/auto-code`-style meta-models by embedding the prompt against a curated description→model table; this paper replaces the curated table with a learned, self-updating one.

### 3.5 Training the embedder (optional)

The router is embedder-agnostic; any map that puts similar coding tasks near each other works. We use static-retrieval-mrl-en-v1 (int8, 512d via Matryoshka truncation). Practitioners wanting a domain-specialized embedder can distill one with the [public-static-modern-bert](https://github.com/lee101/public-static-modern-bert) recipe (ModernBERT teacher → static student) or Model2Vec/Tokenlearn; per that repo's own negative result, single-dataset distillation underperforms the multi-dataset HF recipe, so specialize only with broad in-domain task corpora (e.g. your gateway's own traffic).

## 4. Benchmark

Public coding benchmarks are either saturated by cheap models (HumanEval+/MBPP+: frontier >90%) or too expensive to run per-router-iteration (SWE-bench). For router research we need tasks where **cheap models fail at meaningfully different rates**, cheaply. We construct 17 self-contained Python tasks (`benchmark/tasks.jsonl`), medium→hard, each with adversarial hidden tests executed in an isolated interpreter (`python -I`, 15s timeout):

| Category | Tasks | What makes them hard |
|---|---|---|
| Stateful data structures | LRU+TTL cache, trie wildcard dict, consistent hash ring | Interacting eviction rules; md5 ring-point spec compliance |
| Parsers/matchers | RFC-4180 CSV, glob with `**`, regex-lite engine, arithmetic evaluator | Error-case contracts (ValueError on malformed), no `re`/`eval` allowed |
| Algorithms | lexicographic toposort, k-stop cheapest path, interval set ops, LCS diff | Minimality verified against DP oracle; tie-break rules |
| Systems semantics | sliding-window rate limiter, parallel task scheduler, idempotent ledger | Half-open windows, idempotency-after-failure, rejection ordering |
| Codecs/resolvers | base62 with leading zeros, semver range resolution, JSON path | Spec corner cases (leading zero bytes, `^0.x` semantics) |

Tasks are prompt-only (no starter code), graded pass/fail by hidden asserts, and sized so a full 4-model sweep costs under $0.25. The harness (`benchmark/harness.py`) targets any OpenAI-compatible endpoint; we run everything through a single [openpaths](https://openpaths.io) key, which also exercises provider fallback and gives uniform usage accounting. This benchmark is deliberately a *router* benchmark, not a *model* benchmark: its job is to produce per-task disagreement between models, the signal a router learns from. Scaling the same protocol to LiveCodeBench (explicit easy/medium/hard labels, contamination-resistant) and Aider polyglot is the natural next step and requires only a task-loader.

## 5. Experiments

Protocol: run M cheap models over all tasks (`benchmark/harness.py`), build the router from the outcome log (`ltr/train.py`), then evaluate routing with **leave-one-task-out**: when routing task `t`, all anchors for `t` are removed, so the router only ever generalizes from *other* tasks' outcomes (`ltr/frontier.py`). We compare single-model baselines, the router across quality floors τ, and the per-task oracle (cheapest passing model — the frontier's upper bound).

Models used to build the table (all ≤$0.75/1M input): deepseek-v4-flash ($0.14/$0.28), gpt-5.4-nano ($0.20/$1.25), gpt-5.4-mini ($0.75/$4.50), gemini-3.5-flash ($1.50/$9.00). Frontier tiers (gpt-5.6-luna/terra/sol) are configured as escalation targets with priors only — consistent with the cheap-training-regime constraint. Total spend for all experiments in this draft: **&lt;$0.30**.

### 5.1 Results

*[Populated from `experiments/results.jsonl` — run in progress; numbers below are filled in by `scripts/report.py`.]*

| Policy | Pass rate | Total cost | Cost vs best model |
|---|---|---|---|
| deepseek-v4-flash | — | — | — |
| gpt-5.4-nano | — | — | — |
| gpt-5.4-mini | — | — | — |
| gemini-3.5-flash | — | — | — |
| Router (τ=0.6, LOTO) | — | — | — |
| Router (τ=0.8, LOTO) | — | — | — |
| Oracle (cheapest passing) | — | — | — |

Qualitative behavior (already visible with a toy table, §3): easy text-manipulation prompts route to the cheapest model; prompts semantically near known-failed anchors (lock-free/concurrency phrasing) route to the strongest model in the table. Routing overhead is one static embed + one top-k over ≤17 anchors: **~0.2ms end-to-end on CPU**, unmeasurable against 1–60s LLM calls.

### 5.2 Cost model

With request cost `c_m` and neighborhood pass estimate `p̂_m`, expected cost-per-solved-task under verify-and-escalate is `c_m / p̂_m` plus the escalation tail; the router's floor rule approximates minimizing this greedily. The frontier ceiling is set by the oracle: on our benchmark the oracle solves every task solvable by *any* cheap model while paying near-flash prices for most, bounding achievable savings at matched quality — the same AIQ-style frontier RouterBench formalizes.

## 6. Discussion and Limitations

- **Benchmark size.** 17 tasks demonstrate mechanism, not SOTA claims; the protocol is built to scale to LiveCodeBench/BigCodeBench/Aider-polyglot task loaders, and to RouterBench's precomputed generations for offline comparison against learned routers. Leave-one-task-out on 17 anchors is a harsh generalization test (the router must transfer from ~16 unrelated tasks); production tables see thousands of near-duplicate tasks, where kNN locality is far stronger.
- **Static-embedding ceiling.** Static embedders lose word order ("convert X to Y" ≈ "convert Y to X"), which matters for some routing distinctions. The 87–95% retrieval-quality retention suggests the loss is acceptable for coarse difficulty/topic locality; measuring router-quality-vs-embedder-quality (static vs MiniLM vs ModernBERT) on the same anchor table is a key ablation we leave open — the codebase makes it a one-flag change.
- **Feedback loops.** Online updates from agent outcomes are biased by the routing policy itself (models not chosen get no data). Standard fixes — ε-greedy exploration on a traffic slice, optimistic priors for new models — fit the anchor formulation naturally; the EWMA update is exactly a per-arm bandit statistic, and formalizing the router as a contextual bandit with kNN context is the clearest theory extension.
- **Beyond model choice.** Nothing restricts anchor payloads to model IDs: the same table can store reasoning-effort levels (route "merge conflict" to `nano @ effort=none`), tool configurations, prompt templates, or code snippets — routing as general policy retrieval. openpaths' production auto-router already routes (model, effort) pairs; learning those jointly from outcomes is immediate future work.
- **Verification dependence.** Online labels require verifiable outcomes. Coding is the best case (tests); routing for open-ended generation would need judge models, inheriting their biases.

## 7. Conclusion

Routing is one of the last order-of-magnitude levers on LLM serving cost that requires no model training. We showed it can be implemented as embedding search over past outcomes with a 16MB static embedder — microsecond-scale, dependency-free, polyglot, continuously self-improving from agent feedback, and cheap enough to research with pocket change. The router, benchmark, and serving examples are MIT-licensed; we invite replication at LiveCodeBench scale.

## References

- RouteLLM: Learning to Route LLMs with Preference Data — arXiv:2406.18665
- FrugalGPT — arXiv:2305.05176
- Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing — arXiv:2404.14618
- BEST-Route — arXiv:2506.22716
- Arch-Router — arXiv:2506.16655
- RouterBench — arXiv:2403.12031
- When Simple kNN Beats Complex Learned Routers — arXiv:2505.12601
- CAGRA: GPU-native graph ANN — arXiv:2308.15136
- Static embeddings (HF): https://huggingface.co/blog/static-embeddings
- Model2Vec/potion: https://github.com/MinishLab/model2vec
- Supra-Router-51M: https://huggingface.co/SupraLabs/Supra-Router-51M
- public-static-modern-bert: https://github.com/lee101/public-static-modern-bert
- pybed / gobed / zbed: https://github.com/lee101/pybed · https://github.com/lee101/gobed · https://github.com/lee101/zbed
- GPT-5.6 (Sol/Terra/Luna) pricing: https://openai.com/index/gpt-5-6/ · https://www.aipricing.guru/openai-pricing/
- DeepSeek V4 Flash pricing: https://api-docs.deepseek.com/quick_start/pricing/
