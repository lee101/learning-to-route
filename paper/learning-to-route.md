# Learning to Route: Static Embedding Models as Self Improving LLM Routers for Coding Tasks

**Lee Penkman**
lee.penkman@gmail.com · [openpaths.io](https://openpaths.io) · [github.com/lee101/learning-to-route](https://github.com/lee101/learning-to-route) · [huggingface.co/openpaths/learning-to-route](https://huggingface.co/openpaths/learning-to-route)

*Draft v2, July 2026*

## Abstract

Frontier coding models cost 5x to 55x more per token than small models that solve most real coding tasks just as well. GPT-5.6 Sol is $5.00 per million input tokens while DeepSeek V4 Flash is $0.14 official and $0.09 spot. Routing each task to the cheapest model that can solve it is therefore one of the largest levers left on serving cost, and it needs no model training at all. This paper describes a router where routing is literally embedding search. A task is embedded by a static embedding model, which is a token table lookup and a mean pool with no transformer forward pass, about 0.15 milliseconds on CPU from a 16 MB file. The k nearest previously seen tasks then vote on which model to use, weighted by their observed pass rates. Because the router state is just vectors plus outcome counters, it learns online: every completed agent task folds its result back into the nearest anchor, so the task to model map improves continuously from live traffic with no training loop. On a 27 problem coding benchmark built for this paper, spanning medium to hard exact tasks plus optimization tasks scored by solution quality, a verify and escalate cascade over four cheap models solves 100% of tasks at 26% of the cost of the best single model (85.2%) and at 4% of the cost of a frontier tier model, gpt-5.5, which reaches only 77.8%. Per solved task the cascade costs $0.0041 against the frontier model's $0.1213, a 30x gap. The entire experimental history cost about four dollars of API credit. The router, benchmark, paper and serving code in Python, Go and Zig are all MIT licensed and deployed against the production router at openpaths.io.

![Cost quality frontier](figs/frontier.png)

## 1. Introduction

The biggest waste in agentic coding is not failing hard tasks. It is sending easy tasks to expensive models. A production gateway sees the same shapes of work all day: fix an off by one, write a parser, resolve a merge conflict, add a test. A $0.14 per million model and a $5.00 per million model solve most of these identically. A minority genuinely need the frontier tier, and you usually cannot tell which from a price list.

OpenAI made the tradeoff explicit in July 2026 by shipping price tiers as a product. GPT-5.6 comes as Sol ($5.00/$30.00 per million tokens in/out), Terra ($2.50/$15.00) and Luna ($1.00/$6.00). Anthropic's Claude Fable 5 sits at a similar frontier price point. That is a 5x spread inside one model family and roughly 55x against spot priced small models. Whoever picks the right point on that curve per task, rather than per month, keeps frontier quality while paying small model prices.

Existing routers train a classifier (RouteLLM's BERT router, Arch-Router's 1.5B generative router, Supra-Router-51M's micro LLM), call a commercial black box (Martian, NotDiamond), or hand write rules. Three observations suggest something simpler:

1. Nearest neighbour routing is strong. Recent work shows plain kNN over sentence embeddings matches or beats learned matrix factorization and MLP routers (arXiv:2505.12601). Similar tasks want the same model, and kNN needs very few samples to see that.
2. Embedding can be nearly free. Static embedding models run 100x to 400x faster than transformer encoders on CPU while keeping about 87% to 95% of retrieval quality. Embedding a task is a memory read.
3. Coding agents produce labels for free. Every agent run ends in a verifiable outcome (tests pass, patch applies) with a known cost and latency. That is exactly the supervision a router needs, delivered continuously.

So the router in this paper is an ANN index over past tasks, updated online from agent outcomes, queried with a static embedder. There is no training loop and no inference server. It runs inside a gateway hot path, inside a CLI agent, or on device. The same pattern already powers the auto router inside [openpaths.io](https://openpaths.io), where prompts are matched against a table of task descriptions to pick a model and a reasoning effort. This paper replaces the curated table with a learned, self updating one, and everything found here deploys straight back into that production router.

## 2. Related Work

RouteLLM (arXiv:2406.18665) trains routers on 55k Chatbot Arena preference pairs and cuts cost over 85% on MT-Bench at 95% of GPT-4 quality. FrugalGPT (arXiv:2305.05176) cascades models with a stopping judge and reports up to 98% cost reduction at matched quality. Hybrid LLM (arXiv:2404.14618) trains a difficulty predictor for two model routing, and BEST-Route (arXiv:2506.22716) adds best of n sampling on the small model for up to 60% cost cut with under 1% quality drop. Arch-Router (arXiv:2506.16655) maps queries to human defined policies with a 1.5B model at about 50 ms per decision. Supra-Router-51M is a 51.7M parameter Llama architecture micro LLM fine tuned on 992 samples that emits a structured analysis string ending in a route token. It is the smallest generative router design point we know of, though it publishes no quantitative evaluation and no license, which limits what can be built on it. Commercial routers (Martian, NotDiamond, which powers OpenRouter auto) claim 20% to 97% cost reductions at matched quality.

RouterBench (arXiv:2403.12031) releases 405k precomputed generations from 11 models so routers can be evaluated offline, and its strongest simple baseline is cosine kNN over MiniLM embeddings. The kNN result was sharpened by arXiv:2505.12601, which finds plain kNN beats learned routers across routing benchmarks. This paper leans on that finding and pushes it further in two directions: the embedder shrinks from a transformer to a static lookup table, which is two to three orders of magnitude cheaper, and the anchor table becomes online, updated per outcome rather than fit offline.

On the embedding side, the Hugging Face static embedding recipe (static-retrieval-mrl-en-v1) trains a bare token embedding matrix with contrastive loss and reports 100x to 400x CPU speedups at 87.4% of the retrieval quality of all-mpnet-base-v2, with Matryoshka truncation to smaller dimensions costing under 1%. Model2Vec and the potion models get similar results by distillation. My own recipe for distilling a static model from ModernBERT is public at [lee101/public-static-modern-bert](https://github.com/lee101/public-static-modern-bert) and is reported there honestly as a partial negative result: single dataset distillation did not beat the multi dataset baseline on BEIR. That is exactly why this paper uses the off the shelf static-retrieval-mrl-en-v1 weights, quantized to int8 at 512 dimensions (about 16 MB), rather than a custom embedder.

For serving, CAGRA (arXiv:2308.15136, NVIDIA cuVS) builds fixed degree proximity graphs on GPU and searches 33x to 77x faster than CPU HNSW at 90% to 95% recall, with a build on GPU serve on CPU export path. The anchor tables in this paper are small enough for exact search, but the graph path matters because a production table grows without bound from agent traffic. The three companion libraries, [gobed](https://github.com/lee101/gobed) (Go), [zbed](https://github.com/lee101/zbed) (Zig) and [pybed](https://github.com/lee101/pybed) (Python), all serve the same int8 static model with flat and CAGRA style graph indexes, which is what makes the router portable across languages.

## 3. Method

### 3.1 Routing as embedding search

Let E map text to a unit vector in R^512 via the static embedder. The router state is a set of anchors. Each anchor holds a vector for a previously seen task and, per model, an observation count n, an exponentially weighted pass rate p, and an average cost. To route a new task, retrieve its k nearest anchors by cosine similarity and estimate each model's pass probability as a similarity and evidence weighted average with a weak prior:

```
p_m(t) = ( p0 * n0 + sum_i w_i * p_i,m ) / ( n0 + sum_i w_i )
w_i    = max(cos(v_i, E(t)), 0) * min(n_i,m, 10)
```

The prior (p0 = 0.5, n0 = 1) keeps estimates near 0.5 where evidence is thin, and the cap stops any single anchor from dominating.

### 3.2 Cost aware decision rule

Models are sorted by expected request cost. The router picks the cheapest model whose estimated pass probability clears a floor tau (default 0.6). If nothing clears the floor it maximizes p_m minus a small cost penalty, which means escalate toward quality but stay price aware. Tau is the one product knob: raise it to trade money for reliability. When the deployment can verify results and retry (the normal agentic coding case), tau can sit low because failures are caught by tests rather than shipped, and section 5 shows this cascade mode is where the method shines.

### 3.3 Online learning from agent outcomes

When a routed task finishes, the agent reports the task text, the model, whether it passed, and what it cost. If the nearest anchor is very close (cosine at least 0.92) the observation folds into it. Otherwise the task becomes a new anchor. This update rule gives the router properties trained routers do not have:

- New models join the fleet with just a price and a prior, and acquire an empirical footprint as traffic touches them. There is no retraining event. This is what makes the July 2026 model churn manageable: when GPT-5.6 Luna appeared it could start receiving exploratory traffic the same day.
- When a provider silently improves or degrades a model, the moving average follows within a handful of observations.
- Two deployments specialize differently. An agent fleet on a Go monorepo and one on data science notebooks converge to different tables from the same code.
- The map is inspectable. Every routing decision can be explained by showing the neighbor tasks that voted and their pass rates.

### 3.4 The router is an index

Router state serializes to JSON as (text, vector, stats) triples. Anything that can run the same static embedder and a cosine top k can serve it. The repo ships working examples for pybed (pure Python, flat and CAGRA style graph indexes, optional CUDA kernels, 0.34 ms per query over 20k documents on an RTX 5090), gobed (Go, 0.15 ms per embed) and zbed (Zig, SIMD). The model file is 16 MB and a big anchor table is megabytes, so the whole router fits inside an agent binary, a gateway, or an edge worker. Routing overhead is microseconds against LLM calls that take seconds.

### 3.5 One space, many embedders

A practical lesson from training our own static models ([public-static-modern-bert](https://github.com/lee101/public-static-modern-bert)) is that a 512 dimension embedding space has plenty of spare capacity and redundancy. You can run more than one static embedder over the same text, for example the English retrieval model and the multilingual similarity model, and fuse the outputs by plain averaging into one space, and retrieval still works. For routing this means one anchor table can serve traffic across languages and embedder generations: adding a second embedder is an average, not a migration. We do not use fusion in the experiments below because a single embedder is sufficient at this scale, but it is the upgrade path when a router table needs to cover inputs one embedder handles poorly.

### 3.6 Routing as an intelligence lerp

A useful way to think about what the router does: it linearly interpolates intelligence per request. Model families now expose a price and capability dial with discrete stops, Luna, Terra, Sol on the OpenAI side, Haiku, Sonnet, Opus and now Fable on the Anthropic side, plus a long tail of small open models below them. A fixed choice pins every request to one stop. The router turns the discrete tiers into a continuous curve: easy requests resolve at the cheap end, hard ones at the frontier end, and the blend point per task is learned from outcomes rather than guessed. The aggregate effect is a deployment that sits between tiers, for example 95% of Sol quality at closer to Luna prices.

This is also why the method scales to bigger models without any changes. Adding GPT-5.6 Sol or Claude Fable 5 to the fleet is one price entry and a prior. The anchors and the embedder do not care how large the target model is, they only track who solves what for how much. As frontier models improve, the router shifts traffic toward whichever tier newly dominates its price point, which means users of a routed endpoint track the moving intelligence frontier automatically instead of re benchmarking and re configuring every launch week. That is the product we run at openpaths.io, and the improvements measured in this paper deploy directly into its auto router.

## 4. Benchmark

Public coding benchmarks are either saturated by cheap models (frontier models exceed 90% on HumanEval+ and MBPP+) or too expensive to rerun per router iteration (SWE-bench). Router research needs tasks where cheap models fail at meaningfully different rates, cheaply. So the repo includes 27 self contained Python tasks with adversarial hidden tests run in an isolated interpreter (python -I). 22 are exact pass or fail tasks, medium to hard:

- Stateful data structures: an LRU cache with TTL and interacting eviction rules, a wildcard trie, a consistent hash ring that must match an exact md5 point spec.
- Parsers and matchers: RFC 4180 CSV without the csv module, path globbing with ** semantics, a small regex engine, an arithmetic expression evaluator with Python floor semantics and no eval.
- Algorithms: lexicographically smallest topological sort, cheapest path with a stop budget, interval set operations, minimal LCS diff scripts verified against a DP oracle.
- Systems semantics: a sliding window rate limiter with half open windows, a parallel task scheduler, an idempotent transaction ledger with ordered rejections.
- Codecs and resolvers: base62 with leading zero preservation, semver range resolution including caret on 0.x, JSON path lookup.
- Interpreters and spec heavy parsing: a mini Lisp with closures and recursion, cron expression next run with day of month or day of week OR semantics, a SQL subset over dicts, a JSON Schema subset, POSIX like shell tokenization.

The other 5 are optimization tasks with no optimal requirement, only a quality score in [0, 1] against a reference bound: euclidean TSP tour construction (scored against nearest neighbour plus 2-opt), one dimensional bin packing (against first fit decreasing), 0/1 knapsack with 200 items (against the exact DP optimum), makespan scheduling on identical machines (against the trivial lower bound), and writing a lossless compressor from scratch with a banned library check (scored on size against zlib level 9, with an exact roundtrip required). Each has a pass threshold (for example 0.97 of optimal for knapsack, 0.30 of zlib for compression) so pass or fail policies and score aware policies can both be evaluated. These tasks measure something exact tests cannot: how good an answer a model produces when perfection is off the table, which is where coding agents actually live most of the day.

The hidden tests were validated by writing reference solutions for the most spec heavy tasks in both waves, and four test bugs found this way were fixed before any numbers were recorded. The reference solutions also calibrate the optimization thresholds: plain nearest neighbour TSP scores 0.846 and fails the 0.92 threshold, while adding 2-opt passes, so the thresholds reward algorithm quality rather than boilerplate. Tasks are prompt only, graded pass or fail, and a full four model sweep costs about 25 cents. The harness targets any OpenAI compatible endpoint and every call in this paper went through a single openpaths.io key, which also exercised provider fallbacks and gave uniform usage accounting. This is deliberately a router benchmark rather than a model benchmark: its job is to produce per task disagreement between models, which is the signal a router learns from. Figure 2 shows exactly that.

![Per task outcomes](figs/heatmap.png)

## 5. Experiments

Protocol: run the models over all 27 tasks, build the router from the outcome log, then evaluate with leave one task out. When routing task t every anchor for t is removed first, so the router only generalizes from other tasks. All numbers below are from `experiments/report.json`, regenerated end to end by `scripts/experiments.py`.

The routing fleet is four cheap models: deepseek-v4-flash ($0.14/$0.28 per million), gpt-5.4-nano ($0.20/$1.25), gpt-5.4-mini ($0.75/$4.50), gemini-3.5-flash ($1.50/$9.00). To test the assumption that expensive means better, v2 adds one frontier tier model as a fifth column: gpt-5.5 ($5.00/$30.00), the strongest OpenAI model our org can call today (the GPT-5.6 tiers are configured as escalation targets but were still preview gated for our org at time of writing; slotting in gpt-5.6-luna when access lands is a one line change). Total spend for the full experimental history of this paper: about four dollars, of which gpt-5.5 alone was $2.55.

### 5.1 Single models, router, cascade, oracle

| Policy | Pass rate | Mean score | Total cost | Cost per solved task | Median latency |
|---|---|---|---|---|---|
| deepseek-v4-flash | 74.1% | 0.719 | $0.0252 | $0.0013 | 33s |
| gpt-5.4-nano | 77.8% | 0.778 | $0.0301 | $0.0014 | 6s |
| gpt-5.4-mini | 85.2% | 0.852 | $0.4262 | $0.0185 | 27s |
| gemini-3.5-flash | 44.4% | 0.444 | $0.8563 | $0.0714 | 17s |
| gpt-5.5 (frontier) | 77.8% | 0.778 | $2.5479 | $0.1213 | 28s |
| Router alone (floor 0.5, LOTO) | 77.8% | 0.756 | $0.0251 | $0.0012 | |
| Cascade, price order | **100%** | 0.979 | $0.1107 | $0.0041 | |
| Router start cascade (floor 0.5) | **100%** | 0.979 | $0.1095 | $0.0041 | |
| Oracle, cheapest passing | 100% | 1.000 | $0.0902 | $0.0033 | |

![Cost per solved task](figs/cost_per_solve.png)

Four results stand out.

First, the frontier model does not win. gpt-5.5 at $2.55 lands on exactly the same pass rate as gpt-5.4-nano at $0.03, an 85x price gap for zero quality gain on this workload, and it trails gpt-5.4-mini by 7 points. Per solved task it is the single worst deal on the board at $0.1213, 30x the cascade. Whatever gpt-5.5 is better at (and on long horizon agentic work it surely is), a mixed bag of hard self contained coding tasks is not where its premium pays.

Second, the oracle still solves everything without frontier help. Every one of the 27 tasks, including the five optimization tasks with their quality thresholds, is solved by at least one sub dollar model. The per task disagreement is strong (figure 2): csv_parser falls only to gemini-3.5-flash, the weakest model overall; the from scratch compressor falls only to the two OpenAI minis; deepseek uniquely fails tasks the others find easy. That anti correlation is the entire routing opportunity.

Third, the router alone is now on the pareto frontier. At floor 0.5 it matches gpt-5.4-nano's pass rate at 83% of nano's cost, because it has learned to send deepseek friendly tasks to deepseek and keep nano for the rest. With 26 leave one out anchors the neighbourhoods are still thin, and the router alone still cannot reach mini's 85.2%; its headline value remains cascade mode, where choosing the starting rung keeps full coverage while trimming wasted first attempts ($0.1095 vs $0.1107, a small but real saving that grows with table size).

Fourth, verification turns cheap models into a frontier beating system. The cascade reaches 100% pass and 0.979 mean score at $0.1107: 26% of mini's cost, 4.3% of gpt-5.5's, while beating both outright on quality. On the optimization tasks the cascade's escalation also raises answer quality, not just pass rates, because a later model's better tour or tighter packing replaces an earlier model's marginal one.

### 5.2 Ablations

![k ablation](figs/k_ablation.png)

Neighbourhood size k moves quality by a few points between 1 and 16 with no clean monotone trend, confirming estimates are prior dominated at this scale; k starts mattering when tables reach thousands of anchors, which is where the graph index earns its keep. Swapping the static embedder for a random hash projection of tokens still routes competitively (85.2% vs 74.1% pass at floor 0.6, with different escalation spend), and honesty requires repeating this: at 27 anchors, locality comes mostly from shared vocabulary. The semantic embedder's paraphrase robustness is expected to separate from the hash baseline as anchors densify; measuring exactly where that crossover happens is the first experiment to run at LiveCodeBench scale.

### 5.3 What this costs to research

The full experimental history of this paper, two benchmark waves, test debugging, provider timeouts and retries, and every ablation, cost about four dollars, and the cheap model portion of it under one. The single most expensive line item was measuring that the frontier model is not worth measuring. This is the methodological point of the cheap training regime: a router learns everything it needs from models priced like commodities, and the expensive tiers only ever need to be priced in, not benchmarked at scale.
## 6. Discussion and Limitations

Twenty seven tasks demonstrate mechanism, not state of the art. The benchmark is sized to make router iteration essentially free, and the protocol scales directly to LiveCodeBench (which has explicit easy, medium, hard labels and contamination resistant problems) and to RouterBench's 405k precomputed generations for offline comparison against learned routers. Both need only a task loader.

Leave one task out at n = 27 is close to a worst case for kNN; production tables see near duplicate tasks all day, which is where locality actually pays. The flip side is a feedback loop: models that never get chosen never get data. The anchor statistics are per arm bandit statistics, so epsilon greedy exploration on a small traffic slice and optimistic priors for new models are the natural fixes, and formalizing the router as a contextual bandit with kNN context is the clearest theory extension.

Static embedders lose word order, which caps how fine grained the routing signal can get. The measured gap between the static embedder and a hash baseline is small at this scale, so the honest claim is that the static embedder buys paraphrase robustness and a 16 MB deployment, not benchmark points, yet.

Nothing restricts anchor payloads to model IDs. The same table can store reasoning effort levels (route merge conflicts to nano with reasoning off), tool configurations, prompt templates, or code snippets. Routing generalizes to policy retrieval. The openpaths auto router already routes model and effort pairs; learning those jointly from outcomes is immediate future work, as is verifying results with cheaper signals than full test suites (linting, type checks, partial test selection) so cascades stay cheap on tasks with slow test harnesses.

## 7. Conclusion

Routing is the cheapest way left to move the cost quality frontier of coding agents: on this benchmark a learned cascade of sub dollar models beats every single model on both axes at once, solving 100% of tasks at 26% of the best cheap model's cost and 4% of a frontier model's cost, while the frontier model itself ties a $0.03 model on pass rate. The router that achieves this is a 16 MB static embedding model and a JSON file of task vectors with pass rate counters, updated by the agents it routes, servable from Python, Go or Zig in under a millisecond. As model families keep shipping price tiers, Luna to Sol, Haiku to Fable, this kind of router acts as an intelligence lerp, blending tiers per request so a deployment tracks the frontier continuously instead of re standardizing on a new model every quarter. Everything here, the benchmark, the router, the serving code and this paper, is MIT licensed at [github.com/lee101/learning-to-route](https://github.com/lee101/learning-to-route) and mirrored at [huggingface.co/openpaths/learning-to-route](https://huggingface.co/openpaths/learning-to-route), and the measured improvements deploy directly into the auto router at [openpaths.io](https://openpaths.io).

## References

1. RouteLLM: Learning to Route LLMs with Preference Data. arXiv:2406.18665
2. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176
3. Hybrid LLM: Cost Efficient and Quality Aware Query Routing. arXiv:2404.14618
4. BEST-Route: Adaptive LLM Routing with Test Time Optimal Compute. arXiv:2506.22716
5. Arch-Router: Aligning LLM Routing with Human Preferences. arXiv:2506.16655
6. RouterBench: A Benchmark for Multi LLM Routing Systems. arXiv:2403.12031
7. When Simple kNN Beats Complex Learned Routers. arXiv:2505.12601
8. CAGRA: Highly Parallel Graph Construction and Approximate Nearest Neighbor Search for GPUs. arXiv:2308.15136
9. Train 400x Faster Static Embedding Models. https://huggingface.co/blog/static-embeddings
10. Model2Vec. https://github.com/MinishLab/model2vec
11. Supra-Router-51M. https://huggingface.co/SupraLabs/Supra-Router-51M
12. public-static-modern-bert. https://github.com/lee101/public-static-modern-bert
13. pybed, gobed, zbed. https://github.com/lee101/pybed · https://github.com/lee101/gobed · https://github.com/lee101/zbed
14. GPT-5.6 announcement and pricing. https://openai.com/index/gpt-5-6/
15. DeepSeek V4 pricing. https://api-docs.deepseek.com/quick_start/pricing/
