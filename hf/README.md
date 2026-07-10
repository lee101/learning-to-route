---
license: mit
task_categories:
- text-generation
tags:
- llm-routing
- code-generation
- benchmark
- static-embeddings
pretty_name: Learning to Route
---

# Learning to Route — coding router benchmark + anchor tables

Companion artifacts for [Learning to Route](https://github.com/lee101/learning-to-route): a medium→hard coding benchmark (17 tasks with adversarial hidden tests), per-model outcome logs from cheap LLMs (deepseek-v4-flash, gpt-5.4-nano/mini, gemini-3.5-flash), and trained router anchor tables (static-embedding vectors + per-model pass/cost stats).

Routing = one static embed (~0.15ms CPU, 16MB model) + k-NN over these anchors. Serve from Python ([pybed](https://github.com/lee101/pybed)), Go ([gobed](https://github.com/lee101/gobed)), or Zig ([zbed](https://github.com/lee101/zbed)).

- `tasks.jsonl` — benchmark tasks (prompt, entry_point, difficulty, hidden tests)
- `results.jsonl` — (task, model, passed, cost, latency) outcome log
- `router.json` — trained anchor table: `{text, vec[512], stats: {model: {n, pass, cost}}}`

See the [whitepaper](https://github.com/lee101/learning-to-route/blob/main/paper/learning-to-route.md).
