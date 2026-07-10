# experiments

- `results.jsonl` — one row per (task, model) run from `benchmark/harness.py`
- `router.json` — anchor table trained from results via `ltr.train`
- `results.sample.jsonl` — committed sample so the pipeline runs without API access

Reproduce:

```bash
export LTR_BASE_URL=https://openpaths.io/v1 LTR_API_KEY=op-...
uv run python benchmark/harness.py --models deepseek-v4-flash,gpt-5.4-nano,gpt-5.4-mini,gemini-3.5-flash --out experiments/results.jsonl
uv run python scripts/report.py
```
