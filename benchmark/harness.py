import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

PRICES = {
    "deepseek-v4-flash": (0.14, 0.28),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.6-luna": (1.00, 6.00),
    "gpt-5.6-terra": (2.50, 15.00),
    "gpt-5.6-sol": (5.00, 30.00),
    "gemini-3.5-flash": (1.50, 9.00),
    "claude-haiku": (1.00, 5.00),
}

SYSTEM = (
    "You are a precise coding assistant. Reply with a single Python code block "
    "containing the complete solution and nothing else. No explanations."
)


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    return blocks[-1] if blocks else text


def run_tests(code: str, tests: str, timeout: float = 15.0) -> tuple[bool, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n\n" + tests + "\nprint('LTR_PASS')\n")
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, "-I", path],
            capture_output=True, text=True, timeout=timeout,
            env={"PATH": os.environ.get("PATH", ""), "HOME": tempfile.gettempdir()},
        )
        ok = proc.returncode == 0 and "LTR_PASS" in proc.stdout
        return ok, (proc.stderr or proc.stdout)[-500:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(path)


def call_model(client: httpx.Client, base_url: str, api_key: str, model: str,
               prompt: str, max_tokens: int = 4000, effort: str | None = None) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_completion_tokens": max_tokens,
    }
    if effort:
        body["reasoning_effort"] = effort
    t0 = time.time()
    resp = client.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=body, timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    pin, pout = PRICES.get(model, (1.0, 4.0))
    cost = (usage.get("prompt_tokens", 0) * pin + usage.get("completion_tokens", 0) * pout) / 1e6
    return {
        "text": data["choices"][0]["message"]["content"] or "",
        "cost": cost,
        "latency": time.time() - t0,
        "usage": usage,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks.jsonl"))
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    base_url = os.environ.get("LTR_BASE_URL", "https://api.openpaths.co/v1").rstrip("/")
    api_key = os.environ["LTR_API_KEY"]
    tasks = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
    if args.limit:
        tasks = tasks[: args.limit]
    models = args.models.split(",")

    done = set()
    out_path = Path(args.out)
    if out_path.exists():
        for l in out_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                done.add((r["task_id"], r["model"]))

    with httpx.Client() as client, out_path.open("a") as out:
        for task in tasks:
            for model in models:
                if (task["id"], model) in done:
                    continue
                try:
                    resp = call_model(client, base_url, api_key, model,
                                      task["prompt"], effort=args.effort)
                    code = extract_code(resp["text"])
                    passed, detail = run_tests(code, task["tests"])
                except Exception as e:
                    resp = {"cost": 0.0, "latency": 0.0, "usage": {}}
                    code, passed, detail = "", False, f"api_error: {e}"[:300]
                row = {
                    "task_id": task["id"],
                    "task": task["prompt"],
                    "difficulty": task["difficulty"],
                    "model": model,
                    "passed": passed,
                    "cost": resp["cost"],
                    "latency": resp["latency"],
                    "detail": detail if not passed else "",
                }
                out.write(json.dumps(row) + "\n")
                out.flush()
                print(f"{task['id']:>20} {model:>20} pass={passed} "
                      f"cost=${resp['cost']:.5f} {resp['latency']:.1f}s")


if __name__ == "__main__":
    main()
