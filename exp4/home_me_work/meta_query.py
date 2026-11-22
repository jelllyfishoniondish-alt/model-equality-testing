"""Like the official query(), but also logs response metadata for every answer:
latency (seconds), usage (prompt/completion tokens), finish_reason, every top-level field of the JSON body
except the answer text (model, id, system_fingerprint, ...), all response headers, and logprobs if requested.

Output: one JSON line per answer, same keys as the official files plus "meta". Resumable (skips done answers).
The API key is read from the ME_API_KEY environment variable and is never written to the output."""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

SERVER_URL = os.environ.get("ME_SERVER_URL", "http://<competition-server>")
TASK_ID = "33-model-equality"
BASE_URL = f"{SERVER_URL}/{TASK_ID}/v1"


def session():
    key = os.environ.get("ME_API_KEY", "").strip()
    if not key:
        sys.exit('export ME_API_KEY="your team key" first')
    s = requests.Session()
    s.headers["X-API-Key"] = key
    return s


def budget(s):
    r = s.get(f"{BASE_URL}/models", timeout=60)
    r.raise_for_status()
    return {m["id"]: m["budget"]["limit"] - m["budget"]["used"] for m in r.json()["data"]}


def query_meta(prompts, output, endpoints, repeats, max_tokens=16, parallel=1, logprobs=False):
    s = session()
    left = budget(s)
    output = Path(output)
    done = set()
    if output.exists():
        done = {(r["endpoint"], r["id"], r["repeat"]) for r in map(json.loads, output.open(encoding="utf-8")) if r}
    jobs, planned = [], {e: 0 for e in endpoints}
    for rep in range(repeats):
        for pid, prompt in prompts:
            for ep in endpoints:
                if (ep, pid, rep) not in done and planned[ep] < left.get(ep, 0):
                    planned[ep] += 1
                    jobs.append((ep, pid, rep, prompt))
    print(f"{len(jobs)} answers to query ({len(done)} already in {output}), {parallel} in parallel")
    lock, out, n = threading.Lock(), output.open("a", encoding="utf-8"), [0]

    def one(job):
        ep, pid, rep, prompt = job
        payload = {"model": ep, "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
        if logprobs:
            payload.update({"logprobs": True, "top_logprobs": 5})
        while True:
            t0 = time.perf_counter()
            try:
                r = s.post(f"{BASE_URL}/chat/completions", timeout=180, json=payload)
            except requests.RequestException as exc:
                print(f"unreachable ({type(exc).__name__}), retry in 30 s"); time.sleep(30); continue
            dt = time.perf_counter() - t0
            if r.status_code == 429 and "Retry-After" in r.headers:
                time.sleep(int(r.headers["Retry-After"])); continue
            if r.status_code in (502, 503, 504):
                time.sleep(15); continue
            if r.status_code != 200:
                print(f"{ep} id {pid}: HTTP {r.status_code}: {r.text[:200]}", file=sys.stderr); return
            body = r.json()
            ch = body["choices"][0]
            meta = {"latency_s": round(dt, 4),
                    "finish_reason": ch.get("finish_reason"),
                    "logprobs": ch.get("logprobs"),
                    "body": {k: v for k, v in body.items() if k != "choices"},
                    "headers": {k: v for k, v in r.headers.items() if k.lower() not in ("set-cookie", "x-api-key")}}
            line = {"endpoint": ep, "id": pid, "repeat": rep, "prompt": prompt,
                    "response": ch["message"]["content"], "meta": meta}
            with lock:
                out.write(json.dumps(line, ensure_ascii=False) + "\n"); out.flush()
                n[0] += 1
                if n[0] % 10 == 0 or n[0] == len(jobs):
                    print(f"{n[0]}/{len(jobs)}")
            return

    try:
        with ThreadPoolExecutor(max_workers=max(1, min(4, parallel))) as pool:
            list(pool.map(one, jobs))
    finally:
        out.close()
