"""Experiment 1: send every probe in reference_prompts.txt to ref-01 ... ref-09, REPEATS times each.

Why repeat?
  - If an endpoint returns exactly the same text for the same prompt every time, it uses greedy
    (deterministic) decoding.
  - Sampling endpoints (e.g. ref-08) answer differently each time, so we need several samples to
    estimate their answer distribution.
  The ref endpoints have no query limit and are never scored, so we can query them freely.

Usage (on JURECA, in the same folder as the official query_reference.py):
    export ME_API_KEY="your team key"
    python run_ref.py              # 3 repeats per prompt (default)
    python run_ref.py 5            # 5 repeats per prompt

Answers are appended to reference_answers.jsonl. Stopping and re-running is safe: answers already
in the file are skipped.
"""
import os
import sys

import query_reference as qr   # reuse query() from the official script

REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 3

# 1. API key: read it from an environment variable so it never ends up in a file
key = os.environ.get("ME_API_KEY", "").strip()
if not key:
    sys.exit('Run this first: export ME_API_KEY="your team key"')
qr.API_KEY = key   # the official make_session() reads this global at call time

# 2. Read the probe prompts (one per line)
prompts = qr.read_prompts(qr.PROMPTS_FILE)
print(f"Read {len(prompts)} prompts from {qr.PROMPTS_FILE}")

# 3. Query only the ref endpoints (the ep endpoints have a budget; never touch them here)
endpoints = [f"ref-{i:02d}" for i in range(1, 10)]
print(f"Querying {len(endpoints)} ref endpoints x {len(prompts)} prompts x {REPEATS} repeats "
      f"= {len(endpoints) * len(prompts) * REPEATS} answers")

# Note: pass repeats explicitly. The default in the official query() was fixed to REPEATS=1
# when the function was defined.
qr.query(prompts, output=qr.ANSWERS_FILE, endpoints=endpoints, repeats=REPEATS)
