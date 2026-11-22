"""Experiment 3: test new probe candidates on the ref endpoints only (unlimited, never scored).

Probes (exp3_prompts.txt, ids = line numbers 0..6):
    0  number 1-100      (baseline, exact text used on ep)
    1  animal            (baseline, exact text used on ep)
    2  number 1-10
    3  letter A-Z
    4  color
    5  Chinese number 1-100
    6  FLIPS-style binary sequence of length 100 (needs more tokens)

START THIS ONLY AFTER THE ep RUN HAS FINISHED: the server allows at most 4 requests at a time per team,
and both runs share that limit.

Usage (same folder as the official query_reference.py):
    export ME_API_KEY="your team key"
    python run_exp3.py                     # 30 repeats, 4 parallel
    python run_exp3.py --repeats 20        # faster
    python run_exp3.py --parallel 2        # leave room for another job
    python run_exp3.py --skip-binary       # only the short probes

Short probes use max_tokens 16; the binary probe uses max_tokens 200.
Answers are appended to exp3_answers.jsonl (same format as exp2). Re-running is safe and only fills gaps.
"""
import argparse
import os
import sys
from pathlib import Path

import query_reference as qr

ap = argparse.ArgumentParser()
ap.add_argument("--repeats", type=int, default=30)
ap.add_argument("--parallel", type=int, default=4, help="concurrent requests (server limit per team: 4)")
ap.add_argument("--prompts", default="exp3_prompts.txt")
ap.add_argument("--output", default="exp3_answers.jsonl")
ap.add_argument("--skip-binary", action="store_true")
args = ap.parse_args()

key = os.environ.get("ME_API_KEY", "").strip()
if not key:
    sys.exit('Run this first: export ME_API_KEY="your team key"')
qr.API_KEY = key
qr.PARALLEL = max(1, min(4, args.parallel))   # query() reads this global when it starts its thread pool

prompts = qr.read_prompts(Path(args.prompts))
binary = [(i, p) for i, p in prompts if "0 and 1" in p]
short = [(i, p) for i, p in prompts if "0 and 1" not in p]
endpoints = [f"ref-{i:02d}" for i in range(1, 10)]
out = Path(args.output)

print(f"{len(short)} short probes + {len(binary)} binary probe(s) x {len(endpoints)} ref endpoints x "
      f"{args.repeats} repeats, {qr.PARALLEL} in parallel")
qr.query(short, output=out, endpoints=endpoints, repeats=args.repeats, max_tokens=16)
if binary and not args.skip_binary:
    qr.query(binary, output=out, endpoints=endpoints, repeats=args.repeats, max_tokens=200)
