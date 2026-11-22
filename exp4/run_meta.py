"""Collect answers WITH metadata (latency, usage tokens, headers, optional logprobs).

Default: reference endpoints only (free). ep endpoints spend real budget and need --allow-ep.
    export ME_API_KEY="your team key"
    python run_meta.py --repeats 5                          # ref-01..09, prompts in meta_prompts.txt, 1 request at a time
    python run_meta.py --repeats 5 --logprobs               # also ask for token logprobs (if the server supports them)
    python run_meta.py --endpoints ep-01,ep-12 --repeats 3 --allow-ep
Writes meta_answers.jsonl. Re-running fills gaps only."""
import argparse
import sys
from pathlib import Path

from meta_query import query_meta

ap = argparse.ArgumentParser()
ap.add_argument("--prompts", default="meta_prompts.txt")
ap.add_argument("--output", default="meta_answers.jsonl")
ap.add_argument("--endpoints", default=",".join(f"ref-{i:02d}" for i in range(1, 10)))
ap.add_argument("--repeats", type=int, default=5)
ap.add_argument("--max-tokens", type=int, default=16)
ap.add_argument("--parallel", type=int, default=1, help="1 by default, to leave slots for teammates (team limit 4)")
ap.add_argument("--logprobs", action="store_true")
ap.add_argument("--allow-ep", action="store_true", help="required for ep-* endpoints (spends real budget)")
a = ap.parse_args()
eps = [e.strip() for e in a.endpoints.split(",") if e.strip()]
if any(e.startswith("ep-") for e in eps) and not a.allow_ep:
    sys.exit("ep endpoints spend real budget: add --allow-ep if you really mean it")
if a.prompts.endswith(".jsonl"):
    import json as _j
    prompts = [(str(r["id"]), r["prompt"]) for r in map(_j.loads, open(a.prompts, encoding="utf-8")) if r]
else:
    lines = [l.rstrip("\n") for l in open(a.prompts, encoding="utf-8") if l.strip()]
    prompts = [(str(i), p) for i, p in enumerate(lines)]
query_meta(prompts, a.output, eps, a.repeats, a.max_tokens, a.parallel, a.logprobs)
