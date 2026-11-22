"""Collect answers from the SCORED endpoints ep-01 ... ep-33 for a small probe set.

THIS SPENDS REAL BUDGET (100 answers per endpoint for the whole competition). The script
  - reads the probes from a file (one per line; default ep_probes.txt),
  - plans probes x repeats answers per endpoint, minus answers already in the output file,
  - refuses to run if any endpoint does not have enough budget left for its plan,
  - shows the plan and asks you to type "yes".

Usage (in the same folder as the official task_template.py):
    export ME_API_KEY="your team key"
    python ep_collect.py --repeats 25                       # 3 probes x 25 = 75 per endpoint
    python ep_collect.py --probes my_probes.txt --repeats 20 --max-tokens 16

Answers are appended to ep_answers.jsonl ({"endpoint","id","repeat","prompt","response"}).
Re-running is safe: answers already in the file are skipped.
Probe ids are the probe TEXT's line number, so do not reorder the probes file between runs.
"""
import argparse
import json
import sys
import os
from pathlib import Path

import task_template as tt   # official query() and get_budget()

ap = argparse.ArgumentParser()
ap.add_argument("--probes", default="ep_probes.txt")
ap.add_argument("--repeats", type=int, required=True)
ap.add_argument("--max-tokens", type=int, default=16)
ap.add_argument("--output", default="ep_answers.jsonl")
args = ap.parse_args()

key = os.environ.get("ME_API_KEY", "").strip()
if not key:
    sys.exit('Run this first: export ME_API_KEY="your team key"')
tt.API_KEY = key

probes = tt.read_prompts(Path(args.probes))
out = Path(args.output)
done = set()
if out.exists():
    with out.open(encoding="utf-8") as f:
        done = {(r["endpoint"], r["id"], r["repeat"]) for r in (json.loads(l) for l in f if l.strip())}

left = tt.get_budget(tt.make_session())
eps = sorted(e for e in left if e.startswith("ep-"))
plan = {e: sum((e, pid, r) not in done for pid, _ in probes for r in range(args.repeats)) for e in eps}
short = [e for e in eps if plan[e] > left[e]]

print(f"{len(probes)} probes x {args.repeats} repeats on {len(eps)} endpoints, max_tokens {args.max_tokens}")
for pid, p in probes:
    print(f"  [{pid}] {p}")
print(f"New answers per endpoint: {min(plan.values())}..{max(plan.values())} (total {sum(plan.values())})")
print(f"Budget left per endpoint now: {min(left[e] for e in eps)}..{max(left[e] for e in eps)}; "
      f"after this run: {min(left[e] - plan[e] for e in eps)}..{max(left[e] - plan[e] for e in eps)}")
if short:
    sys.exit(f"Not enough budget on {short}. Lower --repeats. Nothing was queried.")
if sum(plan.values()) == 0:
    sys.exit("Nothing to do: all planned answers are already in the output file.")
if input('Type "yes" to spend this budget: ').strip().lower() != "yes":
    sys.exit("Stopped, nothing was queried.")

tt.query(probes, output=out, endpoints=eps, repeats=args.repeats, max_tokens=args.max_tokens)
left = tt.get_budget(tt.make_session())
print("Budget left now:", ", ".join(f"{e}:{left[e]}" for e in eps))
