"""Exp4: open-ended probes on ref endpoints only (never ep). Style features may separate LoRA/fp8 from disguises."""
import argparse, os, sys
from pathlib import Path
import query_reference as qr
ap = argparse.ArgumentParser()
ap.add_argument("--repeats", type=int, default=12)
ap.add_argument("--max-tokens", type=int, default=100)
args = ap.parse_args()
key = os.environ.get("ME_API_KEY", "").strip()
if not key:
    sys.exit("export ME_API_KEY first")
qr.API_KEY = key
qr.PARALLEL = 4
prompts = qr.read_prompts(Path("exp4_prompts.txt"))
endpoints = [f"ref-{i:02d}" for i in range(1, 10)]
qr.query(prompts, output=Path("exp4_answers.jsonl"), endpoints=endpoints, repeats=args.repeats, max_tokens=args.max_tokens)
