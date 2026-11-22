"""Experiment 2: many short "pick something at random, answer with one word" probes, many repeats.

Idea: experiment 1 showed that every ref endpoint SAMPLES (none is greedy), and that the two best probes
were "pick a random number" and "name a random animal". For such probes the answer is a single word,
so we can count how often each answer appears and compare the answer HISTOGRAMS between endpoints.
Quantization and LoRA change the model's probabilities, so they should change these histograms.

Usage (on JURECA, in the same folder as the official query_reference.py):
    export ME_API_KEY="your team key"
    python run_exp2.py                 # 30 repeats, max_tokens 16 (default)
    python run_exp2.py 50              # 50 repeats
    python run_exp2.py 30 8            # 30 repeats, max_tokens 8

20 prompts x 9 ref endpoints x 30 repeats = 5,400 answers. ref endpoints have no limit and are never scored.
Answers go to exp2_answers.jsonl (same format as reference_answers.jsonl). Re-running is safe: answers
already in the file are skipped, and raising the repeat count only adds the missing repeats.
"""
import os
import sys
from pathlib import Path

import query_reference as qr   # reuse query() from the official script

REPEATS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
MAX_TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 16   # one-word answers: short = fast
PROMPTS_FILE = Path("exp2_prompts.txt")
ANSWERS_FILE = Path("exp2_answers.jsonl")

key = os.environ.get("ME_API_KEY", "").strip()
if not key:
    sys.exit('Run this first: export ME_API_KEY="your team key"')
qr.API_KEY = key   # the official make_session() reads this global at call time

prompts = qr.read_prompts(PROMPTS_FILE)
endpoints = [f"ref-{i:02d}" for i in range(1, 10)]   # only ref endpoints; never spend ep budget here
print(f"{len(prompts)} prompts x {len(endpoints)} endpoints x {REPEATS} repeats "
      f"= {len(prompts) * len(endpoints) * REPEATS} answers, max_tokens={MAX_TOKENS}")

qr.query(prompts, output=ANSWERS_FILE, endpoints=endpoints, repeats=REPEATS, max_tokens=MAX_TOKENS)
