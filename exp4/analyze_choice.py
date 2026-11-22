"""Validation batch: select prompts on round 0, validate on rounds 1-2.
Prompt score (round 0): S-agree (pairs among the 5 same-model endpoints) minus agreement S-vs-variants.
Design comparison by pair AUC over all 36 ref pairs (10 same-model pairs): similarity = mean over prompts of
P(answers match) between the two endpoints, using only the allowed rounds."""
import json, re, sys, itertools, collections, statistics as st
REFUSE = re.compile(r"\b(sorry|cannot|can't|as an ai|i am an ai|i'm an ai)\b", re.I)
def norm(t):
    s = t.strip().strip(" \t\n\"'`*.,!?:;()[]").lower(); s = " ".join(s.split())
    return None if (not s or REFUSE.search(t) or len(s.split()) > 4 or "\n" in t.strip()) else s
path = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 5
S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]; V = ["ref-02", "ref-09", "ref-04", "ref-07"]
kind = dict(zip(S + V, ["plain", "sysprompt", "persona", "watermark", "temp1.0", "fp8", "LoRA0.3", "LoRA0.7", "LoRA1.5"]))
A = collections.defaultdict(dict); P = {}
for l in open(path):
    r = json.loads(l); A[(r["id"], r["endpoint"])][r["repeat"]] = norm(r["response"]); P[r["id"]] = r["prompt"]
pids = sorted(P, key=int)
def ans(pid, e, rounds): return [A[(pid, e)].get(k) for k in rounds if A[(pid, e)].get(k) is not None]
def match(xs, ys):
    pr = [(a == b) for a in xs for b in ys]; return sum(pr) / len(pr) if pr else None
def agree_pairs(pid, rounds, group_a, group_b=None):
    vals = []
    pairs = itertools.combinations(group_a, 2) if group_b is None else [(a, b) for a in group_a for b in group_b]
    for a, b in pairs:
        m = match(ans(pid, a, rounds), ans(pid, b, rounds))
        if m is not None: vals.append(m)
    return st.mean(vals) if vals else float("nan")
print("== per-prompt, selection round 0 (and all 3 rounds for reference) ==")
score = {}
for pid in pids:
    s0, v0 = agree_pairs(pid, [0], S), agree_pairs(pid, [0], S, V)
    s3, v3 = agree_pairs(pid, [0, 1, 2], S), agree_pairs(pid, [0, 1, 2], S, V)
    score[pid] = (s0 - v0) if s0 == s0 and v0 == v0 else -9
    modal = collections.Counter(x for e in S for x in ans(pid, e, [0, 1, 2])).most_common(2)
    per = " ".join(f"{kind[v]}:{','.join(str(x) for x in ans(pid, v, [0,1,2]))}" for v in V)
    print(f"[{pid:>2}] r0 gap {score[pid]:+.2f} (S {s0:.2f} var {v0:.2f}) | 3-round gap {s3-v3:+.2f} (S {s3:.2f} var {v3:.2f}) | S-mode {modal} | {per} | {P[pid][:45]}")
top = sorted(pids, key=lambda p: -score[p])[:K]
print(f"\nselected on round 0 (top {K}): {top}")
truth = lambda a, b: a in S and b in S
def auc(prompts, rounds):
    sims = []
    for a, b in itertools.combinations(S + V, 2):
        vals = [m for m in (match(ans(p, a, rounds), ans(p, b, rounds)) for p in prompts) if m is not None]
        sims.append((st.mean(vals) if vals else 0.0, truth(a, b)))
    pos = [s for s, t in sims if t]; neg = [s for s, t in sims if not t]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
print("\n== design comparison: pair AUC (same-model pairs vs different-model pairs; 0.5 = chance) ==")
print(f"  15 prompts x 1 answer, round 1        : {auc(pids, [1]):.2f}")
print(f"  15 prompts x 1 answer, round 2        : {auc(pids, [2]):.2f}")
print(f"  top-{K} x 2 answers (rounds 1-2, held out): {auc(top, [1, 2]):.2f}")
print(f"  top-{K} x 3 answers (incl. selection round, optimistic): {auc(top, [0, 1, 2]):.2f}")
print(f"  all 15 x 3 answers (45 per endpoint)  : {auc(pids, [0, 1, 2]):.2f}")
