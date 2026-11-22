"""Per-prompt discrimination on ref (2 rounds). For each prompt:
  S-agree  = fraction of answer pairs that match among S endpoints (all rounds, incl. same endpoint across rounds)
  V-agree  = fraction of pairs (S answer, variant answer) that match, per variant and pooled
  gap      = S-agree - V-agree (positive = disguises agree with each other more than with variants)
Invalid answers (blank/refusal/>4 words/multi-line) are excluded and counted."""
import json, re, sys, itertools, collections
REFUSE = re.compile(r"\b(sorry|cannot|can't|as an ai|i am an ai|i'm an ai)\b", re.I)
def norm(t):
    s = t.strip().strip(" \t\n\"'`*.,!?:;()[]").lower()
    s = " ".join(s.split())
    if not s or REFUSE.search(t) or len(s.split()) > 4 or "\n" in t.strip():
        return None
    return s
path = sys.argv[1]
S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]; V = ["ref-02", "ref-09", "ref-04", "ref-07"]
A = collections.defaultdict(list)  # (prompt_id, endpoint) -> [norm answers]
P = {}
for l in open(path):
    r = json.loads(l); A[(r["id"], r["endpoint"])].append(norm(r["response"])); P[r["id"]] = r["prompt"]
def rate(pairs):
    v = [a == b for a, b in pairs if a is not None and b is not None]
    return (sum(v) / len(v) if v else float("nan")), len(v)
rows = []
for pid in sorted(P, key=int):
    sa = [x for e in S for x in A[(pid, e)]]
    s_pairs = list(itertools.combinations(sa, 2))
    s_rate, _ = rate(s_pairs)
    per = {}
    for v in V:
        per[v] = rate([(a, b) for a in sa for b in A[(pid, v)]])[0]
    pooled = rate([(a, b) for a in sa for v in V for b in A[(pid, v)]])[0]
    inval = sum(x is None for e in S + V for x in A[(pid, e)])
    top = collections.Counter(x for x in sa if x).most_common(3)
    rows.append((s_rate - pooled, pid, s_rate, pooled, per, inval, top))
print("gap = S-agree - variant-agree; per variant: fp8 / LoRA0.3 / LoRA0.7 / LoRA1.5")
for gap, pid, s, pooled, per, inval, top in sorted(rows, reverse=True):
    print(f"[{pid:>2}] gap {gap:+.2f}  S {s:.2f}  var {pooled:.2f}  ({' / '.join(f'{per[v]:.2f}' for v in V)})  invalid {inval:2d}  S-top {top}  | {P[pid][:55]}")
