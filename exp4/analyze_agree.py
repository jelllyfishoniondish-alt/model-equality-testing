"""Many prompts x 1 answer: pairwise agreement rate between endpoints (normalised answers).
Usage: python analyze_agree.py answers.jsonl [prefix]
For families, prints within-family agreement matrix; families from FAMILIES below (v1)."""
import json, re, sys, itertools, collections, statistics as st
path = sys.argv[1]; prefix = sys.argv[2] if len(sys.argv) > 2 else "ep-"
def norm(t):
    t = t.strip().lower().split("\n")[0]
    t = re.sub(r"[*\"'`.!,:;()]", "", t).strip()
    t = re.sub(r"^(the|a|an) ", "", t)
    return " ".join(t.split()[:3])
A = collections.defaultdict(dict)
for l in open(path):
    r = json.loads(l)
    if r["endpoint"].startswith(prefix): A[r["endpoint"]][r["prompt"]] = norm(r["response"])
eps = sorted(A)
def agree(a, b):
    ps = set(A[a]) & set(A[b]); return sum(A[a][p] == A[b][p] for p in ps) / len(ps) if ps else float("nan")
FAM = {"X": [1,12,14,17,18,24,25,33], "Y": [3,7,8,21,22,26,28,30,32], "Z": [5,9,11,13,15,16,19,27,29,31], "W": [2,23]}
if prefix == "ref-":
    FAM = {"P": list(range(1, 10))}
for k, v in FAM.items():
    es = [f"{prefix}{i:02d}" for i in v if f"{prefix}{i:02d}" in A]
    if len(es) < 2: continue
    print(f"== family {k}: agreement (fraction of prompts with identical normalised answer), n prompts = {len(A[es[0]])}")
    print("       " + " ".join(e[-2:].rjust(4) for e in es))
    for a in es:
        print(f"  {a} " + " ".join("   -" if a == b else f"{agree(a,b):4.2f}" for b in es) + f"   mean {st.mean(agree(a,b) for b in es if b!=a):.2f}")
allw = [agree(a, b) for k, v in FAM.items() for a, b in itertools.combinations([f"{prefix}{i:02d}" for i in v if f"{prefix}{i:02d}" in A], 2)]
if prefix == "ep-":
    fam_of = {f"ep-{i:02d}": k for k, v in FAM.items() for i in v}
    cross = [agree(a, b) for a, b in itertools.combinations(eps, 2) if fam_of.get(a) != fam_of.get(b) or a not in fam_of]
    print(f"within-family mean {st.mean(allw):.2f}, cross-family mean {st.mean(cross):.2f}")
