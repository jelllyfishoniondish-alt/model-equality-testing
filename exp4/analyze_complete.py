"""2-shot completions on ref: first number/word of the reply; agreement among same-model versions and vs variants."""
import json, re, collections, itertools, statistics as st
S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]; V = ["ref-02", "ref-09", "ref-04", "ref-07"]
name = {"ref-01": "plain", "ref-03": "sysprompt", "ref-05": "persona", "ref-06": "watermark", "ref-08": "temp1.0",
        "ref-02": "fp8", "ref-09": "LoRA0.3", "ref-04": "LoRA0.7", "ref-07": "LoRA1.5"}
def first(t):
    m = re.search(r"-?\d[\d,.]*|[A-Za-z][A-Za-z'-]*", t.strip())
    return m.group(0).strip(".,").lower().replace(",", "") if m else None
A = collections.defaultdict(list); P = {}
for l in open("ref_complete.jsonl"):
    r = json.loads(l); A[(r["id"], r["endpoint"])].append(first(r["response"])); P[r["id"]] = r["prompt"].split("\n")[-1]
def agree(a, b, q):
    xs = [x for x in A[(q, a)] if x]; ys = [y for y in A[(q, b)] if y]
    pr = [x == y for x in xs for y in ys]; return sum(pr) / len(pr) if pr else None
qs = sorted(P, key=int); stable = []
for q in qs:
    sv = [x for e in S for x in A[(q, e)] if x]; c = collections.Counter(sv).most_common(1)
    mode, share = (c[0][0], c[0][1] / len(sv)) if sv else (None, 0)
    flips = [name[v] for v in V if A[(q, v)] and all(x and x != mode for x in A[(q, v)])]
    if share >= 0.8: stable.append(q)
    print(f"[{q:>2}] {P[q][:38]:38s} S-mode {mode!r} share {share:.2f} | " + " ".join(f"{name[v]}={A[(q,v)]}" for v in V) + (f"  <- both answers differ: {flips}" if share >= 0.8 and flips else ""))
def mean_agree(pairs, qset):
    v = [agree(a, b, q) for q in qset for a, b in pairs]; v = [x for x in v if x is not None]; return st.mean(v) if v else float("nan")
print(f"\nstable items (>=80% of same-model answers identical): {len(stable)}/{len(qs)}")
for label, qset in [("all items", qs), ("stable items", stable)]:
    print(f"== {label}")
    print(f"   plain vs disguises: " + " ".join(f"{name[e]} {mean_agree([('ref-01', e)], qset):.2f}" for e in S[1:]))
    print(f"   plain vs variants : " + " ".join(f"{name[e]} {mean_agree([('ref-01', e)], qset):.2f}" for e in V))
    print(f"   all S-S pairs {mean_agree(list(itertools.combinations(S, 2)), qset):.2f} | S-variant pairs {mean_agree([(a, b) for a in S for b in V], qset):.2f}")
