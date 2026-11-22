"""Number 1-100 on ref with n answers per endpoint (subsampled, 5 draws). Pairwise mmd_perm / tvtemp,
average linkage; report the best-threshold model MCC (optimistic) and a held-out version (threshold picked
on one draw, scored on another), plus fp8 / LoRA z vs the same-model group."""
import sys, json, random, itertools, math, statistics as st, collections
sys.argv = sys.argv[:1] + sys.argv[1:]
import cluster_ep as C
from answer_norm import normalise
P = "Pick a random number between 1 and 100. Answer with only the number."
pool = collections.defaultdict(list)
for fn in ["ref_num60.jsonl", "exp2_answers.jsonl", "exp3_answers.jsonl", "reference_answers.jsonl", "ref_four.jsonl", "ref_choice.jsonl"]:
    try:
        for l in open(fn):
            r = json.loads(l)
            if r["prompt"].strip() == P and r["endpoint"].startswith("ref-"): pool[r["endpoint"]].append(normalise(r["response"], P, "category"))
    except FileNotFoundError: pass
eps = sorted(pool); print("available per endpoint:", {e[-2:]: len(pool[e]) for e in eps})
truth = {l.split(",")[0]: l.split(",")[2] for l in open("practice_answers.csv").readlines()[1:]}
S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
def mcc(groups):
    lab = {e: i for i, g in enumerate(groups) for e in g}; tp = fp = fn = tn = 0
    for a, b in itertools.combinations(eps, 2):
        t = truth[a] == truth[b]; g = lab[a] == lab[b]; tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return max(0, (tp * tn - fp * fn) / d) if d else 0
def link(M, t):
    cl = [[e] for e in eps]
    while True:
        best = None
        for i, j in itertools.combinations(range(len(cl)), 2):
            ds = [M[x, y] for x in cl[i] for y in cl[j]]; dd = sum(ds) / len(ds)
            if best is None or dd < best[0]: best = (dd, i, j)
        if best is None or best[0] > t: break
        cl[best[1]] += cl.pop(best[2])
    return cl
for metric in ["mmd_perm", "tvtemp"]:
    f = C.mmd_perm if metric == "mmd_perm" else C.tv_temp
    for n in [20, 40, 60]:
        if min(len(v) for v in pool.values()) < n: continue
        Ms = []
        for seed in range(4):
            rng = random.Random(seed); sub = {e: rng.sample(pool[e], n) for e in eps}
            M = {}
            for a, b in itertools.combinations(eps, 2): M[a, b] = M[b, a] = f(sub[a], sub[b])
            Ms.append(M)
        opt = st.mean(max(mcc(link(M, t)) for t in sorted(set(round(v, 3) for v in M.values()))) for M in Ms)
        held = []
        for i in range(len(Ms)):
            A, B = Ms[i], Ms[(i + 1) % len(Ms)]
            t = max(sorted(set(round(v, 3) for v in A.values())), key=lambda t: mcc(link(A, t)))
            held.append(mcc(link(B, t)))
        M = Ms[0]
        zS = [M[a, b] for a, b in itertools.combinations(S, 2)]
        print(f"[{metric} n={n}] model MCC optimistic {opt:.2f} | held-out {st.mean(held):.2f} | same-model pairs max {max(zS):.2f} mean {st.mean(zS):.2f} | "
              + " ".join(f"{v}: min-to-S {min(M[v, s] for s in S):.2f}" for v, v_ in [("ref-02", 0), ("ref-09", 0), ("ref-04", 0), ("ref-07", 0)]))
