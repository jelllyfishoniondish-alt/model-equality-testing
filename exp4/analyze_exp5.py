import json, re, math, sys, difflib, itertools, collections, statistics as st
kind = {l.split(",")[0]: l.strip().split(",")[3] for l in open("practice_answers.csv").readlines()[1:]}
S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
def load(fn):
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for l in open(fn):
        r = json.loads(l); d[r["endpoint"]][r["prompt"]].append(r["response"])
    return d
def report(name, D, eps):
    """D(a,b) distance. For each endpoint: mean distance to S members (excl. self); S-baseline from S-S pairs."""
    ss = [D(a, b) for a, b in itertools.combinations(S, 2)]
    mu, sd = st.mean(ss), st.pstdev(ss) or 1e-9
    print(f"== {name}: S-S distance mean {mu:.3f} sd {sd:.3f}  (score = (mean dist to S - S mean)/sd; >2 = separated)")
    for e in eps:
        v = st.mean(D(e, s) for s in S if s != e)
        print(f"   {e} {kind[e][:22]:22s} dist {v:.3f}  score {(v-mu)/sd:+.1f}")
# --- translation: 1 - difflib ratio between token sequences, cross minus within ---
def tok(t): return re.findall(r"\w+|[^\w\s]", t.lower())
def sim(a, b): return difflib.SequenceMatcher(None, tok(a), tok(b)).ratio()
def avg_cross(X, Y): return st.mean(sim(a, b) for a in X for b in Y)
def avg_within(X): return st.mean(sim(a, b) for a, b in itertools.combinations(X, 2)) if len(X) > 1 else 1
def flips_feats(t):
    b = [c for c in t if c in "01"]
    n = len(b) or 1
    ones = b.count("1") / n
    trans = [(b[i], b[i+1]) for i in range(len(b)-1)]
    def p(x, y):
        fr = [t for t in trans if t[0] == x]; return sum(1 for t in fr if t[1] == y) / len(fr) if fr else 0.5
    runs = [len(m.group()) for m in re.finditer(r"0+|1+", "".join(b))] or [0]
    return [ones, p("0", "1"), p("1", "0"), st.mean(runs), max(runs), len(b) / 100, float(len(b) == 100),
            float(bool(re.search(r"[^01\s]", t)))]
def zdist(X, Y):
    fx = [flips_feats(t) for t in X]; fy = [flips_feats(t) for t in Y]; out = []
    for k in range(len(fx[0])):
        a = [f[k] for f in fx]; b = [f[k] for f in fy]
        se = math.sqrt(st.pvariance(a)/len(a) + st.pvariance(b)/len(b))
        out.append(((st.mean(a)-st.mean(b))/se)**2 if se > 1e-9 else 0)
    return math.sqrt(sum(out)/len(out))
def bow(t): return collections.Counter(re.findall(r"[a-z']+|[^\w\s]", t.lower()))
def cos(a, b):
    n = sum(a[k]*b[k] for k in a); d = math.sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values())); return n/d if d else 0
def mmd(X, Y):
    X = [bow(t) for t in X]; Y = [bow(t) for t in Y]
    kxx = st.mean(cos(a, b) for a, b in itertools.permutations(X, 2)); kyy = st.mean(cos(a, b) for a, b in itertools.permutations(Y, 2))
    return kxx + kyy - 2*st.mean(cos(a, b) for a in X for b in Y)
which = sys.argv[1:] or ["translate", "flips", "content"]
eps = [f"ref-{i:02d}" for i in range(1, 10)]
if "translate" in which:
    d = load("ans_translate.jsonl"); p = next(iter(d["ref-01"]))
    W = {e: avg_within(d[e][p]) for e in eps}
    print("within-endpoint similarity:", {e[-2:]: round(W[e], 3) for e in eps})
    C = {}
    def Dt(a, b):
        k = tuple(sorted((a, b)))
        if k not in C: C[k] = (W[a] + W[b]) / 2 - avg_cross(d[a][p], d[b][p])
        return C[k]
    report("translation (within - cross similarity)", Dt, eps)
if "flips" in which:
    d = load("ans_flips.jsonl"); p = next(iter(d["ref-01"]))
    for e in eps:
        F = [flips_feats(t) for t in d[e][p]]
        print("  ", e, " ".join(f"{st.mean(f[k] for f in F):.2f}" for k in range(8)), "(ones p01 p10 meanrun maxrun len/100 exact100 nonbinary)")
    report("FLIPS feature z", lambda a, b: zdist(d[a][p], d[b][p]), eps)
if "content" in which:
    d = load("ans_content.jsonl"); ps = sorted(d["ref-01"])
    report("content BoW-MMD (avg over 8 prompts)", lambda a, b: st.mean(mmd(d[a][q], d[b][q]) for q in ps), eps)
    # LoRA markers: tokens over-represented in ref-07 vs S
    c7 = collections.Counter(); cS = collections.Counter()
    for q in ps:
        for t in d["ref-07"][q]: c7.update(set(bow(t)))
        for s in S:
            for t in d[s][q]: cS.update(set(bow(t)))
    n7 = sum(len(d["ref-07"][q]) for q in ps); nS = sum(len(d[s][q]) for s in S for q in ps)
    lo = {w: math.log((c7[w]+1)/(n7+2)) - math.log((cS[w]+1)/(nS+2)) for w in c7}
    markers = [w for w, v in sorted(lo.items(), key=lambda x: -x[1])[:15]]
    print("ref-07 markers:", markers)
    for e in eps:
        tot = sum(len(d[e][q]) for q in ps)
        rate = st.mean(sum(1 for w in markers if w in bow(t)) for q in ps for t in d[e][q])
        print(f"   {e} {kind[e][:22]:22s} markers per answer {rate:.2f}")
