"""Style features of open-ended answers; z-score of each endpoint vs a group, and word-level MMD."""
import json, re, sys, math, random, collections, itertools
path = sys.argv[1]; prefix = sys.argv[2] if len(sys.argv) > 2 else "ref-"
rows = [json.loads(l) for l in open(path) if l.strip()]
def feats(t):
    s = t.strip(); w = s.split()
    return {"chars": len(s), "words": len(w), "sents": len(re.findall(r"[.!?](\s|$)", s)),
            "nl": s.count("\n"), "excl": s.count("!"), "q": s.count("?"),
            "end_dot": float(s.endswith(".")), "starts_I": float(s[:2] in ("I ", "I'")),
            "ai": float(bool(re.search(r"\b(AI|assistant|language model)\b", s))),
            "avgw": sum(map(len, w)) / max(1, len(w)), "trunc": float(not s.endswith((".", "!", "?", '"')))}
data = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if r["endpoint"].startswith(prefix): data[r["endpoint"]][r["prompt"]].append(r["response"])
eps = sorted(data); prompts = sorted({p for e in eps for p in data[e]})
def zfeat(xs, ys):
    """RMS over features of standardized mean difference (Welch); ~1 if same distribution."""
    fx = [feats(t) for t in xs]; fy = [feats(t) for t in ys]; out = []
    for k in fx[0]:
        a = [f[k] for f in fx]; b = [f[k] for f in fy]
        ma, mb = sum(a)/len(a), sum(b)/len(b)
        va = sum((v-ma)**2 for v in a)/max(1, len(a)-1); vb = sum((v-mb)**2 for v in b)/max(1, len(b)-1)
        se = math.sqrt(va/len(a) + vb/len(b))
        out.append(((ma-mb)/se)**2 if se > 1e-9 else 0.0)
    return math.sqrt(sum(out)/len(out))
def bow(t): return collections.Counter(re.findall(r"[a-z']+", t.lower()))
def wordmmd(xs, ys):
    """Unbiased MMD^2 with cosine kernel on bag-of-words."""
    def cos(a, b):
        n = sum(a[k]*b[k] for k in a); d = math.sqrt(sum(v*v for v in a.values())*sum(v*v for v in b.values()))
        return n/d if d else 0
    X = [bow(t) for t in xs]; Y = [bow(t) for t in ys]
    kxx = sum(cos(a, b) for a, b in itertools.permutations(X, 2))/(len(X)*(len(X)-1))
    kyy = sum(cos(a, b) for a, b in itertools.permutations(Y, 2))/(len(Y)*(len(Y)-1))
    kxy = sum(cos(a, b) for a in X for b in Y)/(len(X)*len(Y))
    return kxx + kyy - 2*kxy
groups = None
if prefix == "ref-":
    kind = {l.split(",")[0]: l.strip().split(",")[3] for l in open("practice_answers.csv").readlines()[1:]}
for p in prompts:
    print("####", p[:60])
    print("  mean features:")
    for e in eps:
        F = [feats(t) for t in data[e][p]]
        avg = {k: sum(f[k] for f in F)/len(F) for k in F[0]}
        lab = kind[e][:18] if prefix == "ref-" else ""
        print(f"   {e} {lab:18s} n={len(F):2d} " + " ".join(f"{k}={avg[k]:.1f}" for k in ["chars","words","sents","end_dot","starts_I","ai","nl","trunc"]))
    print("  pairwise style z (rows vs cols) and word-MMD x100:")
    print("          " + " ".join(e[-2:].rjust(9) for e in eps))
    for a in eps:
        print(f"   {a} " + " ".join(("    -    " if a == b else f"{zfeat(data[a][p], data[b][p]):4.1f}/{100*wordmmd(data[a][p], data[b][p]):4.1f}") for b in eps))
