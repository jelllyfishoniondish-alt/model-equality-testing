"""Ref replication of the many-prompts idea: 15 prompts x 2 rounds (repeat 0 / 1) on ref-01..09.
Minimal normalisation (case, surrounding punctuation/whitespace); invalid answers (blank, refusal,
long chatter > 4 words, unparseable) are counted separately and never count as a match.
Reports matches / valid prompts per pair, per round, and self-agreement (round 1 vs round 2)."""
import json, re, sys, itertools, collections, statistics as st
path = sys.argv[1] if len(sys.argv) > 1 else "ref_many.jsonl"
kind = {"ref-01": "plain", "ref-02": "fp8", "ref-03": "sysprompt", "ref-04": "LoRA0.7", "ref-05": "persona",
        "ref-06": "watermark", "ref-07": "LoRA1.5", "ref-08": "temp1.0", "ref-09": "LoRA0.3"}
REFUSE = re.compile(r"\b(sorry|cannot|can't|as an ai|i am an ai|i'm an ai)\b", re.I)
def norm(t):
    s = t.strip().strip(" \t\n\"'`*.,!?:;()[]").lower()
    s = " ".join(s.split())
    if not s or REFUSE.search(t) or len(s.split()) > 4 or "\n" in t.strip():
        return None
    return s
R = collections.defaultdict(dict)   # R[(ep, round)][prompt] = norm or None
raw = collections.defaultdict(dict)
for l in open(path):
    r = json.loads(l)
    R[(r["endpoint"], r["repeat"])][r["id"]] = norm(r["response"])
    raw[(r["endpoint"], r["repeat"])][r["id"]] = r["response"]
eps = sorted(kind)
def agree(a, ra, b, rb):
    A, B = R[(a, ra)], R[(b, rb)]
    ps = [p for p in set(A) & set(B) if A[p] is not None and B[p] is not None]
    m = sum(A[p] == B[p] for p in ps)
    return m, len(ps)
print("invalid answers per endpoint (round 0 / round 1):")
for e in eps:
    print(f"  {e} {kind[e]:9s} " + " / ".join(str(sum(v is None for v in R[(e, k)].values())) + "/" + str(len(R[(e, k)])) for k in (0, 1)))
print("\nself-agreement (same endpoint, round 0 vs round 1):")
for e in eps:
    m, n = agree(e, 0, e, 1); print(f"  {e} {kind[e]:9s} {m}/{n}")
for rnd in (0, 1):
    print(f"\n== round {rnd}: plain (ref-01) vs each endpoint, matches/valid ==")
    for e in eps[1:]:
        m, n = agree("ref-01", rnd, e, rnd); print(f"  ref-01 vs {e} {kind[e]:9s} {m}/{n} = {m/n if n else float('nan'):.2f}")
S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
D = ["ref-02", "ref-04", "ref-07", "ref-09"]
print("\n== group summary (mean agreement rate over pairs) ==")
for rnd in (0, 1):
    ss = [agree(a, rnd, b, rnd) for a, b in itertools.combinations(S, 2)]
    sd = [agree(a, rnd, b, rnd) for a in S for b in D]
    rate = lambda L: st.mean(m / n for m, n in L if n)
    print(f"  round {rnd}: same-model pairs (S-S) {rate(ss):.2f}  |  different-model pairs (S-variant) {rate(sd):.2f}")
    for d in D:
        print(f"      S vs {d} {kind[d]:9s} {rate([agree(s, rnd, d, rnd) for s in S]):.2f}")
    print(f"      S vs ref-08 temp1.0 (same model) {rate([agree(s, rnd, 'ref-08', rnd) for s in S if s != 'ref-08']):.2f}")
