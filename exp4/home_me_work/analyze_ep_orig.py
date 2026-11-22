"""ep overrun: sentences (>=3 words, abbreviations protected) minus required, per endpoint and family; flags per the user's rule."""
import json, re, collections, statistics as st
REQ = {"0": 3, "1": 2, "2": 3}
FAM = {"X": [1,12,14,17,18,24,25,33], "Y": [3,7,8,21,22,26,28,30,32], "Z": [5,9,11,13,15,16,19,27,29,31], "W": [2,23], "04": [4], "06": [6], "10": [10], "20": [20]}
ABBR = r"\b(U\.S|U\.K|Mr|Mrs|Ms|Dr|St|vs|e\.g|i\.e|etc|approx|No)\."
def sents(t):
    t = re.sub(ABBR, lambda m: m.group(0).replace(".", "<DOT>"), t.strip())
    return len([s for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= 3])
A = collections.defaultdict(dict); raw = collections.defaultdict(dict)
for l in open("ep_orig.jsonl"):
    r = json.loads(l); A[r["endpoint"]][r["id"]] = sents(r["response"]) - REQ[r["id"]]; raw[r["endpoint"]][r["id"]] = r["response"]
flag = []
for k, v in FAM.items():
    eps = [f"ep-{i:02d}" for i in v]
    print(f"== family {k}")
    for e in eps:
        o = [A[e].get(p) for p in "012"]; ov = [x for x in o if x is not None]
        mean = st.mean(ov) if ov else float("nan"); n_over = sum(x > 0 for x in ov)
        mates = [st.mean([A[m][p] for p in A[m]]) for m in eps if m != e and A[m]]
        near0 = sum(abs(x) < 1 for x in mates) >= max(1, len(mates) / 2) if mates else False
        f = n_over >= 2 and mean >= 2 and near0
        if f: flag.append(e)
        print(f"   {e} overrun per prompt {o}  mean {mean:+.2f}  over {n_over}/3 {'<- FLAG' if f else ''}")
print("\nflagged:", flag)
for e in flag:
    print(f"\n##### {e}")
    for p in "012": print(f"--- prompt {p}: {raw[e].get(p, '')[:600]!r}")
