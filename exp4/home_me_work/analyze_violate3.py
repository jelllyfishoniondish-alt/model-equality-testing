"""Pre-registered metrics only: overrun rate (count > required) and mean overrun (count - required, as is).
Sentence prompts: count sentences of >=3 words. Tips prompt: count numbered/bulleted items; if none, count sentences."""
import json, re, collections, statistics as st, sys
REQ = {"0": ("sent", 3), "1": ("sent", 2), "2": ("sent", 2)}
S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]; V = ["ref-02", "ref-09", "ref-04", "ref-07"]
name = {"ref-01": "plain", "ref-03": "sysprompt", "ref-05": "persona", "ref-06": "watermark", "ref-08": "temp1.0",
        "ref-02": "fp8", "ref-09": "LoRA0.3", "ref-04": "LoRA0.7", "ref-07": "LoRA1.5"}
ABBR = r"\b(U\.S|U\.K|Mr|Mrs|Ms|Dr|St|vs|e\.g|i\.e|etc|approx|No)\."
def sents(t):
    t = re.sub(ABBR, lambda m: m.group(0).replace(".", "<DOT>"), t.strip())
    return len([s for s in re.split(r"(?<=[.!?])\s+", t) if len(s.split()) >= 3])
def items(t):
    n = len(re.findall(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+\S", t)); return n if n else sents(t)
D = collections.defaultdict(list); P = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(sys.argv[1] if len(sys.argv) > 1 else "ref_violate.jsonl"):
    r = json.loads(l); kind, want = REQ[r["id"]]; c = sents(r["response"]) if kind == "sent" else items(r["response"])
    D[r["endpoint"]].append(c - want); P[r["id"]][r["endpoint"]].append(c - want)
res = {}
for e in S + V:
    v = D[e]; res[e] = (sum(x > 0 for x in v) / len(v), st.mean(v), len(v))
    print(f"{name[e]:9s} overrun rate {res[e][0]:.2f}  mean overrun {res[e][1]:+.2f}  n={res[e][2]}")
for pid in sorted(P): print(f"  prompt {pid}: " + "  ".join(f"{name[e][:6]}:{P[pid][e]}" for e in S + V))
smax = max(res[e][0] for e in S)
print("\nmax same-model overrun rate:", round(smax, 2))
for v in V: print(f"  {name[v]:8s} rate {res[v][0]:.2f} -> {'ABOVE all same-model' if res[v][0] > smax else 'not separated'}")
print("PASS (LoRA0.7 and LoRA1.5 above every same-model version):", res["ref-04"][0] > smax and res["ref-07"][0] > smax)
