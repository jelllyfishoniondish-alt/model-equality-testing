"""Anchor questions. Stage 1: answers -> option text; per question: S consensus, per-variant flips.
LOCKED selection rule (stage 1): a question qualifies if >=4 of the 5 S endpoints give a valid answer and all valid
S answers are the same option; it is ranked by how many of the 4 variants give a different valid option
(ties: fewer invalid answers). Keep the top 9 with at least 1 differing variant.
Stage 2 (pass 'stage2'): per selected question and variant (verbatim/choice/shuffled), compare with S consensus."""
import json, re, sys, collections
S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]; V = ["ref-02", "ref-09", "ref-04", "ref-07"]
name = {"ref-01": "plain", "ref-03": "sysprompt", "ref-05": "persona", "ref-06": "watermark", "ref-08": "temp1.0",
        "ref-02": "fp8", "ref-09": "LoRA0.3", "ref-04": "LoRA0.7", "ref-07": "LoRA1.5"}
items = json.load(open("anchor_items.json"))
def letter(t):
    m = re.match(r"\s*\**\(?([A-Da-d])\b[\).:]?", t.strip())
    return m.group(1).upper() if m else None
if len(sys.argv) < 2 or sys.argv[1] != "stage2":
    A = {}
    for l in open("ref_stage1.jsonl"):
        r = json.loads(l); qi = int(r["id"]); L = letter(r["response"])
        A[(qi, r["endpoint"])] = items[qi]["opts"]["ABCD".index(L)] if L and "ABCD".index(L) < len(items[qi]["opts"]) else None
    rows = []
    by_tier = collections.defaultdict(list)
    for qi, it in enumerate(items):
        s = [A.get((qi, e)) for e in S]; sv = [x for x in s if x]
        cons = sv[0] if sv and len(set(sv)) == 1 and len(sv) >= 4 else None
        flips = [v for v in V if cons and A.get((qi, v)) and A[(qi, v)] != cons]
        inval = sum(A.get((qi, e)) is None for e in S + V)
        by_tier[it["tier"]].append((qi, len(sv), len(set(sv)) if sv else 0, cons, flips, inval))
        if cons and flips: rows.append((-len(flips), inval, qi))
        print(f"[{qi:2d}] {it['tier']:6s} S={['-' if x is None else x for x in s]} -> consensus {cons!r:12} | " +
              " ".join(f"{name[v]}={A.get((qi, v))}" for v in V) + f" | flips {[name[v] for v in flips]}")
    print("\nper tier: S valid rate, S unanimous rate, questions with >=1 variant flip (given consensus)")
    for t, L in by_tier.items():
        print(f"  {t:6s} S valid {sum(x[1] for x in L)/(5*len(L)):.2f}  S unanimous {sum(1 for x in L if x[3])}/{len(L)}  flips {sum(1 for x in L if x[4])}/{len(L)}")
    sel = [qi for _, _, qi in sorted(rows)][:9]
    print("\nSELECTED (locked rule):", sel)
    json.dump(sel, open("stage1_selected.json", "w"))
else:
    mp = json.load(open("stage2_map.json")); cons = {}
    A1 = {}
    for l in open("ref_stage1.jsonl"):
        r = json.loads(l); qi = int(r["id"]); L = letter(r["response"])
        A1[(qi, r["endpoint"])] = items[qi]["opts"]["ABCD".index(L)] if L else None
    res = collections.defaultdict(list)
    for l in open("ref_stage2.jsonl"):
        r = json.loads(l); m = mp[int(r["id"])]; L = letter(r["response"])
        res[(m["q"], r["endpoint"])].append((m["variant"], m["letters"].get(L) if L else None))
    sel = sorted({m["q"] for m in mp})
    point = collections.Counter(); dis = collections.Counter()
    for qi in sel:
        sv = [a for e in S for _, a in res[(qi, e)] if a]
        c = collections.Counter(sv).most_common(1)[0][0] if sv else None
        print(f"[{qi:2d}] S modal {c!r} ({collections.Counter(sv).most_common(3)})")
        for e in S + V:
            ans = res[(qi, e)]; diff = sum(1 for _, a in ans if a and a != c); val = sum(1 for _, a in ans if a)
            print(f"     {name[e]:9s} {[(v[:4], a) for v, a in ans]}  differ {diff}/{val}")
            if e in V and val and diff >= 2: point[name[e]] += 1
            if e in S and val and diff >= 2: dis[name[e]] += 1
    print("\nquestions where variant differs in >=2 of its 3 new answers:", dict(point))
    print("same for disguises (should be ~0):", dict(dis))
    print("GO rule: some variant with >=2 questions AND disguises clean ->",
          any(n >= 2 for n in point.values()) and sum(dis.values()) == 0)
