"""RUT-style proxy scoring on existing ref long answers (0 API queries).
Step 1 (--select): mean per-token log-prob of the 5 same-model ref endpoints' answers under each candidate proxy.
Step 2 (--model M): per answer, teacher-forced log-prob and rank of each response token after the first SKIP tokens;
  features per answer: mean log-prob, rank==1 fraction (temperature signal), fraction rank>1000 (tail / LoRA signal).
  Pairwise endpoint distance = permutation z of the difference in per-answer feature means; reports confusion vs truth."""
import argparse, json, math, random, itertools, collections, statistics as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", required=True); ap.add_argument("--select", action="store_true")
ap.add_argument("--skip", type=int, default=15); ap.add_argument("--max-answers", type=int, default=60)
a = ap.parse_args()
FILES = ["../exp1/reference_answers.jsonl", "exp4_answers.jsonl", "ans_translate.jsonl"]
S = ["ref-01", "ref-03", "ref-05", "ref-06", "ref-08"]
truth = {l.split(",")[0]: l.split(",")[2] for l in open("../practice_answers.csv").readlines()[1:]}
data = collections.defaultdict(list)
for f in FILES:
    for l in open(f):
        r = json.loads(l)
        if r["endpoint"].startswith("ref-") and len(r["response"].split()) >= 25: data[r["endpoint"]].append((r["prompt"], r["response"]))
rng = random.Random(0)
for e in data: rng.shuffle(data[e]); data[e] = data[e][:a.max_answers]
eps = sorted(data)
def score(model_id):
    tok = AutoTokenizer.from_pretrained(model_id); m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).eval()
    chat = tok.chat_template is not None
    out = {}
    for e in eps:
        feats = []
        for p, resp in data[e]:
            pre = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) if chat else p + "\n"
            ids_p = tok(pre, add_special_tokens=not chat)["input_ids"]; ids_r = tok(resp, add_special_tokens=False)["input_ids"]
            if len(ids_r) <= a.skip + 5: continue
            ids = torch.tensor([ids_p + ids_r])
            with torch.no_grad(): lg = torch.log_softmax(m(ids).logits[0, :-1].float(), -1)
            tgt = ids[0, 1:]; pos = range(len(ids_p) - 1 + a.skip, len(ids_p) - 1 + len(ids_r))
            lp = [lg[i, tgt[i]].item() for i in pos]
            rk = [int((lg[i] > lg[i, tgt[i]]).sum().item()) + 1 for i in pos]
            feats.append((st.mean(lp), sum(r == 1 for r in rk) / len(rk), sum(r > 1000 for r in rk) / len(rk)))
        out[e] = feats
    return out
def perm_z(x, y, n=500):
    obs = abs(st.mean(x) - st.mean(y)); pool = x + y; k = len(x); null = []
    r = random.Random(1)
    for _ in range(n):
        r.shuffle(pool); null.append(abs(st.mean(pool[:k]) - st.mean(pool[k:])))
    mu, sd = st.mean(null), st.pstdev(null) or 1e-9
    return (obs - mu) / sd
def mcc(groups):
    lab = {e: i for i, g in enumerate(groups) for e in g}; tp = fp = fn = tn = 0
    for x, y in itertools.combinations(eps, 2):
        t = truth[x] == truth[y]; g = lab[x] == lab[y]; tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return (max(0, (tp * tn - fp * fn) / d) if d else 0), tp, fp, fn, tn
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
print("answers per ref endpoint:", {e[-2:]: len(data[e]) for e in eps})
for mid in a.models:
    F = score(mid)
    sl = st.mean(f[0] for e in S for f in F[e])
    print(f"\n### {mid}: mean log-prob on same-model answers (after first {a.skip} tokens) = {sl:.3f}")
    if a.select: continue
    for k, nm in enumerate(["mean logprob", "rank1 frac", "tail frac (rank>1000)"]):
        M = {}
        for x, y in itertools.combinations(eps, 2):
            M[x, y] = M[y, x] = perm_z([f[k] for f in F[x]], [f[k] for f in F[y]])
        print(f"  [{nm}] per endpoint mean: " + " ".join(f"{e[-2:]}={st.mean(f[k] for f in F[e]):.3f}" for e in eps))
        print("     z vs ref-01: " + " ".join(f"{e[-2:]}={M['ref-01', e]:.1f}" for e in eps if e != "ref-01"))
        best = max(((mcc(link(M, t)), t) for t in sorted(set(round(v, 2) for v in M.values()))), key=lambda z: z[0][0])
        (m_, tp, fp, fn, tn), t = best
        print(f"     best (optimistic) threshold {t}: model MCC {m_:.2f} TP {tp} FP {fp} FN {fn} TN {tn} groups {[sorted(x[-2:] for x in g) for g in link(M, t)]}")
