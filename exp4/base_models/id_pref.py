"""Which public model is each ep family? Compare a candidate's answer distribution on the exact ep prompts
(number 1-100, animal) with each family's observed answer histogram. 0 API queries.
For each candidate answer string a: p(a) = exp(sum log p(tokens of a + EOS | chat prompt)), renormalised over the set.
Score per family: total-variation distance and log-likelihood of the family's observed answers under the model."""
import sys, json, math, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, ".")
from answer_norm import normalise
NUM = "Pick a random number between 1 and 100. Answer with only the number."
ANI = "Name one random animal. Answer with one word only."
FAM = {"X": [1,12,14,17,18,24,25,33], "Y": [3,7,8,21,22,26,28,30,32], "Z": [5,9,11,13,15,16,19,27,29,31], "W": [2,23], "04": [4], "06": [6], "10": [10], "20": [20]}
fam = {f"ep-{i:02d}": k for k, v in FAM.items() for i in v}
obs = {p: collections.defaultdict(collections.Counter) for p in (NUM, ANI)}
for l in open("ep_answers.jsonl"):
    r = json.loads(l); p = r["prompt"].strip()
    if p in obs: obs[p][fam[r["endpoint"]]][normalise(r["response"], p, "category")] += 1
animals = sorted({a for c in obs[ANI].values() for a in c if a.isalpha()} | {"cat", "dog", "elephant", "lion", "tiger", "giraffe", "penguin", "panda", "kangaroo", "zebra", "fox", "dolphin", "owl", "koala", "wolf", "bear", "rabbit", "monkey", "eagle", "leopard"})
cands = {NUM: [str(i) for i in range(1, 101)], ANI: animals}
def dist(tok, m, prompt, answers, cap):
    pre = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    ids_p = tok(pre, add_special_tokens=False)["input_ids"]; out = {}
    eos = tok.eos_token_id if isinstance(tok.eos_token_id, int) else tok.eos_token_id[0]
    for a in answers:
        for form in ([a, a.capitalize()] if cap else [a]):
            ids_a = tok(form, add_special_tokens=False)["input_ids"] + [eos]
            x = torch.tensor([ids_p + ids_a], device=m.device)
            with torch.no_grad(): lg = torch.log_softmax(m(x).logits[0, :-1].float(), -1)
            s = sum(lg[len(ids_p) - 1 + j, t].item() for j, t in enumerate(ids_a))
            out[a] = max(out.get(a, -1e9), s)
    mx = max(out.values()); z = sum(math.exp(v - mx) for v in out.values())
    return {a: math.exp(v - mx) / z for a, v in out.items()}
for mid in sys.argv[1:]:
    tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    print(f"\n##### {mid}")
    for p, nm, cap in ((NUM, "number", False), (ANI, "animal", True)):
        P = dist(tok, m, p, cands[p], cap)
        top = sorted(P.items(), key=lambda x: -x[1])[:5]
        print(f"  {nm}: model top {[(a, round(v, 2)) for a, v in top]}")
        for f in ["X", "Y", "Z", "W", "04", "06", "10", "20"]:
            c = obs[p][f]; n = sum(v for a, v in c.items() if a in P)
            if not n: continue
            ll = sum(v * math.log(max(P[a], 1e-6)) for a, v in c.items() if a in P) / n
            tv = 0.5 * sum(abs(P.get(a, 0) - c.get(a, 0) / n) for a in set(P) | set(c) if a in P)
            print(f"     {f:3s} family top {c.most_common(3)}  TV {tv:.2f}  mean loglik {ll:.2f}")
    del m; torch.cuda.empty_cache()
