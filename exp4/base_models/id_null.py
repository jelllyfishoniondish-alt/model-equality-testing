"""Null range for 'same model, disguised': simulate the base model under system prompts / personas / temperatures,
draw ep-sized samples (20 answers) many times, and record the mean log-lik under the plain base distribution.
Then score each real endpoint and flag those below the null 2.5% quantile on number OR animal.
Usage: id_null.py <model> <endpoints...>"""
import sys, json, math, random, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, ".")
from answer_norm import normalise
mid, eps = sys.argv[1], sys.argv[2:]
NUM = "Pick a random number between 1 and 100. Answer with only the number."
ANI = "Name one random animal. Answer with one word only."
SYS = [None, "You are a helpful assistant.", "You are a concise assistant. Keep answers short.",
       "You are Captain Jack, a cheerful pirate who loves the sea.", "You are a wise old librarian who speaks calmly.",
       "You are a customer support agent for an online shop."]
TEMPS = [0.7, 0.85, 1.0, 1.2]
tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
eos = tok.eos_token_id if isinstance(tok.eos_token_id, int) else tok.eos_token_id[0]
obs = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open("ep_answers.jsonl"):
    r = json.loads(l); p = r["prompt"].strip()
    if p in (NUM, ANI): obs[r["endpoint"]][p].append(normalise(r["response"], p, "category"))
animals = sorted({a for e in obs for a in obs[e][ANI] if a.isalpha()} | {"cat", "dog", "elephant", "lion", "tiger", "giraffe", "penguin", "panda", "kangaroo", "zebra", "fox", "dolphin", "owl", "koala", "wolf", "bear", "rabbit", "monkey", "eagle", "leopard"})
cands = {NUM: [str(i) for i in range(1, 101)], ANI: animals}
def logits(prompt, sysmsg, answers, cap):
    msgs = ([{"role": "system", "content": sysmsg}] if sysmsg else []) + [{"role": "user", "content": prompt}]
    ids_p = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    out = {}
    for a in answers:
        for form in ([a, a.capitalize()] if cap else [a]):
            ids_a = tok(form, add_special_tokens=False)["input_ids"] + [eos]
            x = torch.tensor([ids_p + ids_a], device=m.device)
            with torch.no_grad(): lg = torch.log_softmax(m(x).logits[0, :-1].float(), -1)
            s = sum(lg[len(ids_p) - 1 + j, t].item() for j, t in enumerate(ids_a)); out[a] = max(out.get(a, -1e9), s)
    return out
def norm(lp, T=1.0):
    mx = max(v / T for v in lp.values()); z = sum(math.exp(v / T - mx) for v in lp.values())
    return {a: math.exp(v / T - mx) / z for a, v in lp.items()}
rng = random.Random(0); res = {}
for p, cap, nm in ((NUM, False, "num"), (ANI, True, "ani")):
    base = norm(logits(p, None, cands[p], cap)); null = []
    for s in SYS:
        lp = logits(p, s, cands[p], cap)
        for T in TEMPS:
            q = norm(lp, T); keys = list(q); w = [q[k] for k in keys]
            for _ in range(200):
                smp = rng.choices(keys, w, k=20); null.append(sum(math.log(max(base[a], 1e-6)) for a in smp) / 20)
    null.sort(); lo = null[int(0.025 * len(null))]; lo1 = null[int(0.01 * len(null))]
    print(f"[{nm}] null mean log-lik: median {null[len(null)//2]:.2f}, 2.5% {lo:.2f}, 1% {lo1:.2f}, min {null[0]:.2f}")
    for e in eps:
        a = [x for x in obs[e][p] if x in base]
        ll = sum(math.log(max(base[x], 1e-6)) for x in a) / max(1, len(a))
        res.setdefault(e, {})[nm] = (ll, ll < lo, len(obs[e][p]) - len(a))
print("endpoint  num(ll, below)  ani(ll, below)  -> flag")
for e in eps:
    n, a = res[e]["num"], res[e]["ani"]
    print(f"{e}  {n[0]:6.2f} {'*' if n[1] else ' '}  {a[0]:6.2f} {'*' if a[1] else ' '}   {'SPLIT' if (n[1] or a[1]) else ''}  (unparsed {n[2]}/{a[2]})")
