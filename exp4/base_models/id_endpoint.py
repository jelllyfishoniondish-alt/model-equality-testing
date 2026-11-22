"""Per-endpoint fit to a known base model: (a) mean log-lik of the endpoint's observed number/animal answers
under the base model's answer distribution; (b) mean per-token log-prob of its long answers (ep_orig.jsonl)
under the base model (teacher forcing, full reply). Usage: id_endpoint.py <model> <endpoints...>"""
import sys, json, math, collections, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, ".")
from answer_norm import normalise
mid, eps = sys.argv[1], sys.argv[2:]
NUM = "Pick a random number between 1 and 100. Answer with only the number."
ANI = "Name one random animal. Answer with one word only."
tok = AutoTokenizer.from_pretrained(mid); m = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.bfloat16, device_map="cuda").eval()
eos = tok.eos_token_id if isinstance(tok.eos_token_id, int) else tok.eos_token_id[0]
def pre(p): return tok(tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True), add_special_tokens=False)["input_ids"]
def seqlp(ids_p, ids_a):
    x = torch.tensor([ids_p + ids_a], device=m.device)
    with torch.no_grad(): lg = torch.log_softmax(m(x).logits[0, :-1].float(), -1)
    return [lg[len(ids_p) - 1 + j, t].item() for j, t in enumerate(ids_a)]
obs = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open("ep_answers.jsonl"):
    r = json.loads(l); p = r["prompt"].strip()
    if r["endpoint"] in eps and p in (NUM, ANI): obs[r["endpoint"]][p].append(r["response"].strip())
longs = collections.defaultdict(list)
for l in open("ep_orig.jsonl"):
    r = json.loads(l)
    if r["endpoint"] in eps: longs[r["endpoint"]].append((r["prompt"], r["response"]))
cache = {}
def raw_lp(p, a):
    k = (p, a)
    if k not in cache: cache[k] = sum(seqlp(pre(p), tok(a, add_special_tokens=False)["input_ids"] + [eos]))
    return cache[k]
print(f"model {mid}")
print("endpoint  num_loglik  ani_loglik  long_tok_lp  long_tok_lp(after 15)")
for e in eps:
    nl = sum(raw_lp(NUM, a) for a in obs[e][NUM]) / max(1, len(obs[e][NUM]))
    al = sum(raw_lp(ANI, a) for a in obs[e][ANI]) / max(1, len(obs[e][ANI]))
    toks = []; toks15 = []
    for p, t in longs[e]:
        lp = seqlp(pre(p), tok(t, add_special_tokens=False)["input_ids"]); toks += lp; toks15 += lp[15:]
    print(f"{e}  {nl:8.2f}  {al:8.2f}  {sum(toks)/max(1,len(toks)):8.3f}  {sum(toks15)/max(1,len(toks15)):8.3f}")
