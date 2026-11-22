# Experiment 3: new probe candidates, distance choices and an ep top-up plan

Runs on the **ref endpoints only** (unlimited, never scored). Start it **after the ep run has finished**, because the server allows at most 4 parallel requests per team and both runs share that limit.

Adopted from the SPEC: new short-answer probes (number 1-10, letter A-Z, color, Chinese number 1-100), a FLIPS-style binary sequence, split-half stability, invalid rates, probe-combination ablation, complete vs average linkage, Dirichlet smoothing and number binning, and adaptive top-up planning. Long text, watermark detection and temperature settings are skipped: they do not fit the rules or the budget.

## Files

| File | Purpose |
| --- | --- |
| `exp3_prompts.txt` | 7 probes: 0 number 1-100 and 1 animal (baselines, the exact ep texts), 2 number 1-10, 3 letter, 4 color, 5 Chinese number 1-100, 6 binary sequence of length 100 |
| `run_exp3.py` | Queries ref-01 ... ref-09. Short probes use max_tokens 16 and the binary probe 200. `--repeats` (default 30), `--parallel` (default 4, max 4), `--skip-binary`. Resumable; writes `exp3_answers.jsonl`. |
| `answer_norm.py` | exp2 normaliser plus Chinese number prompts and binary strings. Backward compatible for the English probes. |
| `analyze_exp3.py` | Sections 0-4 below |
| `plan_topup.py` | For ep: lists pairs whose distance interval straddles the model threshold and suggests per-endpoint extra answers. **Prints only; never queries.** |

## Run

```bash
cd /p/scratch/training2666/chander2/argos/model_equality      # or a folder linking query_reference.py and practice_answers.csv
source .venv/bin/activate
export ME_API_KEY="your team key"

python run_exp3.py --repeats 30            # 7 x 9 x 30 = 1,890 answers
python analyze_exp3.py                     # dist js; also try --dist l2u, --dist tv, --alpha 0.5, --bins 10
```

**Runtime.** At the roughly 15 answers per minute seen so far:

| Run | Answers | Time |
| --- | --- | --- |
| `--repeats 30` | 1,890 | about 2 h |
| `--repeats 20` | 1,260 | about 1.4 h |
| `--repeats 20 --skip-binary` | 1,080 | about 1.2 h |

The analysis takes a few seconds. Repeats are queried round-robin, so you can stop early and analyse what is there; re-running later fills the gaps.

## analyze_exp3.py output

- **0. Valid-answer rate** per probe × endpoint. If a probe is mostly invalid on some endpoints, its distances are about format, not about the model.
- **1. Per-probe discrimination.** Shows, for each probe:
  - AUC with a 95% endpoint-bootstrap interval;
  - S(q) = median different-model distance minus median same-model distance;
  - AUC on the even and odd halves;
  - stability: the Spearman correlation of all pair distances between the two halves. Near 1 means the probe measures something real; near 0 means noise.
- **2. Disguise robustness.** Each endpoint's distance to ref-01 per probe. `s` columns (same model: system prompt, persona, watermark, temperature 1.0) should be small and `d` columns (fp8, LoRA 0.3/0.7/1.5) large. Check ref-08 and ref-03/05 for the binary probe in particular.
- **3. Probe combinations**, for average- and complete-linkage, all scored by **held-out MCC**: the threshold is chosen on one half of the repeats and scored on the other.
  - A = number 1-100; B = A + animal (the current ep set); C = B + each new probe.
  - A greedy path that keeps adding probes while held-out MCC improves. It is optimistic, because it also chooses probes on held-out MCC.
  - Combining probes: each probe's distance matrix is divided by its median over all pairs (no labels needed), then averaged.
- **4. Distance choices on B**: TV, JS, JS with α = 0.5, Hellinger with α = 0.5, l2u, and the binned versions.

With only 9 ref endpoints, MCC differences below about 0.1 are noise. Prefer choices that are good under both linkages and both halves.

**Binary probe distance:** each response gives 20 features: balance, runs, mean/max run, P(0→1), P(1→0), 2- and 3-gram frequencies, lag-1..3 autocorrelation and length. Endpoint distance = RMS over features of (mean difference / standard error). It is about 1 for two samples from the same distribution and grows with real differences.

## plan_topup.py (after cluster_ep.py has produced a threshold)

```bash
# same folder as ep_answers.jsonl, ep_probes.txt and this answer_norm.py
python plan_topup.py --threshold <model threshold printed by cluster_ep.py> --metric l2u --family-threshold <family threshold>
python plan_topup.py --threshold 0.61 --metric tv --n-match 8 ...      # if cluster_ep used tv with n_match 8
```

It estimates each candidate pair's uncertainty by half-sampling: half of each endpoint's answers, 100 times. It lists the pairs whose 90% interval contains the threshold and the factor of extra answers each would need. It then suggests extra answers per endpoint, capped at `--remaining` (default 60), and prints the `ep_collect.py --repeats N --endpoints ...` commands that would carry the plan out. `ep_collect.py` now accepts `--endpoints` and still asks for "yes" before spending.

Pairs that would need more than the whole remaining budget are better served by adding a probe that separates them (from section 3) than by more repeats.
