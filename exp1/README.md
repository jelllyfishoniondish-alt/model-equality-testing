# Experiment 1: probe the ref endpoints, compute distances, cluster

Goal: without spending any ep budget, answer three questions.

1. Which ref endpoints are deterministic (greedy decoding)?
2. Which probe prompts best separate "same model, different disguise" from "different model"?
3. How much practice MCC does a simple text distance plus clustering get? This is the **baseline**.

## Files

| File                    | Purpose                                                      |
| ----------------------- | ------------------------------------------------------------ |
| `reference_prompts.txt` | 35 probes, one per line: random number/word, arithmetic, factual QA, verbatim copy, counting, continuation, poems, code, translation, JSON, identity |
| `run_ref.py`            | Sends each probe to ref-01 ... ref-09, N times each, reusing the official `query_reference.py` |
| `analyze_ref.py`        | Determinism check, per-prompt discriminativeness (AUC), distance matrix, hierarchical clustering, writes `my_reference_answer.csv` and scores it |

## How to run (on JURECA)

```bash
# 0) Put the three files in the team folder, next to the official query_reference.py, score_practice.py and practice_answers.csv
cd /p/scratch/training2666/chander2/argos/model_equality

# 1) Activate the environment
source .venv/bin/activate        # with uv: uv pip install requests

# 2) Set the API key (only for this terminal; it is never written to a file)
export ME_API_KEY="Our team API KEY"

# 3) Query: 35 prompts x 9 endpoints x 3 repeats = 945 answers, 4 in parallel, roughly 10-20 minutes
python run_ref.py 3
#    If it stops, run the same command again; answers already collected are skipped.
#    For more repeats, run python run_ref.py 5 later; it only adds repeats 3 and 4.

# 4) Analyse (standard library only; also runs on the Mac if you copy reference_answers.jsonl there)
python analyze_ref.py                    # default metric: mmd
python analyze_ref.py --metric exact     # exact match rate
python analyze_ref.py --metric prefix    # common prefix (good for greedy endpoints)

# 5) Score again with the official script; it should match what analyze_ref.py printed
python score_practice.py my_reference_answer.csv
```

These scripts only send HTTP requests, so they run on the login node: no GPU and no SLURM job needed. This assumes the login node can reach `<competition-server>`; if it can't, run them on the Mac.

## Reading the output

- **Step 1, determinism**: "identical answers on 10/10 prompts" means a greedy endpoint. My guess is that only ref-08 samples; this needs checking.
- **Step 2a, AUC**: one score per prompt. 1.0 means the prompt separates same-model from different-model pairs perfectly; 0.5 means useless; below 0.5 means the disguises dominate. **This is the most valuable output**: it tells you which kinds of prompts to keep and strengthen in the next round.
- **Step 2b, distance matrix and closest pairs**: check how close ref-09 (LoRA 0.3) and ref-02 (fp8) are to ref-01, compared with ref-03, 05, 06 and 08.
- **Step 3, threshold sweep**: the threshold is picked using the answers, so the result is optimistic. The ep endpoints have no answers, so prefer a metric with a wide range of good thresholds over the single highest score.
- `--top N` uses only the N prompts with the highest AUC. It overfits; use it only to explore.

## The three distance metrics

- `exact`: 1 minus the probability that a random answer from each endpoint is identical. Most direct for greedy endpoints, but any small change from a system prompt counts as different.
- `prefix`: the fraction of two answers that is identical from the start. Quantization or a LoRA usually makes the text fork at some position.
- `mmd`: the MMD formula from Gao et al., with character-3-gram cosine similarity as the kernel. It subtracts each endpoint's own random variation, so it is fairer to sampling endpoints.

## Known limitations (to improve in the next round)

- A greedy endpoint outputs the most likely answer; a sampling endpoint outputs a distribution. Even for the same model, the MMD between them is not 0. Possible fix: compare the greedy endpoint's answer with the most frequent answer among the sampling endpoint's repeats.
- System prompts and personas may reword whole answers. Shorter, rigidly formatted prompts (output only a number or a word) should be less affected; check this with the AUC.

# Experiment 2: answer histograms of one-word "random choice" probes

## Why

Experiment 1 (partial run) showed three things:

- **Every ref endpoint samples**; none is greedy. Exact-match and prefix distances are therefore mostly noise (all pairs 0.77-0.97, best MCC 0.33).
- The two best probes by far were "pick a random number between 1 and 100" (AUC 0.97) and "name one random animal" (AUC 0.92). Both have one-word answers.
- Long free-text probes were around AUC 0.4-0.6, i.e. barely useful.

So experiment 2 uses only short one-word probes, asks each one many times, and compares how often each answer appears (the answer histogram). Quantization and LoRA change the model's probabilities, so they should move these histograms. A system prompt or a watermark hopefully moves them less; the per-probe AUC will tell.

## Files

| File               | Purpose                                                      |
| ------------------ | ------------------------------------------------------------ |
| `exp2_prompts.txt` | 20 one-word probes: random numbers (3 ranges), animal, fruit, color, city, country, letter, first name, vegetable, instrument, sport, flower, element, weekday, month, emotion, profession, kitchen object |
| `run_exp2.py`      | Queries ref-01 ... ref-09 with every probe, N repeats, small `max_tokens`; writes `exp2_answers.jsonl` |
| `analyze_exp2.py`  | Normalises answers, builds histograms, per-probe AUC, distance matrix, clustering, `my_exp2_answer.csv`, and a 100-answer budget simulation |

## How to run

```bash
cd /p/scratch/training2666/chander2/argos/model_equality      # next to query_reference.py and practice_answers.csv
source .venv/bin/activate
export ME_API_KEY="your team key"

python run_exp2.py 30          # 20 x 9 x 30 = 5,400 answers, max_tokens 16
python analyze_exp2.py         # metric tv (default)
python analyze_exp2.py --metric tvtemp
python analyze_exp2.py --metric js
python score_practice.py my_exp2_answer.csv

# Later, for a better budget simulation (needs >= 50 answers per probe):
python run_exp2.py 50          # only adds repeats 30..49
```

## Output, step by step

1. **Per-probe AUC**: which probes separate same-model from different-model pairs. Keep the ones near 1.0.
2. **Top answers per endpoint** for the 3 best probes, e.g. `ref-01: 42:9, 37:7, 73:5`. Look at these to understand what actually differs: a different favourite answer, or the same answers with different spread?
3. **Distance matrix and closest pairs**, labelled with the truth.
4. **Clusters** at the best threshold, written to `my_exp2_answer.csv`.
5. **Budget simulation**: on the scored endpoints you have only 100 answers each. The script tries K probes x (100 // K) repeats, re-sampling 30 times, and reports
   - `pair-AUC`: threshold-free separation; near 1.0 means one threshold could split same from different;
   - `best-MCC`: mean and worst case over the re-samples, at the best threshold (an upper bound).
     `best K` picks probes by AUC on this same data, so it is optimistic; `first K` (file order) is honest. If `R` is larger than the answers you collected, the line says so: run more repeats.

### Validation outputs (added after the first design)

- **Step 1** also prints a 95% endpoint-bootstrap interval for each probe's AUC, and `S(q)` = median distance of different-model pairs minus median of same-model pairs. With only 9 endpoints, a probe with AUC 0.9 but an interval of [0.6, 1.0] is not yet trustworthy.
- **Step 4** prints a bootstrap interval for the MCC at the chosen threshold.
- **Step 5, split-half validation**: picks the top-`k` probes (`--k`, default 5) and the threshold on the even repeats, then scores on the odd repeats, and the other way round. **This is the most honest number in the output.** A big drop from the selection half to the held-out half means overfitting.
- **Step 6, disguise diagnostic**: for every endpoint, its distance to ref-01 compared with the noise floor (how far each endpoint is from itself between even and odd repeats). Same-model endpoints (watermark, system prompt, persona, temperature) should have excess near 0, and fp8/LoRA clearly above. `Margin > 0` means the disguises are separable from the real changes.
- **Step 7** is the budget simulation described above.

## Metrics

- `tv`: total variation distance between the two histograms (0 = identical, 1 = no overlap).
- `js`: Jensen-Shannon divergence; similar, less sensitive to rare answers.
- `tvtemp`: temperature-tolerant TV. Decoding temperature does not change model identity but sharpens or flattens a histogram. This metric re-sharpens one side with several exponents and keeps the smallest TV, so a pure temperature difference costs little, while a change in which answers are likely (quantization, LoRA) remains.

## Things to watch

- **Watermark trap**: a Kirchenbauer-style watermark adds a fixed bias to a "green list" of tokens. For one-word answers this may shift the histogram consistently, so ref-06 could look like a different model. Check where ref-06 lands.
- **Persona/system prompt**: may change favourite answers (e.g. a pirate persona picks "parrot"). Check ref-03 and ref-05.
- **Normalisation** keeps only the first word in lowercase without punctuation. If an endpoint answers "Sure! 42", that becomes "sure". If you see that in step 2, improve `normalise()`.
- The ep endpoints may contain several families; the family threshold is not tuned here because ref has only one family.

# Experiment 3: new probe candidates, distance choices and an ep top-up plan

Runs on the **ref endpoints only** (unlimited, never scored). Start it **after the ep run has finished**, because the server allows at most 4 parallel requests per team and both runs share that limit.

Adopted from the SPEC: new short-answer probes (number 1-10, letter A-Z, color, Chinese number 1-100), a FLIPS-style binary sequence, split-half stability, invalid rates, probe-combination ablation, complete vs average linkage, Dirichlet smoothing and number binning, and adaptive top-up planning. Long text, watermark detection and temperature settings are skipped: they do not fit the rules or the budget.

## Files

| File               | Purpose                                                      |
| ------------------ | ------------------------------------------------------------ |
| `exp3_prompts.txt` | 7 probes: 0 number 1-100 and 1 animal (baselines, the exact ep texts), 2 number 1-10, 3 letter, 4 color, 5 Chinese number 1-100, 6 binary sequence of length 100 |
| `run_exp3.py`      | Queries ref-01 ... ref-09. Short probes use max_tokens 16 and the binary probe 200. `--repeats` (default 30), `--parallel` (default 4, max 4), `--skip-binary`. Resumable; writes `exp3_answers.jsonl`. |
| `answer_norm.py`   | exp2 normaliser plus Chinese number prompts and binary strings. Backward compatible for the English probes. |
| `analyze_exp3.py`  | Sections 0-4 below                                           |
| `plan_topup.py`    | For ep: lists pairs whose distance interval straddles the model threshold and suggests per-endpoint extra answers. **Prints only; never queries.** |

## Run

```bash
cd /p/scratch/training2666/chander2/argos/model_equality      # or a folder linking query_reference.py and practice_answers.csv
source .venv/bin/activate
export ME_API_KEY="your team key"

python run_exp3.py --repeats 30            # 7 x 9 x 30 = 1,890 answers
python analyze_exp3.py                     # dist js; also try --dist l2u, --dist tv, --alpha 0.5, --bins 10
```

**Runtime.** At the roughly 15 answers per minute seen so far:

| Run                          | Answers | Time        |
| ---------------------------- | ------- | ----------- |
| `--repeats 30`               | 1,890   | about 2 h   |
| `--repeats 20`               | 1,260   | about 1.4 h |
| `--repeats 20 --skip-binary` | 1,080   | about 1.2 h |

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