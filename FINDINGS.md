# Task 2 (Model Equality Testing): what we tried and what we ruled out

Night of 21–22 Nov 2026. Team argos. Best public score: **0.10** (v1).

## 1. The task and the score

- 33 scored endpoints `ep-01 … ep-33`, 100 answers per endpoint for the whole competition, at most 4 requests at a time per team.
- 9 practice endpoints `ref-01 … ref-09` with known answers (unlimited, unscored). They serve a separate small model, which identifies itself as **SmolLM**:
  one plain model (`ref-01`), four disguises of the same model (system prompt `ref-03`, persona `ref-05`, watermark `ref-06`, temperature 1.0 `ref-08`)
  and four different models (fp8 `ref-02`, LoRA 0.3 `ref-09`, LoRA 0.7 `ref-04`, LoRA 1.5 `ref-07`).
- Score = **0.1 × family MCC** (all pairs) + **0.9 × model MCC** (pairs inside one family); negative MCC counts as 0. The leaderboard moves in steps of 0.05.

## 2. What we know about ep

- Families are right: v1 scores exactly 0.10 (family part 1.0, model part 0).
  Families: X = {01,12,14,17,18,24,25,33}, Y = {03,07,08,21,22,26,28,30,32}, Z = {05,09,11,13,15,16,19,27,29,31}, plus {02,23}, 04, 06, 10, 20.
- Every ep endpoint **samples** (no greedy decoding): no two endpoints ever gave identical long answers, and no endpoint repeats itself on short prompts.
- The server returns **no usable metadata**: no token counts, no logprobs, `created` = 0, identical headers, a fixed ~6.7 s latency floor.

## 3. Submissions (all public score 0.10)

| File | Idea | Public |
|---|---|---|
| v1 | l2u clustering, one model per family | 0.10 |
| v2 | Z: {05,16} vs rest | 0.10 |
| v3 | Y: 28 alone; Z: {05,16}, {13}, rest | 0.10 |
| v4 | most central half of each family = one model, rest singletons | 0.10 |
| gt/s28 | only ep-28 split off | 0.10 |
| gt/v6 | complete linkage z ≤ 1.0 inside families (several models per family) | 0.10 |
| v6c | mutual-nearest pairs inside families | 0.10 |

Why splits never help: every apparent outlier has a near-identical partner in its family (13≈09, 05≈16, 28≈21, 07≈22),
so any split also breaks real same-model pairs. No endpoint differs from **every** member of its family.

## 4. Probes tried on ref, and why they were ruled out

| Probe | Result on ref | Verdict |
|---|---|---|
| Number 1–100 and animal histograms (exp1–3) | With ~18 answers, number-only reached model MCC 0.69 (optimistic). **With 60–78 answers it falls to ~0.4 (held-out 0.16–0.44)**: disguises (system prompt, temperature) really shift the distribution, so same-model z grows with n as fast as variant z. fp8 never separates. | Ruled out |
| Animal question | Separates families, adds noise at model level | Ruled out for models |
| Open-ended answers, style features | fp8 / LoRA 0.3 not separable; watermark differed most | Ruled out |
| Translation of a fixed paragraph | Ref model cannot translate | Untestable |
| FLIPS-style 0/1 sequences | Disguises vary more than variants | Ruled out |
| Many prompts × 1 answer (categories) | Same-model and cross-model agreement equal (0.12–0.16 vs 0.14–0.17) | Ruled out |
| Fixed-choice "choose one of …" (12 prompts × 3 rounds) | Promising on one batch, did not replicate; held-out AUC 0.52–0.59 | Ruled out |
| Anchored multiple choice (27 questions, 3 difficulty tiers) | Ref model guesses with a strong "A" position bias; same-model versions never agree | Untestable on ref |
| 2-shot completions | Server is chat-only; model answers conversationally | Ruled out |
| Proxy-model log-prob (RUT) with SmolLM2 135M/360M/1.7B | Only temperature 1.0 stands out; LoRA not separated (MCC 0.00–0.17) | Ruled out |
| **Length-constraint overrun** ("exactly N sentences") | **Replicated out of sample**: overrun rate same-model 0.05–0.15, LoRA 0.7 0.25, LoRA 1.5 0.75; fp8 0.00, LoRA 0.3 0.10. "High-information" prompts made everyone overrun. | Works only for strong LoRA |

## 5. Probes tried on ep (budget spent)

| Probe | ep queries | Result |
|---|---|---|
| 6 multiple-choice questions on ep-05/16/09/13 | 24 | ep model also mostly guesses (0–2/6 correct); no consistent split |
| Overrun, 3 validated prompts × 33 endpoints | 99 | **0 endpoints flagged.** Z: every answer exactly right; X writes fewer, long sentences. |
| Offline on those 99 answers | 0 | Repetition: only **ep-28** degenerates ("a bustling place, bustling with hustle and bustle…"), 10× above anyone else. TF-IDF embedding: no outliers in any family. |

## 6. Budget left (06:30, 22 Nov)

44 answers on 29 endpoints, 38 on ep-05/16/09/13 (all spent by 10:05; see section 8).

## 7. Take-aways

1. Distribution tests on short answers detect **disguises** (system prompt, temperature) as differences; more samples make this worse, not better.
2. fp8 was never separable on ref by any probe we tried.
3. The only signal that survived out-of-sample validation (length overrun) catches strong LoRA on the small ref model, but the larger ep model follows length limits and showed nothing.
4. ep-28 is the one endpoint with visible degeneration, but splitting it alone (s28) did not change the public score.

## 8. Morning of 22 Nov: the ep families are public models

Comparing each candidate model's answer distribution on the exact ep prompts ("number 1–100", "animal", computed locally, 0 queries)
with each family's observed answers (`exp4/base_models/id_pref.py`, logs `id_pref*.log`):

| Family | Base model | TV number | TV animal | Model's favourites = family's favourites |
|---|---|---|---|---|
| X | **Qwen2.5-3B-Instruct** | 0.16 | 0.18 | 27, 37, 87 / penguin, kangaroo, leopard |
| Y | **Qwen2.5-1.5B-Instruct** | 0.15 | 0.14 | 73, 67, 74 / lion, monkey, fox |
| Z | **Llama-3.2-1B-Instruct** | 0.24 | 0.17 | 14, 73 / giraffe, eagle |

Ruled out: Qwen2.5-0.5B, Phi-3.5-mini, SmolLM2-360M/1.7B, Llama-3.2-3B, gemma-2-2b. The ref endpoints are SmolLM2-360M-Instruct.
Per-endpoint fits (`id_endpoint.py`) put ep-06 and ep-20 with Qwen2.5-1.5B, ep-04 / ep-02 / ep-23 with Qwen2.5-3B, ep-10 with Llama-3.2-1B.

**Null range for disguises** (`id_null.py`, `id_null2*.py`): each base model was run locally with system prompts / personas and
temperatures 0.7–1.2, sampled at the ep sample size, to get how low a same-model endpoint's log-likelihood can go.
With 20 answers only ep-02 (animal) fell below it; after spending the last 570 ep queries on the number prompt
(`exp4/home_me_work/ep_num_final.jsonl`, ~36 number answers per endpoint) **no endpoint falls below the range**.
So the available data cannot separate fp8 / LoRA variants from disguises, consistent with every ref test.

Late submissions (all public 0.10, files in `ep1/queue/final/`):

| File | Idea |
|---|---|
| s08, s05 | single splits |
| v7 | outliers merged into their base-model family (X+04+{02,23}, Y+06+20, Z+10), each its own model |
| v9 | v7 + ep-02 split from ep-23 (the only endpoint below the disguise range at 20 answers) |
| v10 | v7 + ep-30 alone, {05,16} together (lottery, no evidence) |
| v8 (11:03) | v7 + worst base-model fits split off: {05,16}, 31, 30, 07, 33, 14 |
| v11 | v7 + in each family the worse-fitting half as singletons — prepared, **not submitted** |

ep budget at the end: 0 on every endpoint. Final best public score: **0.10**.

## 9. Retrospective: what a 1.00 team did differently

Summary of the publicly shared approach of team Fable 4.2 (https://github.com/AnMaLeNo/Fable-4.2), as we understand it;
their code is not copied here.

| Step | Their approach | Ours |
|---|---|---|
| Base models | Same three as we found (Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-1B) | Found the same, but only on the last morning |
| Reference point | Every endpoint is compared with a local **replica** of its base model, which gives the exact probability of each answer (endpoints sample at temperature 0.7) | We mostly compared endpoints with each other, so disguises looked like differences |
| fp8 | Build a local **fp8 (E4M3) replica** and use the likelihood ratio replica-fp8 vs replica-bf16 per endpoint; on ref only the real fp8 endpoint prefers the fp8 replica | We tried distribution tests, which cannot see a change this small |
| LoRA | **Fit the temperature first** (absorbs the temperature disguise), then look at per-prompt residuals (χ²) against the replica; what still deviates is a LoRA | Our "null range" simulated disguises but never fitted temperature per endpoint |
| Decision rule | Merge into the base group **only** what is positively identified as a disguise (temperature, watermark, system prompt); everything else is a different model | The opposite: "fits the base model ⇒ same model", so we never split anything |
| Budget | ~40 one-word questions in one request; 35–70 queries per endpoint | One question per request; ~100 queries per endpoint |
| Submissions | Each submission changes exactly one thing, used as an experiment; 0.1 → 1.0 | Several changes per submission early on, so scores were hard to attribute |

Lessons: (1) identify the base model first and compare against a local replica, not endpoint vs endpoint;
(2) model the known disguises explicitly (temperature fit, system prompt) before calling anything a difference;
(3) for fp8, quantize the replica yourself and use a likelihood ratio; (4) default to "different model" unless a disguise is identified;
(5) pack many questions per request to save budget; (6) one change per submission.

## Where things are

- `ep1/` clustering, submissions (`queue/`), score log (`queue/scores.log`)
- `exp1/`–`exp3/` first probes on ref
- `exp4/` all experiments of the night; `exp4/README.md` lists each with files; `exp4/home_me_work/` was run from `$HOME` after `/p/scratch/training2666` hit its project inode limit
- The API key is read from `ME_API_KEY`; nothing secret is committed.
