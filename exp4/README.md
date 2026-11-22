# exp4: new signals tested on the ref endpoints (night of 21-22 Nov)

All runs here query **ref only** (unlimited, unscored) unless stated. Answers keep the raw response plus metadata.

| What | Files | Result |
|---|---|---|
| Response metadata (latency, usage tokens, headers, logprobs) | `meta_query.py`, `run_meta.py`, `meta_answers.jsonl` | Nothing usable: no usage/logprobs, `created`=0, identical headers, fixed ~6.72 s latency floor on every endpoint. |
| Open-ended answers, style features | `exp4_prompts.txt`, `exp4_answers.jsonl`, `analyze_exp4.py` | ~6 answers/prompt: fp8 and LoRA 0.3 not separable; the watermark disguise differed most. |
| Translation of a fixed paragraph | `p_translate.txt`, `ans_translate.jsonl`, `analyze_exp5.py` | Untestable on ref: the small ref model cannot translate (self-similarity 0.14-0.30). |
| FLIPS-style 0/1 sequences | `p_flips.txt`, `ans_flips.jsonl`, `analyze_exp5.py` | fp8 / LoRA not separable (score -0.7 to -1.0); disguises vary more. |
| Many prompts x 1 answer (category prompts) | `p_many15.txt`, `ref_many.jsonl`, `analyze_refrep.py`, `analyze_agree.py` | Same-model and different-model agreement equal (0.12-0.16 vs 0.14-0.17). |
| Four prompt types (number/symbol, category, fill-in, fixed choice) | `p_four.txt`, `ref_four.jsonl`, `analyze_perprompt.py` | Fixed-choice "north/south/east/west" looked strong; fill-ins invalid on ref. |
| Validation batch: 12 fixed-choice + number + random/favourite colour, 3 rounds | `p_choice.txt`, `ref_choice.jsonl`, `analyze_choice.py` | Did not replicate. Held-out pair AUC 0.52-0.59 (15x1) and 0.55 (top-5 x 2). Only "Name your favorite color" hinted at LoRA vs disguises. |

Run (needs `export ME_API_KEY=...`): `python run_meta.py --prompts <file> --output <file> --repeats N [--parallel 4] [--allow-ep for ep, spends budget]`.

## Later the same night (all ref-only unless noted)

| What | Files | Result |
|---|---|---|
| Anchored MCQ, 27 questions in 3 tiers (stage 1 of a pre-registered 2-stage design) | `p_stage1.txt`, `anchor_items.json`, `ref_stage1.jsonl`, `analyze_stage.py`, `make_stage2.py` | 0 questions qualified: the 5 same-model ref versions never agreed; strong "A" position bias. Stage 2 not run. |
| 2-shot completions (20 items) | `p_complete.jsonl`, `ref_complete.jsonl`, `analyze_complete.py` | Server is chat-only; model answers conversationally ("I'm…"). The ref model identifies as SmolLM. |
| **ep**: 6 MCQs on ep-05/16/09/13 (24 ep queries) | `p_ep_mcq6.jsonl`, `ep_mcq6.jsonl` | ep model also mostly guesses (0–2/6 correct); no consistent split. |
| RUT proxy scoring with SmolLM2 135M/360M/1.7B (0 queries) | `rut_score.py`, `rut_run.log` | Best proxy 360M-Instruct; only temperature 1.0 stands out, LoRA not separated (model MCC 0.00–0.17). |
| Length-constraint overrun, existing data | (analysis in thread) | Post-hoc: variants overrun more often (10–15/16) than same-model (4–6/16). |
| Overrun retest, 4 new prompts x 5 (pre-registered) | `p_violate.txt`, `ref_violate.jsonl`, `analyze_violate.py` | PASS: overrun rate same-model 0.05–0.15, LoRA0.7 0.25 (thin), LoRA1.5 0.75; fp8 0.00 and LoRA0.3 0.10 not separated. |
| Overrun, 3 "high-information" prompts | `home_me_work/` (`p_ep_violate.txt`, `ref_violate3.jsonl`, `analyze_violate3.py`) | Everyone overruns (same-model 0.38–0.77); LoRA0.7 0.85 barely above; LoRA1.5 always. Lighter prompts separate better. |

`home_me_work/` was run from `$HOME` because `/p/scratch/training2666` hit its project inode limit (Disk quota exceeded) around 05:00. `start_ep_violate.sh` there is prepared but was not run (99 ep queries).
