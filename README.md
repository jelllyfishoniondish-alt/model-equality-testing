# Model Equality Testing: black-box LLM endpoint auditing

Task 2 of the **CISPA · Fraunhofer AISEC Hackathon (European AI Championship), Paris, November 2026**, team *argos*.

**Problem.** 33 black-box chat endpoints, 100 answers each. Some serve the same model behind disguises
(system prompt, persona, decoding watermark, temperature); others serve genuinely different models
(fp8 quantization, LoRA adapters of different strengths). Group the endpoints into base-model families and,
inside each family, into identical models.

**Read first:** [FINDINGS.md](FINDINGS.md): every probe tried and why it was ruled out, identification of the hidden base models, and a retrospective.

## Highlights

- Recovered the three families exactly family part solved.
- Identified the hidden base models from answer distributions alone, with no extra API queries:
  **Qwen2.5-3B-Instruct**, **Qwen2.5-1.5B-Instruct** and **Llama-3.2-1B-Instruct** (total-variation distance 0.14–0.24 on single-token probes).
- Built a "disguise null range" by simulating system prompts / personas / temperatures locally on each base model,
  which showed why histogram and token-likelihood tests cannot separate fp8 and weak LoRA variants at this budget.
- Pre-registered every validation on the unlimited practice endpoints before spending scored budget.

## Layout

| Folder | Content |
|---|---|
| `ep1/` | Two-level clustering of the scored endpoints (`cluster_ep.py`: l2u, MMD permutation z, rank, temperature-invariant TV), submission candidates and score log |
| `exp1/`–`exp3/` | Probe selection on the practice endpoints (number / animal histograms, prompt-type sweeps) |
| `exp4/` | Later experiments: metadata, long-answer style, FLIPS, many-prompt agreement, multiple choice, completions, proxy log-prob (RUT), length-constraint overrun |
| `exp4/base_models/` | Base-model identification and the disguise null range (`id_pref.py`, `id_endpoint.py`, `id_null*.py`) |

## Running

The competition kit (`query_reference.py`, `task_template.py`, `submission.py`, `score_practice.py`, practice answers)
and the server address belong to the organisers and are not included. Put the kit next to the scripts, then:

```bash
export ME_API_KEY="<team key>"
export ME_SERVER_URL="<competition server>"   # used by exp4/meta_query.py
```

Analysis scripts that read the saved `*.jsonl` answers run without the kit or a key.
