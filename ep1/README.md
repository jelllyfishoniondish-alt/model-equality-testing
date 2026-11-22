# ep1: clustering the 33 scored endpoints

Data: `ep_answers.jsonl`, 2 probes (`ep_probes.txt`: number 1-100, animal) x 20 answers per endpoint, collected with `ep_collect.py` (spends real ep budget; asks for confirmation).

| File | Purpose |
| --- | --- |
| `ep_collect.py` | Query ep-01..ep-33 with the probes (needs `ME_API_KEY`). Resumable. |
| `cluster_ep.py` | Two-level average-linkage clustering (family threshold, then model threshold). Metrics: `tv`, `tvtemp`, `l2u`, `mmd_perm`, `rank`. Writes `endpoint,family,model` CSV via `--out`. |
| `answer_norm.py` | Normalises raw answers (numbers, animal names). |
| `submission_family.csv` | l2u, family 0.10 / model 0.10: families X, Y, Z, {02,23}, 04, 06, 10, 20. Public score 0.1 (families right, no model split). |
| `queue/` | Submission candidates v1..v5, `README.txt` explains each; `scores.log` has the public scores. `old/` = earlier family-level variants. |
| `submit_queue.sh` | Submits queue/v1..v5 one at a time, 31 min apart (server limit), first after 23:10. Copies each to `queue/current.csv`, which `../submission.py` points to. |

```bash
source ../.venv/bin/activate
export ME_API_KEY="your team key"
python cluster_ep.py --metric l2u --family-threshold 0.10 --model-threshold 0.10 --out submission_family.csv
```

Findings so far (no new ep queries):
- Family structure is clear on l2u / mmd_perm (stable for family threshold 0.08 to 0.12).
- Inside families, raw formatting gives nothing (ep answers are always clean). Within-family mmd_perm z on both probes:
  X shows no differences; Y: ep-28 differs on both probes; Z splits by number-probe mode into {05,16}, {09,13,31}, {11,15,19,27,29}.
- On ref (exp3, partial), number 1-100 and number 1-10 are the probes that separate models; animal mostly separates families.

## Night of 21-22 Nov: submissions and what we learned

Score = 0.1 x family MCC (all pairs) + 0.9 x model MCC (pairs inside one family), negatives clipped to 0; leaderboard shows 0.05 steps.

| Submission | Idea | Public |
|---|---|---|
| v1 | l2u families, one model per family | 0.10 (families right, model part 0) |
| v2 | Z: {05,16} vs rest | 0.10 |
| v3 | Y: 28 alone; Z: {05,16}, {13}, rest | 0.10 |
| v4 | most central half of each family = one model, others singletons | 0.10 |
| gt/s28 | only ep-28 split off | 0.10 |

Offline check (no queries): no endpoint differs from every member of its family; each "outlier" has a near-identical partner (13~09, 05~16, 28~21, 07~22), so single-endpoint splits always break a real pair.
Current hypothesis: each family holds several models with 1-3 disguised copies each. Candidates: `queue/gt/v6.csv` (complete linkage, z<=1.0), `queue/gt/v6b.csv` (z<=0.5), `queue/v6c.csv` (mutual nearest pairs).
Also prepared: `queue/gt/s01|s13|s05.csv` (single splits), `queue/gt/b1-b5.csv` (one split per family X/Y/Z), `submit_single.sh` (paced queue, not used). New ref experiments are in `../exp4/README.md`.
