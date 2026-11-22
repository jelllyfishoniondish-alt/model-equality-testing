"""Analyse the ref endpoint answers: determinism check -> pairwise distances -> hierarchical clustering
-> write an answer file -> score it.

Usage:
    python analyze_ref.py                         # default metric: mmd (look at this first)
    python analyze_ref.py --metric exact          # "exact match rate" distance
    python analyze_ref.py --metric prefix         # "common prefix length" distance
    python analyze_ref.py --threshold 0.05        # set the clustering threshold by hand
    python analyze_ref.py --top 10                # use only the 10 most discriminative prompts (overfits, see below)

Input:  reference_answers.jsonl (from run_ref.py), practice_answers.csv (official answers, optional, for scoring)
Output: my_reference_answer.csv, which you can check with: python score_practice.py my_reference_answer.csv

Uses only the Python standard library (no numpy/scipy).
"""
import argparse
import collections
import csv
import itertools
import json
import math
from pathlib import Path


# ---------------------------------------------------------------------------
# 0. Load data
# ---------------------------------------------------------------------------
def load_answers(path):
    """Returns data[endpoint][prompt_id] = [answer1, answer2, ...] (sorted by repeat)."""
    rows = collections.defaultdict(lambda: collections.defaultdict(dict))
    prompts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows[r["endpoint"]][r["id"]][r["repeat"]] = r["response"] or ""
            prompts[r["id"]] = r["prompt"]
    data = {ep: {pid: [reps[k] for k in sorted(reps)] for pid, reps in d.items()} for ep, d in rows.items()}
    return data, prompts


def load_truth(path):
    """Reads the official practice_answers.csv -> {endpoint: (family, model, kind)}. Returns None if missing."""
    if not Path(path).exists():
        return None
    truth = {}
    for r in csv.reader(open(path, newline="", encoding="utf-8-sig")):
        r = [c.strip() for c in r]
        if len(r) >= 3 and r[0].startswith("ref-"):
            truth[r[0]] = (r[1], r[2], r[3] if len(r) > 3 else "")
    return truth


# ---------------------------------------------------------------------------
# 1. Similarity between two single answers (the "kernel" k(y, y'), in 0..1, 1 = identical)
# ---------------------------------------------------------------------------
def char_ngrams(text, n=3):
    """Counts of character 3-grams, e.g. "abcd" -> {"abc": 1, "bcd": 1}."""
    text = " ".join(text.split())          # collapse whitespace so formatting differences don't matter
    return collections.Counter(text[i:i + n] for i in range(max(1, len(text) - n + 1)))


def cosine(c1, c2):
    """Cosine similarity of two count vectors."""
    dot = sum(v * c2.get(k, 0) for k, v in c1.items())
    n1 = math.sqrt(sum(v * v for v in c1.values()))
    n2 = math.sqrt(sum(v * v for v in c2.values()))
    return dot / (n1 * n2) if n1 and n2 else float(n1 == n2)


def prefix_sim(a, b):
    """Common prefix length / length of the longer answer. Under greedy decoding the same model gives
    identical answers; quantization or a LoRA usually makes the text "fork" at some position.
    The earlier the fork, the smaller this value."""
    if not a and not b:
        return 1.0
    i = 0
    while i < min(len(a), len(b)) and a[i] == b[i]:
        i += 1
    return i / max(len(a), len(b))


# ---------------------------------------------------------------------------
# 2. Distance between two endpoints on ONE prompt
# ---------------------------------------------------------------------------
def mean_sim(xs, ys, sim, same_set):
    """Mean similarity between every x in xs and every y in ys. same_set=True skips i == j (self-pairs)."""
    vals = [sim(x, y) for i, x in enumerate(xs) for j, y in enumerate(ys) if not (same_set and i == j)]
    return sum(vals) / len(vals) if vals else 1.0


def prompt_distance(xs, ys, metric, cache):
    """
    xs, ys: several answers from two endpoints to the same prompt.
    - exact : 1 - P(a random x and a random y are identical)
    - prefix: 1 - mean common-prefix fraction
    - mmd   : MMD^2 = within-x similarity + within-y similarity - 2 * between similarity,
              with character-3-gram cosine as the kernel.
              Same formula as the MMD in Gao et al., only the kernel differs.
              Advantage: fair to sampling endpoints, because their own normal variation is subtracted.
    """
    if metric == "exact":
        return 1 - mean_sim(xs, ys, lambda a, b: float(a == b), False)
    if metric == "prefix":
        return 1 - mean_sim(xs, ys, prefix_sim, False)
    # mmd
    gx = [cache.setdefault(x, char_ngrams(x)) for x in xs]
    gy = [cache.setdefault(y, char_ngrams(y)) for y in ys]
    kxx = mean_sim(gx, gx, cosine, True) if len(gx) > 1 else 1.0
    kyy = mean_sim(gy, gy, cosine, True) if len(gy) > 1 else 1.0
    kxy = mean_sim(gx, gy, cosine, False)
    return max(0.0, kxx + kyy - 2 * kxy)


def distance_tables(data, metric):
    """Returns per_prompt[(a, b)][pid] = distance, for every endpoint pair and every prompt both answered."""
    eps = sorted(data)
    cache = {}
    per_prompt = {}
    for a, b in itertools.combinations(eps, 2):
        common = sorted(set(data[a]) & set(data[b]), key=lambda s: int(s) if s.isdigit() else s)
        per_prompt[(a, b)] = {pid: prompt_distance(data[a][pid], data[b][pid], metric, cache) for pid in common}
    return per_prompt


def average_over_prompts(per_prompt, use_pids=None):
    """Averages the per-prompt distances into one distance per endpoint pair, D[(a, b)]."""
    D = {}
    for pair, d in per_prompt.items():
        vals = [v for pid, v in d.items() if use_pids is None or pid in use_pids]
        D[pair] = sum(vals) / len(vals) if vals else 1.0
    return D


def dist(D, a, b):
    return 0.0 if a == b else D[(a, b)] if (a, b) in D else D[(b, a)]


# ---------------------------------------------------------------------------
# 3. Hierarchical clustering (average linkage)
# ---------------------------------------------------------------------------
def cluster(items, D, threshold):
    """Start with one cluster per endpoint; repeatedly merge the two clusters with the smallest average
    distance until that distance is > threshold."""
    clusters = [[x] for x in items]
    while len(clusters) > 1:
        best = None
        for i, j in itertools.combinations(range(len(clusters)), 2):
            d = sum(dist(D, a, b) for a in clusters[i] for b in clusters[j]) / (len(clusters[i]) * len(clusters[j]))
            if best is None or d < best[0]:
                best = (d, i, j)
        if best[0] > threshold:
            break
        _, i, j = best
        clusters[i] += clusters.pop(j)
    return clusters


def labels_from(items, D, model_threshold, family_threshold):
    """First split into families with a loose threshold, then split each family into models with a strict
    threshold (this guarantees that the same model always means the same family)."""
    out = {}
    for fi, fam in enumerate(sorted(cluster(items, D, family_threshold), key=min)):
        for mi, mod in enumerate(sorted(cluster(fam, D, model_threshold), key=min)):
            for ep in mod:
                out[ep] = (f"F{fi}", f"F{fi}M{mi}")
    return out


# ---------------------------------------------------------------------------
# 4. Scoring (same as score_practice.py: same model = same family AND same model label)
# ---------------------------------------------------------------------------
def mcc(labels, truth):
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(sorted(truth), 2):
        t = truth[a][:2] == truth[b][:2]
        g = labels[a] == labels[b]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def auc_for_prompt(per_prompt, pid, truth):
    """How discriminative one prompt is: the probability that a random same-model pair is closer than a
    random different-model pair.
    1.0 = perfect; 0.5 = no better than guessing; < 0.5 = inverted (the disguises matter more than the
    real differences)."""
    same, diff = [], []
    for (a, b), d in per_prompt.items():
        if pid not in d or a not in truth or b not in truth:
            continue
        (same if truth[a][:2] == truth[b][:2] else diff).append(d[pid])
    if not same or not diff:
        return float("nan")
    wins = sum((s < t) + 0.5 * (s == t) for s in same for t in diff)
    return wins / (len(same) * len(diff))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="reference_answers.jsonl")
    ap.add_argument("--truth", default="practice_answers.csv")
    ap.add_argument("--metric", choices=["mmd", "exact", "prefix"], default="mmd")
    ap.add_argument("--threshold", type=float, default=None,
                    help="model clustering threshold; if omitted, the best one on the practice answers is used")
    ap.add_argument("--family-threshold", type=float, default=None,
                    help="family clustering threshold; ref has a single family, so by default families are not split")
    ap.add_argument("--top", type=int, default=0,
                    help="use only the N most discriminative prompts (overfits the practice answers; exploration only)")
    ap.add_argument("--out", default="my_reference_answer.csv")
    args = ap.parse_args()

    data, prompts = load_answers(args.answers)
    truth = load_truth(args.truth)
    eps = sorted(data)
    print(f"Loaded {len(eps)} endpoints, {len(prompts)} prompts\n")

    # ---- Step 1: determinism check ----
    print("== Step 1: determinism check (are repeated answers to the same prompt identical?) ==")
    for ep in eps:
        multi = [r for r in data[ep].values() if len(r) > 1]
        if not multi:
            print(f"  {ep}: only 1 answer per prompt, cannot tell (run: python run_ref.py 3 or more)")
            continue
        same = sum(len(set(r)) == 1 for r in multi)
        kind = f"  [{truth[ep][2]}]" if truth and ep in truth else ""
        tag = ("looks greedy/deterministic" if same == len(multi)
               else "looks like sampling" if same < 0.5 * len(multi) else "partly deterministic")
        print(f"  {ep}: identical answers on {same}/{len(multi)} prompts -> {tag}{kind}")
    print()

    # ---- Step 2: pairwise distances ----
    per_prompt = distance_tables(data, args.metric)

    use_pids = None
    if truth:
        print(f"== Step 2a: how discriminative each prompt is (AUC, metric {args.metric}; 1 = perfect, 0.5 = useless) ==")
        aucs = sorted(((auc_for_prompt(per_prompt, pid, truth), pid) for pid in prompts), reverse=True)
        for a, pid in aucs:
            print(f"  AUC {a:.2f}  [{pid:>3}] {prompts[pid][:70]}")
        print()
        if args.top:
            use_pids = {pid for _, pid in aucs[:args.top]}
            print(f"(Using only the top {args.top} prompts. They were picked with the answers, so the score is optimistic.)\n")

    D = average_over_prompts(per_prompt, use_pids)
    print(f"== Step 2b: endpoint distance matrix (metric {args.metric}; smaller = more similar) ==")
    print("        " + " ".join(f"{e[-2:]:>6}" for e in eps))
    for a in eps:
        print(f"{a:>7} " + " ".join(f"{dist(D, a, b):6.3f}" for b in eps))
    print()
    print("The 12 closest pairs:")
    for (a, b), d in sorted(D.items(), key=lambda kv: kv[1])[:12]:
        note = ""
        if truth and a in truth and b in truth:
            same = truth[a][:2] == truth[b][:2]
            note = f"   truth: {'same model' if same else 'different'} ({truth[a][2]} / {truth[b][2]})"
        print(f"  {a} - {b}: {d:.4f}{note}")
    print()

    # ---- Step 3: clustering + threshold choice ----
    fam_t = args.family_threshold if args.family_threshold is not None else float("inf")
    thr = args.threshold
    if truth:
        print("== Step 3: threshold sweep (MCC on the practice answers) ==")
        cands = sorted(set(D.values()))
        # Candidate thresholds are midpoints between neighbouring distances, so each gives a different merge
        mids = [0.0] + [(x + y) / 2 for x, y in zip(cands, cands[1:])] + [cands[-1] + 1e-9]
        results = [(mcc(labels_from(eps, D, t, fam_t), truth), t) for t in mids]
        for s, t in results:
            print(f"  threshold {t:.4f} -> MCC {s:.3f}")
        best = max(results)  # on ties, the larger threshold wins
        if thr is None:
            thr = best[1]
        print(f"\nBest MCC = {best[0]:.3f} (threshold {best[1]:.4f}). "
              "This threshold was picked with the answers, so real performance will be somewhat worse.\n")
    elif thr is None:
        vals = sorted(D.values())
        thr = vals[len(vals) // 2]
        print(f"No practice answers found; using the median distance {thr:.4f} as the threshold\n")

    labels = labels_from(eps, D, thr, fam_t)

    # ---- Step 4: write the answer file ----
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        for ep in eps:
            w.writerow([ep, *labels[ep]])
    print(f"== Step 4: clusters (threshold {thr:.4f}), written to {args.out} ==")
    groups = collections.defaultdict(list)
    for ep in eps:
        groups[labels[ep][1]].append(ep)
    for m, members in sorted(groups.items()):
        extra = f"   truth: {[truth[e][1] for e in members]}" if truth else ""
        print(f"  {m}: {members}{extra}")
    if truth:
        print(f"\nPractice MCC = {mcc(labels, truth):.3f}. You can also run: python score_practice.py {args.out}")


if __name__ == "__main__":
    main()
