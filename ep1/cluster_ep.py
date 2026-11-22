"""Cluster ep-01 ... ep-33 from ep_answers.jsonl and write submission.csv.

Method (same as experiment 2, on the probes in ep_probes.txt collected by ep_collect.py):
  1. Normalise each answer with answer_norm.py (--norm category: clean value or a format class).
  2. For every pair of endpoints and every probe: distance between the two answer histograms
     (TV = total variation, or tvtemp = temperature-tolerant TV). Average over the probes.
  3. Two-level average-linkage clustering:
       family threshold (loose): different base models have very different favourite answers -> large distance;
       model threshold (strict): inside a family, fp8/LoRA versions differ only a little.
  4. Write submission.csv with endpoint,family,model for ep-* only.

Calibration of the MODEL threshold (optional but recommended):
  If ref answers exist (exp2_answers.jsonl and/or reference_answers.jsonl), the script takes the SAME
  probes, the SAME number of repeats as on the ep endpoints, computes ref distances and picks the threshold
  with the best practice MCC. Caveat: ref is a different (small) model, so this threshold is only a guide.
  The FAMILY threshold cannot be calibrated on ref (ref has one family); choose it by looking at the
  sorted distances printed at the end.

Usage:
    python cluster_ep.py                              # calibrate model threshold on ref if possible
    python cluster_ep.py --model-threshold 0.45 --family-threshold 0.85
    python cluster_ep.py --metric tvtemp
    python cluster_ep.py --metric rank --family-threshold <r>       # rank scale 0..1
    python cluster_ep.py --metric mmd_perm --family-threshold <z>   # z-score scale: read the printed distances
Standard library only.
"""
import argparse
import collections
import csv
import itertools
import json
import math
import random
from pathlib import Path

from answer_norm import MODES, normalise   # copy of model_equality_exp2/answer_norm.py


def read_probes(path):
    """Probe texts, one per line (the same file ep_collect.py used)."""
    return [l.strip() for l in open(path, encoding="utf-8") if l.strip()]


def load(path, prefix, probe_texts, mode, max_repeats=None):
    """data[endpoint][probe_text] = [normalised answers], using only lines whose prompt TEXT is one of the
    probes (prompt ids differ between files) and endpoints starting with `prefix`.
    Keeps at most `max_repeats` answers per (endpoint, probe), lowest repeat numbers first."""
    raw = collections.defaultdict(lambda: collections.defaultdict(dict))
    if not Path(path).exists():
        return {}
    wanted = set(probe_texts)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            prompt = r["prompt"].strip()
            if prompt in wanted and r["endpoint"].startswith(prefix):
                raw[r["endpoint"]][prompt].setdefault(int(r["repeat"]), normalise(r["response"], prompt, mode))
    return {ep: {p: [a for _, a in sorted(reps.items())][:max_repeats] for p, reps in d.items()}
            for ep, d in raw.items()}


def load_truth(path):
    if not Path(path).exists():
        return None
    out = {}
    for r in csv.reader(open(path, newline="", encoding="utf-8-sig")):
        r = [c.strip() for c in r]
        if len(r) >= 3 and r[0].startswith("ref-"):
            out[r[0]] = (r[1], r[2], r[3] if len(r) > 3 else "")
    return out


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def to_prob(answers, smoothing=0.0, support=None):
    c = collections.Counter(answers)
    keys = support if support is not None else c.keys()
    total = sum(c[k] + smoothing for k in keys)
    return {k: (c[k] + smoothing) / total for k in keys}


def tv(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))


def sharpen(p, beta):
    w = {k: v ** beta for k, v in p.items() if v > 0}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def tv_temp(xs, ys):
    support = set(xs) | set(ys)
    p, q = to_prob(xs, 0.5, support), to_prob(ys, 0.5, support)
    return min(min(tv(sharpen(p, b), q), tv(p, sharpen(q, b))) for b in (0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0))


def l2_unbiased(xs, ys):
    """Unbiased estimate of sum_k (p_k - q_k)^2 (= the one-hot-kernel MMD from Gao et al.).
    Unlike TV, its EXPECTED value does not depend on how many answers we have, so a threshold found on
    ref with 10 answers per probe can be used on ep with 25. It can be slightly negative by chance."""
    cx, cy = collections.Counter(xs), collections.Counter(ys)
    n, m = len(xs), len(ys)
    if n < 2 or m < 2:
        return float("nan")
    sxx = sum(c * (c - 1) for c in cx.values()) / (n * (n - 1))
    syy = sum(c * (c - 1) for c in cy.values()) / (m * (m - 1))
    sxy = sum(c * cy.get(k, 0) for k, c in cx.items()) / (n * m)
    return sxx + syy - 2 * sxy


def mmd_perm(xs, ys, perms=200, seed=0):
    """Permutation-calibrated one-hot MMD (Gao et al., Model Equality Testing): z = (l2u - mean_null) / std_null,
    where the null comes from shuffling the pooled answers into groups of the same sizes. About 0 (+-1) when both
    endpoints are the same model, whatever the sample sizes; large when they differ. Deterministic (seeded)."""
    obs = l2_unbiased(xs, ys)
    if obs != obs:
        return obs
    rng = random.Random(f"{seed}|{sorted(xs)}|{sorted(ys)}")
    pool, n = list(xs) + list(ys), len(xs)
    null = []
    for _ in range(perms):
        rng.shuffle(pool)
        null.append(l2_unbiased(pool[:n], pool[n:]))
    mu = sum(null) / perms
    sd = math.sqrt(sum((v - mu) ** 2 for v in null) / (perms - 1))
    return (obs - mu) / sd if sd > 1e-12 else 0.0


def rank_distance(xs, ys, k=10):
    """LLMPrint-style rank distance: k most frequent answers of the pooled sample; fraction of answer pairs whose
    frequency ORDER differs between the two endpoints (tie on one side only = 1/2). Temperature-invariant.
    Biased upward at small n like TV, so it is sample-size matched to ref like tv."""
    cx, cy = collections.Counter(xs), collections.Counter(ys)
    top = [a for a, _ in sorted((cx + cy).items(), key=lambda t: (-t[1], str(t[0])))[:k]]
    if len(top) < 2:
        return 0.0
    n, m = len(xs), len(ys)
    bad = 0.0
    pairs = list(itertools.combinations(top, 2))
    for i, j in pairs:
        sx = (cx[i] / n > cx[j] / n) - (cx[i] / n < cx[j] / n)
        sy = (cy[i] / m > cy[j] / m) - (cy[i] / m < cy[j] / m)
        bad += 1.0 if sx * sy < 0 else 0.5 if sx != sy else 0.0
    return bad / len(pairs)


SIZE_FREE = ("l2u", "mmd_perm")   # no subsampling needed


def one_distance(xs, ys, metric):
    if metric == "mmd_perm":
        return mmd_perm(xs, ys)
    if metric == "rank":
        return rank_distance(xs, ys)
    if metric == "tv":
        return tv(to_prob(xs), to_prob(ys))
    if metric == "tvtemp":
        return tv_temp(xs, ys)
    return l2_unbiased(xs, ys)


def distances(data, metric, n=None, draws=30, seed=0):
    """D[(a, b)] = mean over the shared probes of the histogram distance.

    n: if given and smaller than the number of answers, TV-type distances are computed on random subsets of
    exactly n answers per (endpoint, probe) and averaged over `draws` subsets. Reason: TV between two
    histograms of n samples is biased upward, more so for small n. To use a threshold calibrated on ref
    with n answers, the ep distances must also be measured with n answers. (l2u needs no matching.)"""
    rng = random.Random(seed)
    D = {}
    for a, b in itertools.combinations(sorted(data), 2):
        vals = []
        for p in set(data[a]) & set(data[b]):
            xs, ys = data[a][p], data[b][p]
            if metric not in SIZE_FREE and n and (len(xs) > n or len(ys) > n):
                vals.append(sum(one_distance(rng.sample(xs, min(n, len(xs))), rng.sample(ys, min(n, len(ys))), metric)
                                for _ in range(draws)) / draws)
            else:
                vals.append(one_distance(xs, ys, metric))
        D[(a, b)] = sum(vals) / len(vals) if vals else 1.0
    return D


def dist(D, a, b):
    return 0.0 if a == b else D[(a, b)] if (a, b) in D else D[(b, a)]


# ---------------------------------------------------------------------------
# Clustering and scoring
# ---------------------------------------------------------------------------
def cluster(items, D, threshold):
    clusters = [[x] for x in items]
    while len(clusters) > 1:
        best = None
        for i, j in itertools.combinations(range(len(clusters)), 2):
            d = sum(dist(D, a, b) for a in clusters[i] for b in clusters[j]) / (len(clusters[i]) * len(clusters[j]))
            if best is None or d < best[0]:
                best = (d, i, j)
        if best[0] > threshold:
            break
        clusters[best[1]] += clusters.pop(best[2])
    return clusters


def family_letter(i):
    """0 -> A, 1 -> B, ..., 25 -> Z, 26 -> AA (the organisers' format: family = letter, model = letter + number)."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def labels_from(items, D, model_t, family_t):
    """Families first (loose threshold), then models inside each family (strict threshold)."""
    out = {}
    for fi, fam in enumerate(sorted(cluster(items, D, family_t), key=min)):
        for mi, mod in enumerate(sorted(cluster(fam, D, model_t), key=min)):
            for ep in mod:
                out[ep] = (family_letter(fi), f"{family_letter(fi)}{mi + 1}")
    return out


def mcc(labels, truth):
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(sorted(truth), 2):
        t, g = truth[a][:2] == truth[b][:2], labels[a] == labels[b]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def calibrate(ref, truth, metric):
    """Best practice-MCC threshold on ref. Returns (mcc, low, high): the whole RANGE of thresholds that reach
    the best MCC, so we can take its middle instead of an edge (more robust)."""
    eps = sorted(e for e in ref if e in truth)
    D = distances({e: ref[e] for e in eps}, metric)
    vals = sorted(set(D.values()))
    mids = [0.0] + [(x + y) / 2 for x, y in zip(vals, vals[1:])] + [vals[-1] + 1e-9]
    scored = [(mcc(labels_from(eps, D, t, float("inf")), {e: truth[e] for e in eps}), t) for t in mids]
    best = max(s for s, _ in scored)
    good = [t for s, t in scored if s == best]
    return best, min(good), max(good), D


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="ep_answers.jsonl")
    ap.add_argument("--probes", default="ep_probes.txt", help="the probes to use (must be the ones collected)")
    ap.add_argument("--norm", choices=sorted(MODES), default="category")
    ap.add_argument("--metric", choices=["tv", "tvtemp", "l2u", "mmd_perm", "rank"], default="tv",
                    help="tv / tvtemp (sample-size matched to ref automatically), l2u (unbiased, size-free) or "
                         "mmd_perm (l2u as a permutation z-score; size-free; ~0 for same model) or "
                         "rank (order of top-10 answer frequencies; temperature-invariant; size-matched like tv)")
    ap.add_argument("--model-threshold", type=float, default=None,
                    help="strict threshold inside a family; default: calibrated on ref, else 0.45")
    ap.add_argument("--family-threshold", type=float, default=0.7,
                    help="loose threshold between families (cannot be calibrated on ref; look at the printed distances)")
    ap.add_argument("--ref", nargs="*", default=["exp2_answers.jsonl", "reference_answers.jsonl"],
                    help="ref answer files used to calibrate the model threshold")
    ap.add_argument("--truth", default="practice_answers.csv")
    ap.add_argument("--out", default="submission.csv")
    args = ap.parse_args()

    probe_texts = read_probes(args.probes)
    ep = load(args.answers, "ep-", probe_texts, args.norm)
    if not ep:
        raise SystemExit(f"No ep answers for the probes found in {args.answers}. Run ep_collect.py first.")
    eps = sorted(ep)
    reps = min(len(v) for d in ep.values() for v in d.values())
    print(f"Loaded {len(eps)} ep endpoints; at least {reps} answers per probe (metric {args.metric})\n")

    # ---- Calibrate the model threshold on ref (same probes, same number of repeats) ----
    model_t = args.model_threshold
    truth = load_truth(args.truth)
    # Ref files are matched to the ep probes by prompt TEXT (ids differ between files and orders differ).
    ref = {}
    for path in args.ref:
        for e, d in load(path, "ref-", probe_texts, args.norm).items():
            for p, a in d.items():
                ref.setdefault(e, {}).setdefault(p, []).extend(a)    # pool all files
    missing = sorted({p for d in ref.values() for p in probe_texts if p not in d}) if ref else probe_texts
    ref = {e: d for e, d in ref.items() if all(len(d.get(p, [])) >= 2 for p in probe_texts)}
    n_match = None
    if truth and len(ref) >= 3:
        n_ref = min(len(v) for d in ref.values() for v in d.values())
        n_match = min(reps, n_ref)
        if args.metric not in SIZE_FREE:   # l2u is size-free, so it can use every ref answer
            ref = {e: {p: v[:n_match] for p, v in d.items()} for e, d in ref.items()}
        best, lo, hi, Dref = calibrate(ref, truth, args.metric)
        print(f"== Calibration on ref ({len(ref)} endpoints, {n_match} answers per probe; ep has {reps}) ==")
        if args.metric not in SIZE_FREE and n_match < reps:
            print(f"  ep distances will be measured on random subsets of {n_match} answers (averaged over 30 draws),")
            print(f"  so they are on the same scale as this threshold. Use --metric l2u to avoid subsampling.")
        same = sorted(v for (a, b), v in Dref.items() if truth[a][:2] == truth[b][:2])
        diff = sorted(v for (a, b), v in Dref.items() if truth[a][:2] != truth[b][:2])
        print(f"  same-model ref pairs: {min(same):.3f} .. {max(same):.3f} (median {same[len(same) // 2]:.3f})")
        print(f"  different-model ref pairs: {min(diff):.3f} .. {max(diff):.3f} (median {diff[len(diff) // 2]:.3f})")
        print(f"  best practice MCC {best:.3f} for model thresholds in [{lo:.3f}, {hi:.3f}]")
        if model_t is None:
            model_t = (lo + hi) / 2
            print(f"  -> using the middle, {model_t:.3f}. Ref is a different model, so treat this as a guide.\n")
    else:
        print("No usable ref answers for calibration. Need practice_answers.csv and ref answers containing ALL probes "
              f"(by exact text). Probes missing from the ref files: {missing}\n")
    if model_t is None:
        model_t = 0.45
        print(f"Using default model threshold {model_t}\n")

    # ---- Distances between ep endpoints ----
    D = distances(ep, args.metric, n_match)
    print(f"== Distance matrix ({args.metric}; smaller = more similar) ==")
    print("       " + " ".join(f"{e[-2:]:>5}" for e in eps))
    for a in eps:
        print(f"{a:>6} " + " ".join(f"{dist(D, a, b):5.2f}" for b in eps))
    print()

    # ---- Cluster and write ----
    labels = labels_from(eps, D, model_t, args.family_threshold)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        for e in eps:
            w.writerow([e, *labels[e]])
    fams = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in eps:
        fams[labels[e][0]][labels[e][1]].append(e)
    print(f"== Clusters (family threshold {args.family_threshold}, model threshold {model_t:.3f}) -> {args.out} ==")
    for f_, models in sorted(fams.items(), key=lambda kv: (len(kv[0]), kv[0])):
        print(f"  {f_}: " + "  ".join(f"{m}={members}" for m, members in sorted(models.items())))
        for m, members in sorted(models.items()):
            for i, pt in enumerate(probe_texts[:2]):
                top = collections.Counter(a for e in members for a in ep[e].get(pt, [])).most_common(4)
                print(f"      {m} probe {i}: {top}")
    print()

    # ---- Help for choosing thresholds by eye ----
    print("== Each endpoint's 3 nearest neighbours (use this to sanity-check the thresholds) ==")
    for a in eps:
        nn = sorted((dist(D, a, b), b) for b in eps if b != a)[:3]
        print(f"  {a}: " + ", ".join(f"{b} {d:.2f}" for d, b in nn))
    print("\nAll pair distances, sorted (a big jump usually marks the family boundary):")
    vals = sorted(D.values())
    print("  " + " ".join(f"{v:.2f}" for v in vals))


if __name__ == "__main__":
    main()
