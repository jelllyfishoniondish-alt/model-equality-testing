"""Experiment 2 analysis: compare answer HISTOGRAMS of short one-word probes between endpoints.

Steps
  1. Normalise each answer (lowercase, strip punctuation, keep the first word) -> a histogram per (endpoint, prompt).
  2. Show the top answers of each endpoint for the best probes (to build intuition).
  3. Distance between two endpoints on one prompt = distance between their histograms; average over prompts.
  4. Per-probe AUC with an endpoint-bootstrap CI, and separation S(q) = median(different) - median(same).
  5. Hierarchical clustering + threshold sweep + write my_exp2_answer.csv (+ bootstrap CI for the MCC).
  6. Split-half validation: choose probes and threshold on even repeats, score on odd repeats (and vice versa).
  7. Disguise diagnostic: each endpoint's distance to ref-01 versus its own sampling-noise floor.
  8. Budget simulation: what happens with only 100 answers per endpoint (like the ep endpoints)?

Usage:
    python analyze_exp2.py                       # default metric tv
    python analyze_exp2.py --metric js           # Jensen-Shannon divergence
    python analyze_exp2.py --metric tvtemp       # temperature-tolerant TV (see below)
    python analyze_exp2.py --top 5               # cluster with the 5 best probes only (optimistic)
    python analyze_exp2.py --no-sim              # skip the budget simulation (faster)

Uses only the Python standard library.
"""
import argparse
import collections
import csv
import itertools
import json
import math
import random
import re
import statistics
from pathlib import Path


# ---------------------------------------------------------------------------
# 0. Load and normalise
# ---------------------------------------------------------------------------
from answer_norm import FORMAT_CLASSES, MODES, is_invalid, normalise   # see answer_norm.py


def load_answers(path, mode="category"):
    """Returns data[endpoint][prompt_id] = [normalised answers] (sorted by repeat),
    halves = (even-repeat data, odd-repeat data) in the same shape, and prompts[prompt_id] = prompt text."""
    raw = collections.defaultdict(lambda: collections.defaultdict(dict))
    texts = collections.defaultdict(lambda: collections.defaultdict(list))   # raw responses, for --show-raw
    prompts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                raw[r["endpoint"]][r["id"]][int(r["repeat"])] = normalise(r["response"], r["prompt"], mode)
                texts[r["endpoint"]][r["id"]].append(r["response"] or "")
                prompts[r["id"]] = r["prompt"]
    def pick(cond):
        return {ep: {pid: [a for k, a in sorted(reps.items()) if cond(k)] for pid, reps in d.items()}
                for ep, d in raw.items()}
    return pick(lambda k: True), (pick(lambda k: k % 2 == 0), pick(lambda k: k % 2 == 1)), prompts, texts


def load_truth(path):
    if not Path(path).exists():
        return None
    truth = {}
    for r in csv.reader(open(path, newline="", encoding="utf-8-sig")):
        r = [c.strip() for c in r]
        if len(r) >= 3 and r[0].startswith("ref-"):
            truth[r[0]] = (r[1], r[2], r[3] if len(r) > 3 else "")
    return truth


# ---------------------------------------------------------------------------
# 1. Distances between two histograms
# ---------------------------------------------------------------------------
def to_prob(answers, smoothing=0.0, support=None):
    c = collections.Counter(answers)
    keys = support if support is not None else c.keys()
    total = sum(c[k] + smoothing for k in keys)
    return {k: (c[k] + smoothing) / total for k in keys}


def tv(p, q):
    """Total variation distance: half the sum of |p - q|. 0 = identical, 1 = no overlap."""
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


def js(p, q):
    """Jensen-Shannon divergence (base 2), in 0..1."""
    keys = set(p) | set(q)
    m = {k: 0.5 * (p.get(k, 0) + q.get(k, 0)) for k in keys}
    def kl(a):
        return sum(a[k] * math.log2(a[k] / m[k]) for k in a if a[k] > 0)
    return 0.5 * kl(p) + 0.5 * kl(q)


def sharpen(p, beta):
    """p^beta, renormalised. beta > 1 = like a lower temperature (sharper), beta < 1 = higher temperature."""
    w = {k: v ** beta for k, v in p.items() if v > 0}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


BETAS = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0]


def tv_temp(xs, ys):
    """Temperature-tolerant TV: the decoding temperature does NOT change model identity, but it does
    sharpen or flatten the histogram. So we try re-sharpening one side with several betas and keep the
    smallest TV. Quantization/LoRA change WHICH answers are likely, which re-sharpening cannot undo.
    (A small smoothing is needed so that unseen answers can get probability after re-sharpening.)"""
    support = set(xs) | set(ys)
    p = to_prob(xs, 0.5, support)
    q = to_prob(ys, 0.5, support)
    return min(min(tv(sharpen(p, b), q), tv(p, sharpen(q, b))) for b in BETAS)


def prompt_distance(xs, ys, metric):
    if metric == "tv":
        return tv(to_prob(xs), to_prob(ys))
    if metric == "js":
        return js(to_prob(xs), to_prob(ys))
    return tv_temp(xs, ys)


def per_prompt_distances(data, metric, pids):
    eps = sorted(data)
    return {(a, b): {pid: prompt_distance(data[a][pid], data[b][pid], metric)
                     for pid in pids if data[a].get(pid) and data[b].get(pid)}
            for a, b in itertools.combinations(eps, 2)}


def average(per_prompt, use_pids=None):
    D = {}
    for pair, d in per_prompt.items():
        vals = [v for pid, v in d.items() if use_pids is None or pid in use_pids]
        D[pair] = sum(vals) / len(vals) if vals else 1.0
    return D


def dist(D, a, b):
    return 0.0 if a == b else D[(a, b)] if (a, b) in D else D[(b, a)]


# ---------------------------------------------------------------------------
# 2. Clustering and scoring (same as experiment 1)
# ---------------------------------------------------------------------------
def cluster(items, D, threshold):
    """Average-linkage agglomerative clustering; stop when the closest clusters are > threshold apart."""
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


def pair_auc(values_by_pair, truth):
    """P(a random same-model pair is closer than a random different-model pair). Threshold-free."""
    same = [v for (a, b), v in values_by_pair.items() if truth[a][:2] == truth[b][:2]]
    diff = [v for (a, b), v in values_by_pair.items() if truth[a][:2] != truth[b][:2]]
    if not same or not diff:
        return float("nan")
    return sum((s < t) + 0.5 * (s == t) for s in same for t in diff) / (len(same) * len(diff))


def sweep(eps, D, truth, family_t):
    """Best MCC over all thresholds (optimistic: uses the answers). Returns (mcc, threshold)."""
    cands = sorted(set(D.values()))
    mids = [0.0] + [(x + y) / 2 for x, y in zip(cands, cands[1:])] + [cands[-1] + 1e-9]
    return max((mcc(labels_from(eps, D, t, family_t), truth), t) for t in mids)


def separation(values_by_pair, truth):
    """S(q) = median distance of different-model pairs - median distance of same-model pairs.
    Positive and large = good probe. Unlike AUC it also tells you HOW FAR apart the two groups are,
    which matters because the ep endpoints need a threshold chosen without answers."""
    same = [v for (a, b), v in values_by_pair.items() if truth[a][:2] == truth[b][:2]]
    diff = [v for (a, b), v in values_by_pair.items() if truth[a][:2] != truth[b][:2]]
    if not same or not diff:
        return float("nan")
    return statistics.median(diff) - statistics.median(same)


def percentile_ci(vals):
    vals = sorted(v for v in vals if not math.isnan(v))
    if not vals:
        return float("nan"), float("nan")
    return vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]


def boot_auc(values_by_pair, truth, eps, rng, n=200):
    """Endpoint-level bootstrap CI for an AUC: redraw the 9 endpoints with replacement and recompute the
    AUC over pairs of DIFFERENT endpoints. This shows how much the AUC depends on which endpoints we
    happen to have (with only 9 endpoints, it can depend a lot)."""
    out = []
    for _ in range(n):
        s = [rng.choice(eps) for _ in eps]
        same, diff = [], []
        for i, j in itertools.combinations(range(len(s)), 2):
            a, b = s[i], s[j]
            if a == b:
                continue
            v = values_by_pair[(a, b)] if (a, b) in values_by_pair else values_by_pair[(b, a)]
            (same if truth[a][:2] == truth[b][:2] else diff).append(v)
        if same and diff:
            out.append(sum((x < y) + 0.5 * (x == y) for x in same for y in diff) / (len(same) * len(diff)))
    return percentile_ci(out)


def boot_mcc(eps, D, truth, thr, family_t, rng, n=200):
    """Endpoint-level bootstrap CI for the MCC at a FIXED threshold: redraw endpoints with replacement,
    keep the distinct ones, cluster them and score only their pairs."""
    out = []
    for _ in range(n):
        s = sorted(set(rng.choice(eps) for _ in eps))
        sub = {e: truth[e] for e in s}
        pairs = list(itertools.combinations(s, 2))
        if not any(sub[a][:2] == sub[b][:2] for a, b in pairs) or all(sub[a][:2] == sub[b][:2] for a, b in pairs):
            continue
        out.append(mcc(labels_from(s, D, thr, family_t), sub))
    return percentile_ci(out)


def cross_half(halves, a, b, metric, pids):
    """Distance between endpoint a and b computed ACROSS halves (a's even repeats vs b's odd repeats and
    vice versa). With a == b this is the noise floor: how far an endpoint is from ITSELF just because of
    sampling. Every number here uses half the answers, so they are directly comparable."""
    even, odd = halves
    vals = []
    for pid in pids:
        if even[a].get(pid) and odd[b].get(pid) and odd[a].get(pid) and even[b].get(pid):
            if a == b:
                vals.append(prompt_distance(even[a][pid], odd[a][pid], metric))
            else:
                vals.append(0.5 * (prompt_distance(even[a][pid], odd[b][pid], metric)
                                   + prompt_distance(odd[a][pid], even[b][pid], metric)))
    return sum(vals) / len(vals) if vals else float("nan")


def split_half(select, evaluate, eps, pids, truth, metric, K, family_t):
    """Choose the K best probes AND the threshold on one half of the repeats, then score on the other half.
    This is the honest estimate: nothing about the evaluation half was used to make choices."""
    pp_s = per_prompt_distances(select, metric, pids)
    auc_s = {pid: pair_auc({pr: d[pid] for pr, d in pp_s.items() if pid in d}, truth) for pid in pids}
    chosen = sorted(pids, key=lambda p: -auc_s[p])[:K] if K else list(pids)
    mcc_s, t = sweep(eps, average(pp_s, set(chosen)), truth, family_t)
    D_e = average(per_prompt_distances(evaluate, metric, chosen))
    return chosen, mcc_s, t, mcc(labels_from(eps, D_e, t, family_t), truth), pair_auc(D_e, truth)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="exp2_answers.jsonl")
    ap.add_argument("--truth", default="practice_answers.csv")
    ap.add_argument("--metric", choices=["tv", "js", "tvtemp"], default="tv")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--family-threshold", type=float, default=None)
    ap.add_argument("--top", type=int, default=0, help="cluster using only the N probes with the best AUC (optimistic)")
    ap.add_argument("--norm", choices=sorted(MODES), default="category",
                    help="how answers are normalised: category (value or format class), lenient, first3")
    ap.add_argument("--show-raw", type=int, default=0, help="print N raw responses per endpoint for the probes in --raw-probe")
    ap.add_argument("--raw-probe", nargs="*", default=["0", "3"], help="probe ids for --show-raw (default: random number, animal)")
    ap.add_argument("--k", type=int, default=5, help="number of probes chosen in the split-half validation")
    ap.add_argument("--no-sim", action="store_true", help="skip the budget simulation")
    ap.add_argument("--budget", type=int, default=100, help="answers per endpoint in the simulation")
    ap.add_argument("--trials", type=int, default=30, help="random subsamples per setting in the simulation")
    ap.add_argument("--out", default="my_exp2_answer.csv")
    args = ap.parse_args()

    data, halves, prompts, texts = load_answers(args.answers, args.norm)
    rng_boot = random.Random(1)
    truth = load_truth(args.truth)
    eps = sorted(data)
    pids = sorted(prompts, key=lambda s: int(s) if s.isdigit() else s)
    fam_t = args.family_threshold if args.family_threshold is not None else float("inf")
    n_per = [len(v) for ep in eps for v in data[ep].values()]
    print(f"Loaded {len(eps)} endpoints, {len(pids)} prompts, {min(n_per)}-{max(n_per)} answers per (endpoint, prompt)\n")

    # ---- Format check: how often does each endpoint give an answer we cannot parse? ----
    print(f"== Step 0: answer format per endpoint (norm mode: {args.norm}) ==")
    print("  valid = a clean in-format answer; the other columns count format classes (see answer_norm.py)")
    for ep in eps:
        allv = [a for v in data[ep].values() for a in v]
        kind = f"  [{truth[ep][2]}]" if truth and ep in truth else ""
        c = collections.Counter(a if is_invalid(a) else "valid" for a in allv)
        cols = "  ".join(f"{k} {c[k]:>4}" for k in ("valid",) + FORMAT_CLASSES if c[k])
        print(f"  {ep}: {cols}  (of {len(allv)}){kind}")
    print()
    if args.show_raw:
        for pid in args.raw_probe:
            if pid not in prompts:
                continue
            print(f"== Raw responses for [{pid}] {prompts[pid]} ==")
            for ep in eps:
                for t in texts[ep].get(pid, [])[:args.show_raw]:
                    print(f"  {ep}: {t[:80]!r:84} -> {normalise(t, prompts[pid], args.norm)}")
            print()

    per_prompt = per_prompt_distances(data, args.metric, pids)

    # ---- Per-probe AUC ----
    aucs = {}
    if truth:
        print(f"== Step 1: how discriminative each probe is (metric {args.metric}) ==")
        print("  AUC: 1 = perfect, 0.5 = useless. CI: 95% endpoint-bootstrap interval (wide = unreliable).")
        print("  S(q) = median distance of different-model pairs - median of same-model pairs (bigger = better).")
        sep = {}
        for pid in pids:
            vals = {pair: d[pid] for pair, d in per_prompt.items() if pid in d}
            aucs[pid] = pair_auc(vals, truth)
            sep[pid] = (separation(vals, truth), boot_auc(vals, truth, eps, rng_boot))
        for pid in sorted(pids, key=lambda p: -aucs[p]):
            s, (lo, hi) = sep[pid]
            print(f"  AUC {aucs[pid]:.2f} [{lo:.2f}, {hi:.2f}]  S(q) {s:+.3f}  [{pid:>2}] {prompts[pid][:60]}")
        print()

    # ---- Histograms of the best probes ----
    show = sorted(pids, key=lambda p: -aucs.get(p, 0))[:3]
    print("== Step 2: top answers per endpoint for the 3 best probes (answer:count) ==")
    for pid in show:
        print(f"\n  [{pid}] {prompts[pid]}")
        for ep in eps:
            top = collections.Counter(data[ep][pid]).most_common(6)
            kind = f"  [{truth[ep][1]}: {truth[ep][2]}]" if truth and ep in truth else ""
            print(f"    {ep}: " + ", ".join(f"{a}:{c}" for a, c in top) + kind)
    print()

    # ---- Distance matrix ----
    use = set(sorted(pids, key=lambda p: -aucs[p])[:args.top]) if (args.top and truth) else None
    if use:
        print(f"(Using only the top {args.top} probes. They were picked with the answers, so this is optimistic.)\n")
    D = average(per_prompt, use)
    print(f"== Step 3: endpoint distance matrix (metric {args.metric}; smaller = more similar) ==")
    print("        " + " ".join(f"{e[-2:]:>6}" for e in eps))
    for a in eps:
        print(f"{a:>7} " + " ".join(f"{dist(D, a, b):6.3f}" for b in eps))
    print("\nThe 12 closest pairs:")
    for (a, b), d in sorted(D.items(), key=lambda kv: kv[1])[:12]:
        note = ""
        if truth:
            note = f"   truth: {'SAME' if truth[a][:2] == truth[b][:2] else 'different'} ({truth[a][2]} / {truth[b][2]})"
        print(f"  {a} - {b}: {d:.4f}{note}")
    print()

    # ---- Clustering ----
    thr = args.threshold
    if truth:
        print(f"Pair AUC over all pairs (threshold-free separation): {pair_auc(D, truth):.3f}")
        best_mcc, best_t = sweep(eps, D, truth, fam_t)
        print(f"Best MCC over thresholds: {best_mcc:.3f} at threshold {best_t:.4f} (optimistic: picked with the answers)\n")
        if thr is None:
            thr = best_t
    elif thr is None:
        vals = sorted(D.values())
        thr = vals[len(vals) // 2]
    labels = labels_from(eps, D, thr, fam_t)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        for ep in eps:
            w.writerow([ep, *labels[ep]])
    print(f"== Step 4: clusters at threshold {thr:.4f}, written to {args.out} ==")
    groups = collections.defaultdict(list)
    for ep in eps:
        groups[labels[ep][1]].append(ep)
    for m, members in sorted(groups.items()):
        extra = f"   truth: {[truth[e][1] for e in members]}" if truth else ""
        print(f"  {m}: {members}{extra}")
    if truth:
        print(f"\nPractice MCC = {mcc(labels, truth):.3f}. Check with: python score_practice.py {args.out}")
        lo, hi = boot_mcc(eps, D, truth, thr, fam_t, rng_boot)
        print(f"Endpoint-bootstrap 95% CI for this MCC at the fixed threshold: [{lo:.3f}, {hi:.3f}]\n")
    if not truth:
        return

    # ---- Split-half validation ----
    K = args.k
    print(f"== Step 5: split-half validation (choose top-{K} probes + threshold on one half, score on the other) ==")
    print("  This is the honest number: the evaluation half was never used to make a choice.")
    for name, sel, ev in (("even -> odd", halves[0], halves[1]), ("odd -> even", halves[1], halves[0])):
        for k in (K, 0):
            chosen, m_sel, t, m_ev, a_ev = split_half(sel, ev, eps, pids, truth, args.metric, k, fam_t)
            label = f"top-{k} probes" if k else "all probes  "
            print(f"  {name}, {label}: MCC on selection half {m_sel:.3f} (threshold {t:.4f}) "
                  f"-> MCC on held-out half {m_ev:.3f}, pair-AUC {a_ev:.3f}")
            if k:
                print(f"      probes chosen: {chosen}")
    print("  A big drop from selection half to held-out half = overfitting to these 9 endpoints.\n")

    # ---- Watermark / temperature diagnostic ----
    anchor = "ref-01" if "ref-01" in truth else eps[0]
    print(f"== Step 6: disguise diagnostic, distances to {anchor} ({truth[anchor][2]}), half-vs-half, all probes ==")
    print("  noise = how far each endpoint is from ITSELF (even vs odd repeats). excess = distance - noise.")
    print("  Goal: same-model endpoints (watermark, system prompt, persona, temperature) have excess near 0,")
    print("  and every different-model endpoint (fp8, LoRA) has clearly larger excess.")
    floor_anchor = cross_half(halves, anchor, anchor, args.metric, pids)
    rows = []
    for e in eps:
        if e == anchor:
            continue
        d = cross_half(halves, anchor, e, args.metric, pids)
        noise = 0.5 * (floor_anchor + cross_half(halves, e, e, args.metric, pids))
        rows.append((d - noise, e, d, noise))
    for exc, e, d, noise in sorted(rows):
        tag = "SAME     " if truth[e][:2] == truth[anchor][:2] else "different"
        print(f"  {e} {tag} {truth[e][2]:<32} distance {d:.3f}  noise {noise:.3f}  excess {exc:+.3f}")
    worst_same = max((r for r in rows if truth[r[1]][:2] == truth[anchor][:2]), default=None)
    best_diff = min((r for r in rows if truth[r[1]][:2] != truth[anchor][:2]), default=None)
    if worst_same and best_diff:
        gap = best_diff[0] - worst_same[0]
        print(f"  Margin = smallest different-model excess ({best_diff[1]}) - largest same-model excess "
              f"({worst_same[1]}) = {gap:+.3f}  ({'separable' if gap > 0 else 'NOT separable'} from {anchor})")
    print()

    # ---- Budget simulation ----
    if args.no_sim:
        return
    print(f"== Step 7: budget simulation ({args.budget} answers per endpoint, {args.trials} random subsamples each) ==")
    print("  K probes = how many probes we would use; R = repeats per probe = budget // K.")
    print("  'best K' = the K probes with the highest AUC above (optimistic); 'first K' = file order (honest).")
    print(f"  {'probes':>12} {'K':>3} {'R':>4}   pair-AUC (mean)   best-MCC (mean, min)")
    rng = random.Random(0)
    by_auc = sorted(pids, key=lambda p: -aucs[p])
    for K in (2, 3, 4, 5, 10):
        R = args.budget // K
        for name, chosen in (("best K", by_auc[:K]), ("first K", pids[:K])):
            pa, mc = [], []
            for _ in range(args.trials):
                sub = {ep: {pid: rng.sample(data[ep][pid], min(R, len(data[ep][pid]))) for pid in chosen} for ep in eps}
                Dsub = average(per_prompt_distances(sub, args.metric, chosen))
                pa.append(pair_auc(Dsub, truth))
                mc.append(sweep(eps, Dsub, truth, fam_t)[0])
            have = min(len(data[ep][pid]) for ep in eps for pid in chosen)
            warn = "" if have >= R else f"   (only {have} answers collected; run more repeats)"
            print(f"  {name:>12} {K:>3} {R:>4}   {statistics.mean(pa):.3f}            "
                  f"{statistics.mean(mc):.3f}, {min(mc):.3f}{warn}")
    print("\n  Read it like this: pair-AUC near 1.0 means a single threshold could separate same from different.")
    print("  On the real ep endpoints the threshold must be chosen without answers, so best-MCC is an upper bound.")


if __name__ == "__main__":
    main()
