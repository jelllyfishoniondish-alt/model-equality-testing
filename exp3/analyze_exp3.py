"""Experiment 3 analysis (ref endpoints only): which probes, distances and clustering choices generalise?

Sections
  0. Format quality: invalid rate per probe and endpoint.
  1. Per-probe report: AUC with endpoint-bootstrap CI, separation S(q), split-half stability.
  2. Disguise robustness per probe: distance of every endpoint to ref-01 (same-model disguises should be close,
     fp8/LoRA far). Pay attention to ref-08 (temperature) and ref-03/ref-05 (system prompt / persona).
  3. Greedy forward ablation of probe combinations (A = number 1-100, B = A + animal, then + each new probe),
     scored by held-out MCC (threshold chosen on one half of the repeats, scored on the other), for
     average- and complete-linkage.
  4. Distance-choice comparison on the baseline combination (dist, Dirichlet alpha, number binning).

Distances per probe
  histogram probes: --dist js (default) | tv | hellinger | l2u | mmd_perm (l2u as permutation z-score) | rank
                    (order of the top-10 answer frequencies, temperature-invariant) ; optional --alpha (Dirichlet smoothing over the
                    union of observed answers) and --bins 10 (numbers binned into 10 equal ranges; format classes
                    such as SEQ/CHAT keep their own bucket).
  binary probe:     mean feature vector per endpoint (balance, runs, transitions, 2/3-gram frequencies,
                    autocorrelation, length), distance = RMS over features of the difference divided by its
                    standard error (a noise-calibrated, t-like distance; ~1 when two endpoints are the same).
  Combining probes: every probe's distance matrix is divided by its median over all pairs (no labels used),
                    then averaged.

Usage:
    python analyze_exp3.py                          # defaults
    python analyze_exp3.py --dist l2u
    python analyze_exp3.py --dist mmd_perm --perms 100
    python analyze_exp3.py --dist rank
    python analyze_exp3.py --dist js --alpha 0.5 --bins 10
    python analyze_exp3.py --answers exp3_answers.jsonl --truth practice_answers.csv
Standard library only. Needs answer_norm.py (the exp3 version) in the same folder.
"""
import argparse
import collections
import csv
import itertools
import json
import math
import random
import statistics
from pathlib import Path

from answer_norm import FORMAT_CLASSES, binary_string, normalise, number_range, probe_kind

SHORT_NAMES = {}   # filled in main(): prompt id -> short label


def short_label(prompt):
    """ASCII label for a probe, e.g. 'number 1-100', 'number 1-100 (zh)', 'animal', 'letter', 'binary'."""
    kind = probe_kind(prompt)
    if kind == "binary":
        return "binary"
    if kind == "letter":
        return "letter"
    if kind == "number":
        rng = number_range(prompt)
        lab = f"number {rng[0]}-{rng[1]}" if rng else "number"
        return lab + (" (zh)" if any(ord(c) > 127 for c in prompt) else "")
    words = prompt.lower().split(".")[0].split()
    return words[-1] if words else "word"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(path, norm_mode):
    """Returns prompts {pid: text} and rows[ep][pid] = list of (repeat, raw_text, normalised)."""
    prompts = {}
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            prompts[r["id"]] = r["prompt"]
            rows[r["endpoint"]][r["id"]].append((int(r["repeat"]), r["response"] or "",
                                                 normalise(r["response"], r["prompt"], norm_mode)))
    for ep in rows:
        for pid in rows[ep]:
            rows[ep][pid].sort()
    return prompts, rows


def load_truth(path):
    truth = {}
    for r in csv.reader(open(path, newline="", encoding="utf-8-sig")):
        r = [c.strip() for c in r]
        if len(r) >= 3 and r[0].startswith("ref-"):
            truth[r[0]] = (r[1], r[2], r[3] if len(r) > 3 else "")
    return truth


def half(rows, parity):
    """Keep only even (parity 0) or odd (parity 1) repeats; parity None keeps all."""
    if parity is None:
        return rows
    return {ep: {pid: [x for x in v if x[0] % 2 == parity] for pid, v in d.items()} for ep, d in rows.items()}


# ---------------------------------------------------------------------------
# Histogram distances
# ---------------------------------------------------------------------------
def bin_value(v, prompt, bins):
    rng = number_range(prompt)
    if not bins or rng is None or not v.isdigit():
        return v
    lo, hi = rng
    return f"bin{min(bins - 1, (int(v) - lo) * bins // (hi - lo + 1))}"


def probs(xs, support, alpha):
    c = collections.Counter(xs)
    tot = len(xs) + alpha * len(support)
    return {k: (c[k] + alpha) / tot for k in support}


PERMS = 100   # permutations for mmd_perm (set by --perms)


def l2u(xs, ys):
    cx, cy = collections.Counter(xs), collections.Counter(ys)
    n, m = len(xs), len(ys)
    return (sum(c * (c - 1) for c in cx.values()) / (n * (n - 1))
            + sum(c * (c - 1) for c in cy.values()) / (m * (m - 1))
            - 2 * sum(c * cy.get(k, 0) for k, c in cx.items()) / (n * m))


def rank_distance(xs, ys, k=10):
    """LLMPrint-style rank distance: take the k most frequent answers of the POOLED sample, and count the
    fraction of answer pairs (i, j) whose frequency ORDER differs between the two endpoints (Kendall-tau
    distance; a tie on one side and not the other counts 1/2). Sharpening or flattening a distribution
    (temperature) keeps the order, so this distance ignores temperature. 0 = same order, 1 = reversed."""
    cx, cy = collections.Counter(xs), collections.Counter(ys)
    pooled = collections.Counter(xs) + collections.Counter(ys)
    top = [a for a, _ in sorted(pooled.items(), key=lambda t: (-t[1], str(t[0])))[:k]]
    if len(top) < 2:
        return 0.0
    n, m = len(xs), len(ys)
    bad = tot = 0
    for i, j in itertools.combinations(top, 2):
        sx = (cx[i] / n > cx[j] / n) - (cx[i] / n < cx[j] / n)
        sy = (cy[i] / m > cy[j] / m) - (cy[i] / m < cy[j] / m)
        bad += 1.0 if sx * sy < 0 else 0.5 if sx != sy else 0.0
        tot += 1
    return bad / tot


def hist_distance(xs, ys, dist, alpha):
    if len(xs) < 2 or len(ys) < 2:
        return float("nan")
    if dist == "l2u":   # unbiased one-hot MMD; size-free, ignores alpha
        return l2u(xs, ys)
    if dist == "rank":   # frequency-order distance; temperature-invariant, ignores alpha
        return rank_distance(xs, ys)
    if dist == "mmd_perm":   # l2u as a permutation z-score (Gao et al.); ~0 for the same model
        obs = l2u(xs, ys)
        rng = random.Random(f"{sorted(xs)}|{sorted(ys)}")
        pool, n = list(xs) + list(ys), len(xs)
        null = []
        for _ in range(PERMS):
            rng.shuffle(pool)
            null.append(l2u(pool[:n], pool[n:]))
        mu = sum(null) / PERMS
        sd = math.sqrt(sum((v - mu) ** 2 for v in null) / (PERMS - 1))
        return (obs - mu) / sd if sd > 1e-12 else 0.0
    support = set(xs) | set(ys)
    p, q = probs(xs, support, alpha), probs(ys, support, alpha)
    if dist == "tv":
        return 0.5 * sum(abs(p[k] - q[k]) for k in support)
    if dist == "hellinger":
        return math.sqrt(max(0.0, 1 - sum(math.sqrt(p[k] * q[k]) for k in support)))
    m = {k: 0.5 * (p[k] + q[k]) for k in support}   # Jensen-Shannon, base 2
    return 0.5 * sum(p[k] * math.log2(p[k] / m[k]) for k in support if p[k] > 0) + \
        0.5 * sum(q[k] * math.log2(q[k] / m[k]) for k in support if q[k] > 0)


# ---------------------------------------------------------------------------
# Binary-sequence features (FLIPS-style)
# ---------------------------------------------------------------------------
FEATURES = ["balance", "runs_per_char", "mean_run", "max_run", "p01", "p10"] + \
    [f"g2_{a}{b}" for a in "01" for b in "01"] + [f"g3_{a}{b}{c}" for a in "01" for b in "01" for c in "01"] + \
    ["ac1", "ac2", "ac3", "length"]


def binary_features(s):
    n = len(s)
    if n < 10:
        return None
    runs = 1 + sum(s[i] != s[i - 1] for i in range(1, n))
    run_lengths, cur = [], 1
    for i in range(1, n):
        if s[i] == s[i - 1]:
            cur += 1
        else:
            run_lengths.append(cur)
            cur = 1
    run_lengths.append(cur)
    zeros = [i for i in range(n - 1) if s[i] == "0"]
    ones = [i for i in range(n - 1) if s[i] == "1"]
    f = {
        "balance": s.count("1") / n,
        "runs_per_char": runs / n,
        "mean_run": statistics.mean(run_lengths),
        "max_run": max(run_lengths),
        "p01": sum(s[i + 1] == "1" for i in zeros) / len(zeros) if zeros else 0.5,
        "p10": sum(s[i + 1] == "0" for i in ones) / len(ones) if ones else 0.5,
        "length": n,
    }
    for k in (2, 3):
        grams = collections.Counter(s[i:i + k] for i in range(n - k + 1))
        tot = max(1, n - k + 1)
        for g in itertools.product("01", repeat=k):
            f[f"g{k}_{''.join(g)}"] = grams["".join(g)] / tot
    x = [1 if c == "1" else -1 for c in s]
    for lag in (1, 2, 3):
        f[f"ac{lag}"] = sum(x[i] * x[i + lag] for i in range(n - lag)) / (n - lag)
    return f


def binary_distance(ra, rb):
    """ra, rb: lists of raw responses. RMS over features of (mean_a - mean_b) / standard error."""
    fa = [binary_features(b) for b in (binary_string(t) for t in ra) if b]
    fb = [binary_features(b) for b in (binary_string(t) for t in rb) if b]
    fa, fb = [f for f in fa if f], [f for f in fb if f]
    if len(fa) < 3 or len(fb) < 3:
        return float("nan")
    zs = []
    for k in FEATURES:
        a, b = [f[k] for f in fa], [f[k] for f in fb]
        se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
        zs.append(((statistics.mean(a) - statistics.mean(b)) / se) ** 2 if se > 1e-12 else 0.0)
    return math.sqrt(sum(zs) / len(zs))


# ---------------------------------------------------------------------------
# Distance matrices
# ---------------------------------------------------------------------------
def probe_matrix(rows, pid, prompt, eps, opts):
    """D[(a, b)] for one probe."""
    D = {}
    is_bin = probe_kind(prompt) == "binary"
    for a, b in itertools.combinations(eps, 2):
        xa, xb = rows[a].get(pid, []), rows[b].get(pid, [])
        if is_bin:
            D[(a, b)] = binary_distance([t for _, t, _ in xa], [t for _, t, _ in xb])
        else:
            va = [bin_value(v, prompt, opts.bins) for _, _, v in xa]
            vb = [bin_value(v, prompt, opts.bins) for _, _, v in xb]
            D[(a, b)] = hist_distance(va, vb, opts.dist, opts.alpha)
    return D


def combine(mats):
    """Divide each probe's matrix by its median (scale-free, label-free), then average over probes."""
    out = {}
    scaled = []
    for D in mats:
        vals = [v for v in D.values() if not math.isnan(v)]
        med = statistics.median(vals) if vals else 1.0
        med = med if med > 1e-9 else 1.0   # z-score or l2u medians can be <= 0; never flip signs
        scaled.append({k: v / med for k, v in D.items()})
    for k in scaled[0]:
        vals = [S[k] for S in scaled if not math.isnan(S[k])]
        out[k] = sum(vals) / len(vals) if vals else float("nan")
    return out


def dist(D, a, b):
    return 0.0 if a == b else D[(a, b)] if (a, b) in D else D[(b, a)]


# ---------------------------------------------------------------------------
# Clustering and scoring
# ---------------------------------------------------------------------------
def cluster(items, D, thr, linkage):
    clusters = [[x] for x in items]
    agg = max if linkage == "complete" else (lambda v: sum(v) / len(v))
    while len(clusters) > 1:
        best = None
        for i, j in itertools.combinations(range(len(clusters)), 2):
            d = agg([dist(D, a, b) for a in clusters[i] for b in clusters[j]])
            if best is None or d < best[0]:
                best = (d, i, j)
        if best[0] > thr:
            break
        clusters[best[1]] += clusters.pop(best[2])
    return clusters


def labels(items, D, thr, linkage):
    return {e: i for i, c in enumerate(cluster(items, D, thr, linkage)) for e in c}


def mcc(lab, truth, items):
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(items, 2):
        t, g = truth[a][:2] == truth[b][:2], lab[a] == lab[b]
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / d) if d else 0.0


def best_threshold(items, D, truth, linkage):
    """Middle of the threshold range that gives the best MCC."""
    vals = sorted(set(v for v in D.values() if not math.isnan(v)))
    if not vals:
        return 0.0, 0.0
    mids = [vals[0] - 1e-9] + [(x + y) / 2 for x, y in zip(vals, vals[1:])] + [vals[-1] + 1e-9]
    scored = [(mcc(labels(items, D, t, linkage), truth, items), t) for t in mids]
    top = max(s for s, _ in scored)
    good = [t for s, t in scored if s == top]
    return top, (min(good) + max(good)) / 2


def pair_auc(D, truth):
    same = [v for (a, b), v in D.items() if not math.isnan(v) and truth[a][:2] == truth[b][:2]]
    diff = [v for (a, b), v in D.items() if not math.isnan(v) and truth[a][:2] != truth[b][:2]]
    if not same or not diff:
        return float("nan")
    return sum((s < t) + 0.5 * (s == t) for s in same for t in diff) / (len(same) * len(diff))


def boot_auc(D, truth, eps, rng, n=200):
    out = []
    for _ in range(n):
        s = [rng.choice(eps) for _ in eps]
        same, diff = [], []
        for i, j in itertools.combinations(range(len(s)), 2):
            a, b = s[i], s[j]
            if a == b:
                continue
            v = dist(D, a, b)
            if not math.isnan(v):
                (same if truth[a][:2] == truth[b][:2] else diff).append(v)
        if same and diff:
            out.append(sum((x < y) + 0.5 * (x == y) for x in same for y in diff) / (len(same) * len(diff)))
    out.sort()
    return (out[int(0.025 * (len(out) - 1))], out[int(0.975 * (len(out) - 1))]) if out else (float("nan"),) * 2


def separation(D, truth):
    same = [v for (a, b), v in D.items() if not math.isnan(v) and truth[a][:2] == truth[b][:2]]
    diff = [v for (a, b), v in D.items() if not math.isnan(v) and truth[a][:2] != truth[b][:2]]
    return statistics.median(diff) - statistics.median(same) if same and diff else float("nan")


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    if len(pairs) < 3:
        return float("nan")
    rx, ry = ranks([a for a, _ in pairs]), ranks([b for _, b in pairs])
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


# ---------------------------------------------------------------------------
# Held-out evaluation of a probe combination
# ---------------------------------------------------------------------------
def heldout(halves_mats, combo, eps, truth, linkage):
    """Threshold chosen on one half, MCC measured on the other; both directions, averaged."""
    res = []
    for sel, ev in ((0, 1), (1, 0)):
        Ds = combine([halves_mats[sel][p] for p in combo])
        De = combine([halves_mats[ev][p] for p in combo])
        _, t = best_threshold(eps, Ds, truth, linkage)
        res.append(mcc(labels(eps, De, t, linkage), truth, eps))
    return statistics.mean(res), res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="exp3_answers.jsonl")
    ap.add_argument("--truth", default="practice_answers.csv")
    ap.add_argument("--norm", default="category", choices=["category", "lenient", "first3"])
    ap.add_argument("--dist", default="js", choices=["js", "tv", "hellinger", "l2u", "mmd_perm", "rank"])
    ap.add_argument("--perms", type=int, default=100, help="permutations per pair for --dist mmd_perm")
    ap.add_argument("--alpha", type=float, default=0.0, help="Dirichlet smoothing for js/tv/hellinger")
    ap.add_argument("--bins", type=int, default=0, help="bin number answers into this many ranges (0 = off)")
    ap.add_argument("--anchor", default="ref-01")
    ap.add_argument("--no-compare", action="store_true", help="skip section 4 (distance-choice comparison)")
    opts = ap.parse_args()
    global PERMS
    PERMS = opts.perms

    prompts, rows = load(opts.answers, opts.norm)
    truth = load_truth(opts.truth)
    eps = sorted(e for e in rows if e in truth)
    pids = sorted(prompts, key=lambda s: int(s) if s.isdigit() else s)
    for pid in pids:
        SHORT_NAMES[pid] = f"[{pid}] {short_label(prompts[pid])}"
    rng = random.Random(0)
    print(f"{len(eps)} ref endpoints, {len(pids)} probes; dist={opts.dist} alpha={opts.alpha} bins={opts.bins} "
          f"norm={opts.norm}\n")

    # ---- 0. Format quality ----
    print("== 0. Valid-answer rate per probe (rows) and endpoint (columns) ==")
    print(" " * 38 + " ".join(f"{e[-2:]:>5}" for e in eps))
    for pid in pids:
        cells = []
        for e in eps:
            v = rows[e].get(pid, [])
            ok = sum(1 for _, _, x in v if x not in FORMAT_CLASSES)
            cells.append(f"{ok / len(v):5.2f}" if v else "    -")
        n = min(len(rows[e].get(pid, [])) for e in eps)
        print(f"  {SHORT_NAMES[pid]:<34}{n:>3} " + " ".join(cells))
    print("  (first number = fewest answers any endpoint has for that probe)\n")

    # ---- matrices: full data and the two halves ----
    full = {pid: probe_matrix(rows, pid, prompts[pid], eps, opts) for pid in pids}
    halves_rows = [half(rows, 0), half(rows, 1)]
    halves_mats = [{pid: probe_matrix(h, pid, prompts[pid], eps, opts) for pid in pids} for h in halves_rows]

    # ---- 1. Per-probe report ----
    print("== 1. Per-probe discrimination (same-model vs different-model ref pairs) ==")
    print("  AUC [95% endpoint-bootstrap CI] | S(q)=median diff - median same | AUC even/odd half | "
          "stability = Spearman of pair distances between halves")
    keys = sorted(full[pids[0]])
    for pid in sorted(pids, key=lambda p: -pair_auc(full[p], truth) if not math.isnan(pair_auc(full[p], truth)) else 0):
        D = full[pid]
        lo, hi = boot_auc(D, truth, eps, rng)
        a0, a1 = pair_auc(halves_mats[0][pid], truth), pair_auc(halves_mats[1][pid], truth)
        stab = spearman([halves_mats[0][pid][k] for k in keys], [halves_mats[1][pid][k] for k in keys])
        print(f"  {SHORT_NAMES[pid]:<34} AUC {pair_auc(D, truth):.2f} [{lo:.2f},{hi:.2f}]  S {separation(D, truth):+.3f}"
              f"  halves {a0:.2f}/{a1:.2f}  stability {stab:+.2f}")
    print()

    # ---- 2. Disguise robustness ----
    anc = opts.anchor
    others = [e for e in eps if e != anc]
    print(f"== 2. Distance to {anc} per probe (same-model disguises should be SMALL, fp8/LoRA LARGE) ==")
    print(" " * 38 + " ".join(f"{e[-2:]}{'s' if truth[e][:2] == truth[anc][:2] else 'd'}".rjust(6) for e in others)
          + "  margin")
    for pid in pids:
        D = full[pid]
        vals = {e: dist(D, anc, e) for e in others}
        same = [vals[e] for e in others if truth[e][:2] == truth[anc][:2] and not math.isnan(vals[e])]
        diff = [vals[e] for e in others if truth[e][:2] != truth[anc][:2] and not math.isnan(vals[e])]
        margin = min(diff) - max(same) if same and diff else float("nan")
        print(f"  {SHORT_NAMES[pid]:<36}" + " ".join(f"{vals[e]:6.3f}" for e in others) + f"  {margin:+.3f}")
    kinds = {e: truth[e][2] for e in others}
    print("  columns: s = same model as anchor, d = different. " +
          "; ".join(f"{e[-2:]}={k}" for e, k in kinds.items()))
    print("  margin = smallest 'd' minus largest 's' (> 0: this probe alone separates everything from the anchor)\n")

    # ---- 2b. Watermark check ----
    s_group = [e for e in eps if truth[e][:2] == truth[anc][:2]]
    wm = next((e for e in s_group if "watermark" in truth[e][2].lower()), "ref-06" if "ref-06" in s_group else None)
    if wm and len(s_group) >= 3:
        rest_s = [e for e in s_group if e != wm]
        print(f"== 2b. Watermark check: is {wm} the odd one out inside its own model group {[e[-2:] for e in s_group]}? ==")
        print("  A Kirchenbauer-style watermark boosts a pseudo-random 'green list' of tokens, which can shift the")
        print(f"  first-token histogram. For each probe: mean distance of {wm} to the other same-model endpoints, versus")
        print("  the same number for every other member (their spread without the watermark endpoint).")
        print(f"  {'probe':<36}{wm[-2:]+'->S':>8}{'others: min':>13}{'max':>8}  flag")
        flagged = []
        for pid in pids:
            D = full[pid]
            def mean_to(e):
                v = [dist(D, e, o) for o in s_group if o != e and not math.isnan(dist(D, e, o))]
                return sum(v) / len(v) if v else float("nan")
            w = mean_to(wm)
            oth = [mean_to(e) for e in rest_s]
            oth = [v for v in oth if not math.isnan(v)]
            flag = ""
            if oth and not math.isnan(w) and w > max(oth):
                flag = "OUTLIER" + (" (strong)" if w > max(oth) + (max(oth) - min(oth)) else "")
                flagged.append(SHORT_NAMES[pid])
            print(f"  {SHORT_NAMES[pid]:<36}{w:8.3f}{(min(oth) if oth else float('nan')):13.3f}"
                  f"{(max(oth) if oth else float('nan')):8.3f}  {flag}")
        print(f"  -> {wm} is the outlier on: {flagged or 'no probe'}. With {len(s_group)} members, one of them is the")
        print("     largest by chance on ~1/%d of probes; worry only if it is flagged (strong) on most probes.\n" % len(s_group))

    # ---- 3. Greedy forward ablation ----
    print("== 3. Probe combinations: held-out MCC (threshold picked on one half of the repeats, scored on the other) ==")
    num100 = next((p for p in pids if "between 1 and 100" in prompts[p]), pids[0])
    animal = next((p for p in pids if "animal" in prompts[p].lower()), None)
    for link in ("average", "complete"):
        print(f"  -- {link}-linkage --")
        base = [num100]
        m, parts = heldout(halves_mats, base, eps, truth, link)
        print(f"    A  {[SHORT_NAMES[p] for p in base]}: held-out MCC {m:.3f} ({parts[0]:.3f}/{parts[1]:.3f})")
        if animal:
            base = [num100, animal]
            m, parts = heldout(halves_mats, base, eps, truth, link)
            print(f"    B  A + animal: held-out MCC {m:.3f} ({parts[0]:.3f}/{parts[1]:.3f})")
        cur, cur_m = list(base), m
        rest = [p for p in pids if p not in base]
        for p in rest:
            m2, parts2 = heldout(halves_mats, cur[:2] + [p], eps, truth, link)
            print(f"    C  B + {SHORT_NAMES[p]:<34}: held-out MCC {m2:.3f} ({parts2[0]:.3f}/{parts2[1]:.3f})")
        path = []
        while rest:
            scored = sorted(((heldout(halves_mats, cur + [p], eps, truth, link)[0], p) for p in rest), reverse=True)
            if scored[0][0] <= cur_m:
                break
            cur_m, p = scored[0]
            cur.append(p)
            rest.remove(p)
            path.append(f"+{SHORT_NAMES[p]} -> {cur_m:.3f}")
        print("    greedy from B: " + (" | ".join(path) if path else "no probe improves on B"))
        Dfull = combine([full[p] for p in cur])
        top, t = best_threshold(eps, Dfull, truth, link)
        print(f"    chosen set on ALL repeats: MCC {top:.3f} at threshold {t:.3f} (optimistic), "
              f"pair-AUC {pair_auc(Dfull, truth):.3f}")
    print("  Note: with only 9 ref endpoints, differences below ~0.1 MCC are within noise. The greedy path also")
    print("  CHOOSES probes by held-out MCC, so its final number is optimistic; trust the A/B/C rows more.\n")

    # ---- 4. Distance-choice comparison on the baseline combination ----
    if opts.no_compare or not animal:
        return
    print("== 4. Distance choices on B (number 1-100 + animal): held-out MCC, average / complete linkage ==")
    for d_, a_, b_ in (("tv", 0, 0), ("js", 0, 0), ("js", 0.5, 0), ("hellinger", 0.5, 0), ("l2u", 0, 0), ("mmd_perm", 0, 0), ("rank", 0, 0),
                       ("js", 0.5, 10), ("l2u", 0, 10)):
        o = argparse.Namespace(**{**vars(opts), "dist": d_, "alpha": a_, "bins": b_})
        hm = [{p: probe_matrix(h, p, prompts[p], eps, o) for p in (num100, animal)} for h in halves_rows]
        r = [heldout(hm, [num100, animal], eps, truth, link)[0] for link in ("average", "complete")]
        print(f"  dist={d_:<9} alpha={a_:<3} bins={b_:<2}: {r[0]:.3f} / {r[1]:.3f}")


if __name__ == "__main__":
    main()
