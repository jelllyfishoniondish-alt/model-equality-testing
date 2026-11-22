"""Plan how to spend the remaining ep budget. PRINTS A PLAN ONLY; it never queries any endpoint.

Idea (adaptive probing): most ep pairs are clearly "same" or clearly "different" already. Extra answers help
only where a pair's distance is close to the model threshold. We bootstrap the answers of each endpoint to get
an interval for every pair distance; pairs whose interval contains the threshold are "uncertain". Endpoints
involved in uncertain pairs get extra answers; the others keep their budget.

How many extra answers? The interval half-width shrinks roughly like 1/sqrt(n). To make the interval stop
touching the threshold we need n_new ~= n_now * (half_width / |distance - threshold|)^2.

Usage (same folder as ep_answers.jsonl, ep_probes.txt, answer_norm.py):
    python plan_topup.py --threshold 0.136 --metric l2u
    python plan_topup.py --threshold 0.61 --metric tv --n-match 8      # if cluster_ep used tv with n_match 8
    python plan_topup.py ... --boot 200                                   # more half-samples, smoother intervals
    python plan_topup.py --threshold 0.136 --metric l2u --family-threshold 0.5 --remaining 60

Use the SAME metric, norm and threshold that cluster_ep.py printed ("using the middle, ...").
--n-match: cluster_ep.py measures tv on random subsets of n answers when ref had fewer repeats; pass that n.
"""
import argparse
import collections
import itertools
import json
import math
import random
from pathlib import Path

from answer_norm import normalise


def load(path, probe_texts, mode):
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    wanted = set(probe_texts)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                p = r["prompt"].strip()
                if r["endpoint"].startswith("ep-") and p in wanted:
                    data[r["endpoint"]][p].append(normalise(r["response"], p, mode))
    return data


def tv(xs, ys):
    cx, cy = collections.Counter(xs), collections.Counter(ys)
    return 0.5 * sum(abs(cx[k] / len(xs) - cy[k] / len(ys)) for k in set(cx) | set(cy))


def l2u(xs, ys):
    cx, cy = collections.Counter(xs), collections.Counter(ys)
    n, m = len(xs), len(ys)
    return (sum(c * (c - 1) for c in cx.values()) / (n * (n - 1)) + sum(c * (c - 1) for c in cy.values()) / (m * (m - 1))
            - 2 * sum(c * cy.get(k, 0) for k, c in cx.items()) / (n * m))


def pair_distance(da, db, probes, metric, n_match, rng, draws):
    vals = []
    for p in probes:
        xs, ys = da.get(p, []), db.get(p, [])
        if len(xs) < 2 or len(ys) < 2:
            continue
        if metric == "l2u":
            vals.append(l2u(xs, ys))
        elif n_match and (len(xs) > n_match or len(ys) > n_match):
            vals.append(sum(tv(rng.sample(xs, min(n_match, len(xs))), rng.sample(ys, min(n_match, len(ys))))
                            for _ in range(draws)) / draws)
        else:
            vals.append(tv(xs, ys))
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="ep_answers.jsonl")
    ap.add_argument("--probes", default="ep_probes.txt")
    ap.add_argument("--norm", default="category", choices=["category", "lenient", "first3"])
    ap.add_argument("--metric", default="l2u", choices=["l2u", "tv"])
    ap.add_argument("--threshold", type=float, required=True, help="model threshold (from cluster_ep.py)")
    ap.add_argument("--family-threshold", type=float, default=None,
                    help="ignore pairs whose distance is above this (clearly different families)")
    ap.add_argument("--n-match", type=int, default=None)
    ap.add_argument("--remaining", type=int, default=60, help="answers left per endpoint")
    ap.add_argument("--boot", type=int, default=100, help="number of half-samples")
    ap.add_argument("--level", type=float, default=0.9, help="bootstrap interval level")
    args = ap.parse_args()

    probes = [l.strip() for l in open(args.probes, encoding="utf-8") if l.strip()]
    data = load(args.answers, probes, args.norm)
    eps = sorted(data)
    if not eps:
        raise SystemExit("No ep answers for these probes.")
    n_now = {e: min(len(data[e][p]) for p in probes) for e in eps}
    rng = random.Random(0)
    draws = 5

    point = {(a, b): pair_distance(data[a], data[b], probes, args.metric, args.n_match, rng, 30)
             for a, b in itertools.combinations(eps, 2)}
    fam_t = args.family_threshold if args.family_threshold is not None else float("inf")
    cand = [k for k, v in point.items() if v <= fam_t]
    print(f"{len(eps)} endpoints, {len(probes)} probes, {min(n_now.values())}-{max(n_now.values())} answers per probe; "
          f"metric {args.metric}, threshold {args.threshold}; {len(cand)} candidate pairs "
          f"(distance <= family threshold {fam_t})")

    # Uncertainty by HALF-SAMPLING: draw half of each endpoint's answers without replacement and recompute.
    # (An ordinary bootstrap with replacement creates duplicate answers, which inflates collision-based
    # distances such as l2u.) For half-samples, the spread around their own mean is about the same as the
    # spread of the full-data estimate around the truth, so we centre that spread on the full-data distance.
    boots = {k: [] for k in cand}
    for _ in range(args.boot):
        res = {e: {p: rng.sample(v, max(2, len(v) // 2)) for p, v in data[e].items()} for e in eps}
        for a, b in cand:
            boots[(a, b)].append(pair_distance(res[a], res[b], probes, args.metric, args.n_match, rng, draws))
    for k, v in boots.items():
        v = [x for x in v if not math.isnan(x)]
        mu = sum(v) / len(v) if v else 0.0
        boots[k] = [point[k] + (x - mu) for x in v]
    lo_q, hi_q = (1 - args.level) / 2, 1 - (1 - args.level) / 2

    uncertain = []
    for k in cand:
        v = sorted(x for x in boots[k] if not math.isnan(x))
        if not v:
            continue
        lo, hi = v[int(lo_q * (len(v) - 1))], v[int(hi_q * (len(v) - 1))]
        if lo <= args.threshold <= hi:
            d = point[k]
            half_w = (hi - lo) / 2
            gap = abs(d - args.threshold)
            factor = (half_w / gap) ** 2 if gap > 1e-9 else float("inf")
            uncertain.append((k, d, lo, hi, factor))

    print(f"\n== Uncertain pairs ({len(uncertain)}): {int(args.level * 100)}% interval contains the threshold ==")
    for (a, b), d, lo, hi, f in sorted(uncertain, key=lambda u: -u[4]):
        side = "same" if d <= args.threshold else "different"
        print(f"  {a} - {b}: distance {d:.3f} [{lo:.3f}, {hi:.3f}] -> currently '{side}', needs ~x{f:.1f} answers")

    need = collections.defaultdict(float)
    for (a, b), d, lo, hi, f in uncertain:
        for e in (a, b):
            need[e] = max(need[e], f)
    print(f"\n== Suggested extra answers per endpoint (at most {args.remaining}; spread evenly over the probes) ==")
    total, hopeless, target = 0, [], collections.defaultdict(list)
    for e in eps:
        if e not in need:
            continue
        extra = math.ceil(n_now[e] * (need[e] - 1)) * len(probes)
        extra = max(len(probes), min(args.remaining, extra))
        extra -= extra % len(probes)
        if need[e] > 1 + args.remaining / max(1, n_now[e] * len(probes)):
            hopeless.append(e)
        total += extra
        target[n_now[e] + extra // len(probes)].append(e)
        print(f"  {e}: +{extra} answers ({extra // len(probes)} more repeats per probe)")
    print(f"  endpoints with no uncertain pair keep their budget: "
          f"{[e for e in eps if e not in need] or 'none'}")
    print(f"  total extra answers: {total}")
    if hopeless:
        print(f"\n  Even the full remaining budget may not settle pairs involving {hopeless} with these probes;")
        print("  consider adding a probe that separates them better (see experiment 3) instead of more repeats.")
    print("\nThis script only prints a plan; nothing was queried. Commands that would carry it out")
    print("(check the budget first; ep_collect.py asks for confirmation before spending):")
    for reps, group in sorted(target.items()):
        print(f"  python ep_collect.py --probes {args.probes} --repeats {reps} --endpoints {','.join(group)}")


if __name__ == "__main__":
    main()
