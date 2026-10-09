#!/usr/bin/env python
"""Patient-simulator preference study statistics (paper Sec. 6.3, Appendix Table 12).

Input CSV (data/human_eval/simulator_eval/preference_labels.csv): one row per scenario with the three physicians'
choices coded 1 = "Response A", 2 = "Response B", 3 = "Comparable", and a ``gold`` column telling which letter
was produced by OUR simulator (responses were shown in randomized order).

Reports per-physician preference for ours, pairwise agreement, Fleiss' kappa and Gwet's AC1, on all scenarios and
on the subset where physician 1 gave a definitive (non-"comparable") judgment.

  python scripts/human_eval/simulator_preference_stats.py --csv data/human_eval/simulator_eval/preference_labels.csv
"""
import argparse
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd

CATS = ["ours", "baseline", "tie"]


def recode(row, cols):
    out = []
    for c in cols:
        v = int(row[c])
        if v == 3:
            out.append("tie")
        else:
            letter = "A" if v == 1 else "B"
            out.append("ours" if letter == row["gold"] else "baseline")
    return out


def fleiss_kappa(table):
    """table: n_items x n_categories counts."""
    table = np.asarray(table, dtype=float)
    n = table.sum(axis=1)[0]
    N = table.shape[0]
    p_j = table.sum(axis=0) / (N * n)
    P_i = ((table ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar, P_e = P_i.mean(), (p_j ** 2).sum()
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else float('nan')


def gwet_ac1(table):
    table = np.asarray(table, dtype=float)
    n = table.sum(axis=1)[0]
    N, q = table.shape
    p_j = table.sum(axis=0) / (N * n)
    P_a = (((table ** 2).sum(axis=1) - n) / (n * (n - 1))).mean()
    P_e = (p_j * (1 - p_j)).sum() / (q - 1)
    return (P_a - P_e) / (1 - P_e)


def report(df, cols, title):
    labels = [recode(r, cols) for _, r in df.iterrows()]
    print(f"\n=== {title} (n={len(labels)}) ===")
    for i, c in enumerate(cols):
        pref = sum(1 for l in labels if l[i] == "ours") / len(labels)
        print(f"  pref. ours {c}: {100*pref:.1f}%")
    for i, j in combinations(range(len(cols)), 2):
        agree = sum(1 for l in labels if l[i] == l[j]) / len(labels)
        print(f"  agreement {cols[i]}-{cols[j]}: {100*agree:.1f}%")
    table = [[Counter(l)[c] for c in CATS] for l in labels]
    print(f"  Gwet's AC1 = {gwet_ac1(table):.2f}   Fleiss' kappa = {fleiss_kappa(table):.2f}")
    maj = Counter()
    for l in labels:
        cnt = Counter(l)
        top, k = cnt.most_common(1)[0]
        maj[top if k >= 2 else "no majority"] += 1
    print("  majority:", dict(maj))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    cols = [c for c in df.columns if c.startswith("physician")]
    report(df, cols, "all scenarios")
    sub = df[df[cols[0]] != 3]
    report(sub, cols, f"scenarios where {cols[0]} gave a definitive judgment")


if __name__ == "__main__":
    main()
