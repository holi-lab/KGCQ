#!/usr/bin/env python
"""Faithfulness audit of the patient simulator (paper Sec. 3.4): share of simulator turns that contradict the
patient profile, from the per-turn annotations of three annotators, plus Gwet's AC1 on the per-turn flags.

  python scripts/human_eval/hallucination_audit.py --csv data/human_eval/hallucination_eval/hallucination_annotations_per_turn.csv
"""
import argparse

import numpy as np
import pandas as pd


def gwet_ac1_binary(flags):
    """flags: n_items x n_raters 0/1 matrix."""
    flags = np.asarray(flags, dtype=float)
    n = flags.shape[1]
    ones = flags.sum(axis=1)
    zeros = n - ones
    P_a = ((ones * (ones - 1) + zeros * (zeros - 1)) / (n * (n - 1))).mean()
    p1 = flags.mean()
    P_e = 2 * p1 * (1 - p1)
    return (P_a - P_e) / (1 - P_e)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--confirmed", type=int, default=None, help="number of flagged turns confirmed as contradictions after adjudication")
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    flag_cols = [c for c in df.columns if c.startswith("flag_")]
    n_turns, n_dialogues = len(df), df['case'].nunique()
    print(f"dialogues {n_dialogues}, patient turns {n_turns}, annotators {len(flag_cols)}")
    for c in flag_cols:
        print(f"  raw flags {c}: {int(df[c].sum())}")
    any_flag = int((df[flag_cols].sum(axis=1) > 0).sum())
    print(f"  turns flagged by >= 1 annotator: {any_flag} ({100*any_flag/n_turns:.2f}%)")
    print(f"  Gwet's AC1 (per-turn flags): {gwet_ac1_binary(df[flag_cols].values):.2f}")
    if args.confirmed is not None:
        print(f"  confirmed contradictions after adjudication: {args.confirmed} ({100*args.confirmed/n_turns:.2f}% of turns)")


if __name__ == "__main__":
    main()
