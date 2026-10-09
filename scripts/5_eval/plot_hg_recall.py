#!/usr/bin/env python
"""Fig. 3: Recall@k of hypothesis-generation methods on the HG test split.

Values are read from a JSON {"method": [r@1, r@2, r@3, r@4]}; the paper's numbers are in
results/hg_standalone/hg_recall_methods.json.

  python scripts/5_eval/plot_hg_recall.py --input results/hg_standalone/hg_recall_methods.json --output figures/recall_graph.pdf
"""
import argparse
import json
import os

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="recall_graph.pdf")
    args = ap.parse_args()
    data = json.load(open(args.input))
    x = [1, 2, 3, 4]
    plt.figure(figsize=(6.5, 5))
    for model, scores in data.items():
        plt.plot(x, scores, marker="o", linewidth=2, label=model)
    plt.xticks(x, fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel("k Value", fontsize=13)
    plt.ylabel("Recall@k", fontsize=13)
    plt.ylim(0, 1.0)
    plt.grid(True, which="major", axis="both", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=2, fontsize=12, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, bbox_inches="tight", dpi=300)
    print("saved", args.output)


if __name__ == "__main__":
    main()
