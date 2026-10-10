#!/usr/bin/env python
"""Fig. 5: Recall@4 by patient persona for GPT-only, GPT+KG and KGCQ (reads metrics_anal.json from summarize_run.py).

  python scripts/4_eval/plot_robustness.py --gpt_only runs/gpt41mini_no_kg --gpt_kg runs/gpt41mini_kg_hg \
      --ours runs/kgcq_n2_tau0.005 --output figures/robustness.pdf
"""
import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

METRICS = ['cefr_level', 'personality', 'recall_level', 'dazed_level']
SORTERS = {"cefr_level": ["low", "medium", "high"],
           "personality": ["plain", "verbose", "distrust", "pleasing", "impatient", "overanxious"],
           "recall_level": ["low", "high"], "dazed_level": ["normal", "high"]}
TITLES = {"cefr_level": "Language Proficiency", "personality": "Personality",
          "recall_level": "Recall Level", "dazed_level": "Confusion Level"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpt_only", required=True)
    ap.add_argument("--gpt_kg", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--output", default="robustness.pdf")
    args = ap.parse_args()

    def load(run):
        return json.load(open(os.path.join(run, "metrics_anal.json")))

    models = {"GPT-only": load(args.gpt_only), "GPT+KG": load(args.gpt_kg), "SFT+KG (Ours)": load(args.ours)}
    rows = []
    for model, data in models.items():
        for metric in METRICS:
            for category, score in data[metric].items():
                if metric == 'cefr_level':
                    category = {'A': 'low', 'B': 'medium', 'C': 'high'}[category]
                rows.append({"Model": model, "Metric": metric, "Category": category, "Score": score})
    df = pd.DataFrame(rows)

    mpl.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif", "Liberation Serif", "serif"],
                         "font.size": 16, "axes.titlesize": 18, "axes.labelsize": 15, "xtick.labelsize": 16,
                         "ytick.labelsize": 16, "legend.fontsize": 16, "mathtext.fontset": "stix"})
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6), sharey=True)
    for i, metric in enumerate(METRICS):
        ax = axes.flatten()[i]
        subset = df[df["Metric"] == metric]
        sns.barplot(data=subset, x="Category", y="Score", hue="Model", order=SORTERS[metric], ax=ax, palette="viridis")
        ax.set_title(TITLES[metric], pad=8)
        ax.set_xlabel("")
        ax.set_ylabel("Recall@4" if i % 2 == 0 else "")
        ax.set_ylim(0, 0.7)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6])
        if metric == "personality":
            ax.tick_params(axis="x", rotation=30)
            for label in ax.get_xticklabels():
                label.set_ha("right")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, bbox_inches='tight', dpi=300)
    print("saved", args.output)


if __name__ == "__main__":
    main()
