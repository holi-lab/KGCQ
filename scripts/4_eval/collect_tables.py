#!/usr/bin/env python
"""Print paper-style rows (Recall@1-4, avg turns) from run directories (metrics.json); default: runs/*.

  python scripts/4_eval/collect_tables.py runs/kgcq_n2_tau0.005 runs/gpt41mini_no_kg
"""
import json
import os
import sys


def main():
    paths = sys.argv[1:] or sorted(os.path.join("runs", d) for d in os.listdir("runs") if os.path.isdir(os.path.join("runs", d)))
    print(f"{'run':48s} {'n':>4s}  R@1    R@2    R@3    R@4   turns")
    for p in paths:
        f = os.path.join(p, "metrics.json") if os.path.isdir(p) else p
        if not os.path.exists(f):
            print(f"{p:48s} (no metrics.json)")
            continue
        m = json.load(open(f))
        r = m['recall']
        extra = ""
        fa = os.path.join(os.path.dirname(f), "metrics_anal.json")
        if os.path.exists(fa):
            a = json.load(open(fa))
            if a.get('last_subgraph_retrieval'):
                extra = f"  sub_recall={a['last_subgraph_retrieval'][2]:.3f}"
            elif a.get('subgraph_recall_last_turn') is not None:
                extra = f"  sub_recall={a['subgraph_recall_last_turn']:.3f}"
        print(f"{os.path.basename(os.path.dirname(f)) if not os.path.isdir(p) else os.path.basename(p):48s} {m['total_dialogues']:4d}  "
              f"{r['1']:.3f}  {r['2']:.3f}  {r['3']:.3f}  {r['4']:.3f}  {m['avg_turns']:5.1f}{extra}")


if __name__ == "__main__":
    main()
