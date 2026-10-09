#!/usr/bin/env python3
"""Aggregate results/*.json -> faithfulness-audit numbers.

Prints, per evaluator and overall: # patient turns reviewed (N), # flagged as
hallucination, hallucination rate (X.X%), and a dump of memos. These fill the
[N] / [X.X]% placeholders in writing/report.md and the Limitations footnote.
"""
import os, json, glob, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "human_eval", "hallucination_eval")  # eval_set.json, data.json, translations
_EVAL = os.path.join(DATA_DIR, "eval_set.json")
DATA = json.load(open(_EVAL if os.path.exists(_EVAL) else os.path.join(DATA_DIR, "data.json"), encoding="utf-8"))
N_PATIENT_TURNS = sum(len(it["turns"]) for it in DATA)


def main():
    files = sorted(glob.glob(os.path.join(HERE, "results", "*.json")))
    if not files:
        print("no results yet."); return
    overall_reviewed = overall_flagged = 0
    memos = []
    print(f"{'evaluator':<20}{'items_done':>11}{'turns_reviewed':>16}{'hallucinated':>14}{'rate':>9}")
    print("-" * 70)
    for f in files:
        r = json.load(open(f, encoding="utf-8"))
        name = r.get("evaluator", os.path.basename(f)[:-5])
        items = r.get("items", {})
        done = sum(1 for v in items.values() if v.get("done"))
        reviewed = flagged = 0
        for idx, it in items.items():
            for tn, tt in it.get("turns", {}).items():
                # count a turn as reviewed if explicitly checked OR the item was completed
                reviewed_turn = it.get("done") or tt.get("hallucination") or tt.get("memo")
                if reviewed_turn:
                    reviewed += 1
                if tt.get("hallucination"):
                    flagged += 1
                    if tt.get("memo"):
                        memos.append((name, int(idx), int(tn), tt["memo"]))
        # if items marked done, count all their patient turns as reviewed
        reviewed = sum(len(DATA[int(idx)]["turns"]) for idx, it in items.items() if it.get("done"))
        flagged = sum(1 for it in items.values() if it.get("done")
                      for tt in it.get("turns", {}).values() if tt.get("hallucination"))
        rate = (100 * flagged / reviewed) if reviewed else 0.0
        overall_reviewed += reviewed; overall_flagged += flagged
        print(f"{name:<20}{done:>11}{reviewed:>16}{flagged:>14}{rate:>8.1f}%")
    print("-" * 70)
    orate = (100 * overall_flagged / overall_reviewed) if overall_reviewed else 0.0
    print(f"{'OVERALL':<20}{'':>11}{overall_reviewed:>16}{overall_flagged:>14}{orate:>8.1f}%")
    print(f"\nFaithfulness audit  ->  N = {overall_reviewed} patient turns "
          f"(of {N_PATIENT_TURNS} total over {len(DATA)} dialogues), "
          f"hallucination rate = {orate:.1f}%")
    if memos:
        print(f"\n--- memos on flagged turns ({len(memos)}) ---")
        for name, idx, tn, m in memos:
            print(f"  [{name}] case {idx+1}, turn {tn+1}: {m}")


if __name__ == "__main__":
    main()
