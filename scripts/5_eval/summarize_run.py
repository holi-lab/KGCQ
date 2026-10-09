#!/usr/bin/env python
"""Per-run analysis of a dialog.json produced by scripts/4_infer/run_dialogue.py.

Adds to metrics.json (written as metrics_anal.json):
  * Recall@4 broken down by patient persona (language proficiency, personality, recall level, confusion level),
    likelihood rating and KG relevance bucket           -> Fig. 5 (robustness) inputs
  * last-turn HG Recall@k (gold within the HG's top-k anchors) and Subgraph Recall (gold disease inside the
    final extracted subgraph)                           -> Table 4 "HG Recall@k" and "Sub Recall"
  * mean number of subgraph lines per turn (subgraph size calibration across n/tau settings)
Also writes dialog_anal.json (flat list, one row per dialogue, used by human-evaluation set construction).

  python scripts/5_eval/summarize_run.py --run runs/kgcq_n2_tau0.005 --profiles data/profiles/test_balanced_275.json
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from kgcq import paths  # noqa: E402
from kgcq.utils import load_json, save_to_json  # noqa: E402

PERSONA_KEYS = ['cefr_type', 'personality_type', 'recall_level_option', 'dazed_level_option']
META_KEYS = ['ground_truth', 'likelihood', 'relevance', 'subgraph'] + PERSONA_KEYS


def recall_at_k(gold, pred, k):
    if not pred:
        return 0.0
    pred = [p.strip().lower() for p in pred]
    gold = [g.strip().lower() for g in gold]
    gold_set = set(gold)
    hit = len(gold_set & set(pred[:k]))
    return hit / len(gold_set) if gold_set else 0.0


def breakdown(all_results):
    buckets = {
        'likelihood': {str(k): [] for k in range(1, 6)},
        'relevance': {str(k): [] for k in range(1, 4)},
        'cefr_level': {k: [] for k in ["A", "B", "C"]},
        'personality': {k: [] for k in ["plain", "verbose", "pleasing", "impatient", "distrust", "overanxious"]},
        'recall_level': {k: [] for k in ["low", "high"]},
        'dazed_level': {k: [] for k in ["normal", "high"]},
    }
    for pid, pdata in all_results.items():
        gold = pdata['ground_truth']
        rel = pdata['relevance']
        rel_level = '1' if rel < 0.3 else ('2' if rel < 0.5 else '3')
        for symptom, item in pdata.items():
            if symptom in META_KEYS:
                continue
            r4 = recall_at_k(gold, item["diagnosis"], 4) if len(gold) <= 4 else 0.0
            buckets['likelihood'][str(pdata['likelihood'])].append(r4)
            buckets['relevance'][rel_level].append(r4)
            buckets['cefr_level'][pdata['cefr_type']].append(r4)
            buckets['personality'][pdata['personality_type']].append(r4)
            buckets['recall_level'][pdata['recall_level_option']].append(r4)
            buckets['dazed_level'][pdata['dazed_level_option']].append(r4)
    out = {}
    for name, groups in buckets.items():
        out[name] = {k: (sum(v) / len(v) if v else 0) for k, v in groups.items()}
        out[name + '_n'] = {k: len(v) for k, v in groups.items()}
    return out


def hg_and_subgraph_recall(all_results):
    """Last-turn HG Recall@k over anchors, subgraph recall (all turns / last turn), mean subgraph lines."""
    hg_recall = {k: [] for k in (1, 2, 3, 4)}
    sub_hits = sub_total = last_hits = last_total = 0
    n_lines = []
    for pid, pdata in all_results.items():
        gold = pdata['ground_truth']
        for symptom, item in pdata.items():
            if symptom in META_KEYS:
                continue
            fp = item['full_process']
            probs = [t for t in fp if t.get('role') == 'disease_prob']
            subs = [t for t in fp if t.get('role') == 'subgraph_extractor']
            if probs:
                pred = [p[0] for p in probs[-1]['content']['top_k_diseases']]
                for k in hg_recall:
                    if k <= len(pred):
                        hg_recall[k].append(recall_at_k(gold, pred, k))
            for t in subs:
                text = ''.join(t['content']) if isinstance(t['content'], list) else str(t['content'])
                n_lines.append(len(t['content']) if isinstance(t['content'], list) else 1)
                sub_total += 1
                sub_hits += int(any(g in text for g in gold))
            if subs:
                text = ''.join(subs[-1]['content']) if isinstance(subs[-1]['content'], list) else str(subs[-1]['content'])
                last_total += 1
                last_hits += int(any(g in text for g in gold))
    return {
        'hg_recall_last_turn': {str(k): (float(np.mean(v)) if v else None) for k, v in hg_recall.items()},
        'subgraph_recall_all_turns': (sub_hits / sub_total) if sub_total else None,
        'subgraph_recall_last_turn': (last_hits / last_total) if last_total else None,
        'avg_subgraph_lines': float(np.mean(n_lines)) if n_lines else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory containing dialog.json")
    ap.add_argument("--profiles", default=str(paths.PROFILE_DIR / "test_balanced_275.json"))
    args = ap.parse_args()

    all_results = load_json(os.path.join(args.run, "dialog.json"))
    profiles = load_json(args.profiles)
    flat = []
    for pid, pdata in all_results.items():
        prof = profiles[pid]
        pdata['likelihood'] = prof['likelihood_rating']
        pdata['relevance'] = len(prof['found_symptoms']) / len(prof['gold_symptoms']) if prof.get('gold_symptoms') else 0.0
        for key in PERSONA_KEYS:
            pdata[key] = prof[key]
        for symptom, item in pdata.items():
            if symptom in META_KEYS:
                continue
            flat.append({"patient_id": pid, "likelihood": pdata['likelihood'], "kg_relevance": pdata['relevance'],
                         **{k: pdata[k] for k in PERSONA_KEYS}, "gold_diseases": pdata['ground_truth'],
                         "pred_diseases": item['diagnosis'], "recall": item['recall'],
                         "dialogue_length": item['dialogue_length'], "dialogue": item['dialogue']})
    save_to_json(flat, os.path.join(args.run, "dialog_anal.json"))

    metrics = load_json(os.path.join(args.run, "metrics.json")) if os.path.exists(os.path.join(args.run, "metrics.json")) else {}
    metrics.update(breakdown(all_results))
    metrics.update(hg_and_subgraph_recall(all_results))
    save_to_json(metrics, os.path.join(args.run, "metrics_anal.json"))
    print("Recall@1-4:", {k: round(v, 3) for k, v in metrics.get('recall', {}).items()}, "| turns", round(metrics.get('avg_turns', 0), 2))
    print("HG Recall@k (last turn):", {k: (round(v, 3) if v is not None else None) for k, v in metrics['hg_recall_last_turn'].items()})
    print("Subgraph recall (last turn):", metrics['subgraph_recall_last_turn'], "| avg lines:", metrics['avg_subgraph_lines'])
    for name in ['cefr_level', 'personality', 'recall_level', 'dazed_level']:
        print(name, {k: round(v, 3) for k, v in metrics[name].items()})


if __name__ == "__main__":
    main()
