"""Recall@k over final diagnoses and run-level aggregation. recall@k is filled only for k >= |gold|."""
import os

from .utils import save_to_json


def recall_at_k(gold, pred, k):
    if not pred:
        return 0.0
    topk_pred = set(pred[:k])
    gold_set = set(gold)
    hit = len(gold_set & topk_pred)
    return hit / len(gold_set) if len(gold_set) > 0 else 0.0


def fill_recall(gold_list, pred_list):
    """Recall@1..4 dict as stored per dialogue (k < |gold| stays 0)."""
    recall = {"1": 0, "2": 0, "3": 0, "4": 0}
    for k in range(len(gold_list), 5):
        recall[str(k)] = recall_at_k(gold_list, pred_list, k)
    return recall


def aggregate_run(all_results, skip_keys=("ground_truth", "subgraph")):
    """Aggregate a dialog.json (pid -> {chief_complaint -> dialogue result}) into run metrics."""
    total_dialogues = 0
    total_turns = 0
    correct_dialogues = {}
    correct_turns = {}
    no_diagnosis = 0
    for pid, patient_data in all_results.items():
        for symptom, result in patient_data.items():
            if symptom in skip_keys:
                continue
            total_dialogues += 1
            total_turns += result["dialogue_length"]
            for k, r in result["recall"].items():
                correct_dialogues.setdefault(str(k), []).append(r)
                if r > 0:
                    correct_turns.setdefault(str(k), []).append(result["dialogue_length"])
            no_diagnosis += 1 if not result["diagnosis"] else 0

    avg_turns = (total_turns / total_dialogues) if total_dialogues > 0 else 0.0
    recall, avg_correct_turns = {}, {}
    for k in ["1", "2", "3", "4"]:
        vals = correct_dialogues.get(k, [])
        recall[k] = sum(vals) / total_dialogues if total_dialogues else 0.0
        hits = sum(vals)
        avg_correct_turns[k] = sum(correct_turns.get(k, [])) / hits if hits > 0 else 0
    return {
        "total_patients": len(all_results),
        "total_dialogues": total_dialogues,
        "recall": recall,
        "avg_turns": avg_turns,
        "avg_turns_correct_dialogues": avg_correct_turns,
        "no_diagnosis": no_diagnosis,
    }


def save_run_metrics(all_results, output_dir):
    metrics = aggregate_run(all_results)
    save_to_json(metrics, os.path.join(output_dir, "metrics.json"))
    return metrics
