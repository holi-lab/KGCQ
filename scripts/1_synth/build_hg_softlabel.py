#!/usr/bin/env python
"""Build HG training rows (truncated dialogue histories) from synthetic dialogues; only dialogues with
Recall@4 >= 0.5 are used.

  python scripts/1_synth/build_hg_softlabel.py --dialogues data/dialogues/hg_train.json --output data/hg_softlabel/train.json
"""
import argparse
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from tqdm import tqdm  # noqa: E402

from kgcq import paths  # noqa: E402
from kgcq.utils import load_json, load_csv, save_to_json  # noqa: E402

JUDGE_PROMPT = """Act as a strictly logical medical expert.
Your task is to identify the top {n} most likely differential diagnoses from a provided list of candidate diseases, based on the patient's dialogue history.

### Input Data
1. ** Disease Candidates (Total {disease_names_len}):**
{disease_names}

2. **Dialogue History:**
{dialogue_history}

### Instructions
Step 1: Analyze the **Dialogue History**. Extract all reported symptoms (both positive and negative).
Step 2: Based **ONLY** on the extracted symptoms, reason about potential conditions.
Step 3: Scan the **Allowed Disease Candidates** list. Select exactly {n} diseases that match your reasoning.
- **CRITICAL:** Rank the selected diseases in descending order of probability (most likely first).
- **CRITICAL:** You must select diseases **EXACTLY** as they appear in the list. Do not modify names or invent new ones.
- If the symptoms are vague (e.g., just "headache"), select the most common/broad diseases from the list that fit.
- If specific symptoms are present, prioritize diseases that explain those specific features.

### Output Format
Answer in the format ‘disease1, disease2, ..' separated by commas without any additional explanation.
"""


def format_history(dialog_prefix):
    out = []
    for t in dialog_prefix:
        if t['role'] == 'user':
            out.append(f"Patient: {t['content']}")
        elif t['role'] == 'assistant':
            out.append(f"Doctor: {t['content']}")
    return out


def truncate(dialogues, divisor, min_recall4=0.5, seed=None):
    rng = random.Random(seed) if seed is not None else random
    rows = []
    for pid, profile in dialogues.items():
        gt = profile['ground_truth']
        for symptom, result in profile.items():
            if symptom in ('ground_truth', 'subgraph', '_meta') or not isinstance(result, dict) or 'recall' not in result:
                continue
            if result['recall']['4'] < min_recall4:
                continue
            dialog = result['dialogue']
            nums = [i for i in range(2, len(dialog) - 1, 2)]
            if not nums:
                continue
            picks = sorted(rng.sample(nums, min(max(1, len(dialog) // divisor), len(nums))))
            for i in picks:
                rows.append({"patient_id": pid, "symptom": symptom, "dialogue_history": format_history(dialog[:i]),
                             "ground_truth": gt})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dialogues", required=True, help="dialog.json (pid -> {chief complaint -> dialogue})")
    ap.add_argument("--output", required=True)
    ap.add_argument("--divisor", type=int, default=5, help="variants per dialogue = len(dialogue)//divisor (5 for all released splits)")
    ap.add_argument("--judge", default="none", help="API model for predicted_diseases (e.g. gpt-4.1-mini) or 'none'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--kg_nodes", default=str(paths.KG_NODES))
    ap.add_argument("--kg_edges", default=str(paths.KG_EDGES))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = truncate(load_json(args.dialogues), args.divisor, seed=args.seed)
    print(f"{len(rows)} truncated rows")

    if args.judge.lower() != "none":
        from kgcq.graph import DiagnosticKnowledgeGraph
        from kgcq.models import EmbeddingModel, get_openai_response
        kg = DiagnosticKnowledgeGraph(load_csv(args.kg_nodes), load_csv(args.kg_edges),
                                      embedding_model=EmbeddingModel(paths.EMBEDDING_MODEL))
        done = {}
        if os.path.exists(args.output):
            for r in load_json(args.output):
                if 'predicted_diseases' in r:
                    done[(r['patient_id'], r['symptom'], len(r['dialogue_history']))] = r['predicted_diseases']

        def judge(row):
            key = (row['patient_id'], row['symptom'], len(row['dialogue_history']))
            if key in done:
                return done[key]
            inp = {"dialogue_history": "\n".join(row['dialogue_history']), "disease_names": "\n".join(kg.disease_names),
                   "disease_names_len": len(kg.disease_names), "n": 4}
            response = get_openai_response(args.judge, [{"role": "user", "content": JUDGE_PROMPT.format(**inp)}], temperature=0.0)
            return list({kg.find_matching_disease(d.strip())[0] for d in response.split(',')})

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            preds = list(tqdm(ex.map(judge, rows), total=len(rows)))
        for row, pred in zip(rows, preds):
            row['predicted_diseases'] = pred

    save_to_json(rows, args.output)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
