#!/usr/bin/env python
"""Standalone HG Recall@k on the truncated HG test rows (Fig. 3). --hg_model scores the classification head;
--pred_field scores predictions stored in the rows (GPT-4.1-mini or the generative HG).

  python scripts/4_eval/eval_hg_standalone.py --hg_model models/hg_qwen2.5-7b_clf_head --gpu 0
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from kgcq import paths  # noqa: E402
from kgcq.utils import load_json, save_to_json  # noqa: E402
from kgcq.metrics import recall_at_k  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default=str(paths.SOFTLABEL_DIR / "test.json"))
    ap.add_argument("--hg_model", default=None)
    ap.add_argument("--hg_base", default=paths.BASE_LLM)
    ap.add_argument("--kg_nodes", default=str(paths.KG_NODES))
    ap.add_argument("--pred_field", default=None, help="score stored predictions instead of running the HG")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    rows = load_json(args.rows)

    if args.pred_field:
        preds = [r[args.pred_field] for r in rows]
    else:
        assert args.hg_model, "--hg_model or --pred_field is required"
        from kgcq.models import DiseaseDetector, disease_names_from_nodes, load_label_config
        from tqdm import tqdm
        names = load_label_config(args.hg_model) or disease_names_from_nodes(args.kg_nodes)
        hg = DiseaseDetector(args.hg_model, names, base_model=args.hg_base)
        preds = []
        for r in tqdm(rows):
            msgs = []
            for line in r['dialogue_history']:
                if line.startswith("Patient: "):
                    msgs.append({"role": "user", "content": line[len("Patient: "):]})
                elif line.startswith("Doctor: "):
                    msgs.append({"role": "assistant", "content": line[len("Doctor: "):]})
            result, _ = hg.rank_diseases(msgs, top_k=4)
            preds.append([d for d, _ in result['ranked_list']])

    # Two averaging conventions were used in the original code:
    #   all_rows   : average over every row (fraction of gold diseases in the top-k)   [train_hg.py test eval, Fig. 3 "Ours"]
    #   k_ge_gold  : average only over rows with |gold| <= k                           [clf_test.py; Fig. 3 generative/GPT curves]
    metrics = {"n_rows": len(rows)}
    for k in (1, 2, 3, 4):
        vals_all = [recall_at_k(set(r['ground_truth']), p, k) for r, p in zip(rows, preds)]
        vals_sub = [recall_at_k(set(r['ground_truth']), p, k) for r, p in zip(rows, preds) if len(set(r['ground_truth'])) <= k]
        metrics[f"recall_at_{k}"] = float(np.mean(vals_all))
        metrics[f"recall_at_{k}_rows_with_gold_le_k"] = float(np.mean(vals_sub))
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()})
    if args.output:
        save_to_json(metrics, args.output)


if __name__ == "__main__":
    main()
