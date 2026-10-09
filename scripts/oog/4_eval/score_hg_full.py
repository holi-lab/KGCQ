"""exp3/exp4 HG 전체 스코어링 — valid + test, recall@1~4, ID/OOD 분리.

valid: HG 모델(base+adapter) 로드 후 valid_combined 예측 → @1-4 ID/OOD.
test : 학습 때 저장된 test_predictions.json(top4 순위) 활용 → @1-4 ID/OOD (GPU 불필요).
ID/OOD: exp3 = gold 'Other'면 OOD / exp4 = gold 에 paper 밖 질병(rescued) 있으면 OOD.
출력: 4_eval/results_hg_full.json
사용: python score_hg_full.py [gpu]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import sys, os, json
GPU = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

RV2 = P.RV2
MODEL_ID = f'{P.BASE_MODEL}'   # 로컬 base (이식용)
RATIOS = [0, 5, 10, 15, 20]


def dl(csv):
    df = pd.read_csv(csv)
    return [r['name'] for _, r in df.iterrows() if r['label'] == 'Disease']


PAPER = set(str(n).strip().lower() for n in dl(f'{P.KG_DIR}/paper/nodes.csv'))


def labels_of(exp):
    return dl(f'{P.KG_DIR}/paper/nodes.csv') + ['Other'] if exp == 'exp3' else dl(f'{P.KG_DIR}/augmented/nodes.csv')


def is_ood(gold_lower, exp):
    return ('other' in gold_lower) if exp == 'exp3' else any(d not in PAPER for d in gold_lower)


def rec(true, ranked, k):
    return len(set(true) & set(ranked[:k])) / len(true) if true else None


def avg_buckets(items, exp):
    B = {'ID': {1: [], 2: [], 3: [], 4: []}, 'OOD': {1: [], 2: [], 3: [], 4: []}}
    for gold, ranked in items:
        if not gold:
            continue
        b = 'OOD' if is_ood(gold, exp) else 'ID'
        for k in (1, 2, 3, 4):
            v = rec(gold, ranked, k)
            if v is not None:
                B[b][k].append(v)
    out = {}
    for b, ks in B.items():
        out[b] = {f'recall@{k}': (round(float(np.mean(v)), 4) if v else None) for k, v in ks.items()}
        out[b]['n'] = len(ks[1])
    return out


tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="right")


def score_valid(exp, ratio, id2name):
    adapter = f'{P.MODEL_DIR}/{exp}_hg_r{ratio:02d}_Qwen2.5-7B'
    if not os.path.exists(f'{adapter}/adapter_config.json'):
        return None
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto",
        num_labels=len(id2name), problem_type="multi_label_classification")
    base.resize_token_embeddings(len(tok))
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    valid = json.load(open(f'{P.TRAIN_DATA_DIR}/data_{exp}/valid_combined.json'))
    items = []
    BS = 8
    with torch.no_grad():
        for i in range(0, len(valid), BS):
            chunk = valid[i:i + BS]
            texts = ["Conversation:\n" + "\n".join(x['dialogue_history']) +
                     "\n\nBased on the conversation, what diseases does the patient have?" for x in chunk]
            enc = tok(texts, truncation=True, max_length=2048, padding=True, return_tensors="pt").to(model.device)
            logits = model(**enc).logits.float().cpu().numpy()
            for j, x in enumerate(chunk):
                ranked = [id2name[idx] for idx in np.argsort(-logits[j])]
                gold = [str(d).strip().lower() for d in set(x['ground_truth'])]
                items.append((gold, [r.lower() for r in ranked]))
    del model, base
    torch.cuda.empty_cache()
    return avg_buckets(items, exp)


def score_test_saved(exp, ratio):
    p = f'{P.MODEL_DIR}/{exp}_hg_r{ratio:02d}_Qwen2.5-7B/test_predictions.json'
    if not os.path.exists(p):
        return None
    preds = json.load(open(p))
    items = [([str(d).strip().lower() for d in x['true_labels']],
              [str(d).strip().lower() for d in x['top4']]) for x in preds]
    return avg_buckets(items, exp)


def main():
    out = {}
    for exp in ['exp3', 'exp4']:
        names = labels_of(exp)
        id2name = {i: n for i, n in enumerate(names)}
        for ratio in RATIOS:
            print(f"[{exp} r{ratio}] valid 예측...", flush=True)
            v = score_valid(exp, ratio, id2name)
            t = score_test_saved(exp, ratio)
            out[f'{exp}_r{ratio:02d}'] = {'valid': v, 'test': t}
            if v:
                print(f"  valid ID@4={v['ID']['recall@4']} OOD@4={v['OOD']['recall@4']} | "
                      f"test ID@4={t['ID']['recall@4']} OOD@4={t['OOD']['recall@4']}", flush=True)
    json.dump(out, open(f'{P.RESULTS_DIR}/results_hg_full.json', 'w'), ensure_ascii=False, indent=2)
    print("\n저장 → 4_eval/results_hg_full.json")


if __name__ == '__main__':
    main()
