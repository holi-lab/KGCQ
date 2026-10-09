"""paper HG 분류기(model/1220_clf_head_Qwen2.5-7B-Instruct, 338-label full model)를
우리 exp34 valid_combined / test_combined 에 돌려 recall@1~4 (ID/OOD 분리) 측정.
- ID = gold ∈ paper 338 (label space) / OOD = gold ∉ 338 (paper 구조상 예측 불가 → recall 0)
- 프롬프트·max_length(2048)·recall 정의 = paper clf_head_train.py 와 동일.
사용: python eval_paper_hg.py [gpu]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import sys, os, json
GPU = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = P.RV2
# paper HG (merged classification-head model). Default = main pipeline output; override with PAPER_HG_DIR.
MDIR = os.environ.get("PAPER_HG_DIR") or (
    f"{P.MODEL_DIR}/1220_clf_head_Qwen2.5-7B-Instruct" if os.path.isdir(f"{P.MODEL_DIR}/1220_clf_head_Qwen2.5-7B-Instruct")
    else f"{P.ROOT}/models/hg_qwen2.5-7b_clf_head")

cfg = json.load(open(f"{MDIR}/config.json"))
id2label = {int(k): v for k, v in cfg["id2label"].items()}
label2id = {str(v).strip().lower(): k for k, v in id2label.items()}   # 소문자 매칭
print(f"paper label space: {len(id2label)} diseases", flush=True)

tok = AutoTokenizer.from_pretrained(MDIR, padding_side="right")
model = AutoModelForSequenceClassification.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()


def recall_at_k(gold_idx, ranked, k):
    if not gold_idx:
        return None
    return len(set(gold_idx) & set(ranked[:k].tolist())) / len(gold_idx)


def evaluate(path, tag):
    data = json.load(open(path))
    r = {1: [], 2: [], 3: [], 4: []}
    rid = {1: [], 2: [], 3: [], 4: []}
    rood = {1: [], 2: [], 3: [], 4: []}
    with torch.no_grad():
        for item in data:
            gold = [str(g).strip().lower() for g in (item.get("ground_truth") or [])]
            if not gold:
                continue
            gold_idx = [label2id[g] for g in gold if g in label2id]
            is_ood = len(gold_idx) == 0   # gold가 paper 338 밖 → OOD (paper 예측 불가)
            text = "Conversation:\n" + "\n".join(item["dialogue_history"]) + \
                   "\n\nBased on the conversation, what diseases does the patient have?"
            enc = tok(text, truncation=True, max_length=2048, return_tensors="pt").to(model.device)
            logits = model(**enc).logits[0].float().cpu().numpy()
            ranked = np.argsort(-logits)
            for k in (1, 2, 3, 4):
                if is_ood:
                    # paper는 OOD 라벨이 없어 구조적으로 0 (gold가 label space 밖)
                    r[k].append(0.0); rood[k].append(0.0)
                else:
                    v = recall_at_k(gold_idx, ranked, k)
                    r[k].append(v); rid[k].append(v)
    out = {"tag": tag, "n": len(r[4]), "n_ID": len(rid[4]), "n_OOD": len(rood[4])}
    for k in (1, 2, 3, 4):
        out[f"all@{k}"] = round(float(np.mean(r[k])), 4) if r[k] else None
        out[f"ID@{k}"] = round(float(np.mean(rid[k])), 4) if rid[k] else None
        out[f"OOD@{k}"] = round(float(np.mean(rood[k])), 4) if rood[k] else None
    return out


results = {}
for tag, path in [("valid", f"{P.TRAIN_DATA_DIR}/data_exp5/valid_combined.json"),
                  ("test",  f"{P.TRAIN_DATA_DIR}/data_exp5/test_combined.json")]:
    res = evaluate(path, tag)
    results[tag] = res
    print(f"\n===== paper HG on {tag} (n={res['n']}, ID {res['n_ID']} / OOD {res['n_OOD']}) =====", flush=True)
    for sub in ("all", "ID", "OOD"):
        print(f"  {sub:3s}: @1={res[f'{sub}@1']}  @2={res[f'{sub}@2']}  @3={res[f'{sub}@3']}  @4={res[f'{sub}@4']}")

json.dump(results, open(f"{P.RESULTS_DIR}/results_paper_hg_on_exp34.json", "w"),
          ensure_ascii=False, indent=2)
print(f"\nSaved → results_paper_hg_on_exp34.json", flush=True)
