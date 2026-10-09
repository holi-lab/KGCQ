"""Run trained OOG HG adapters on the HG validation set and save valid_predictions.json next to each adapter.

train_hg_exp34.py stores only test_predictions.json; make_hg_csv_3to1.py needs the validation top-4 as well
(IG/OOG split of the validation rows). exp5 uses data_exp5/valid_combined.json (OOG gold = 'Other', 339 labels),
exp6 uses data_exp6/valid_combined.json (OOG gold = real disease, 528 labels).

Usage: python eval_hg_valid.py [gpu] [--exp exp5,exp6] [--ratios 0,5,...,35] [--force]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import sys, os, json, argparse

ap = argparse.ArgumentParser()
ap.add_argument('gpu', nargs='?', default='0')
ap.add_argument('--exp', default='exp6,exp5')
ap.add_argument('--ratios', default='0,5,10,15,20,25,30,35')
ap.add_argument('--force', action='store_true', help='recompute even if valid_predictions.json exists')
args = ap.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
import numpy as np, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

BASE = P.BASE_MODEL


def disease_list(csv):
    df = pd.read_csv(csv)
    return [r['name'] for _, r in df.iterrows() if r['label'] == 'Disease']


paper_d = disease_list(P.KG['original']['nodes'])
aug_d = disease_list(P.KG['augmented']['nodes'])
tok = AutoTokenizer.from_pretrained(BASE, padding_side="right")


def run_model(adapter_dir, disease_names, valid, out_path):
    id2label = {i: n for i, n in enumerate(disease_names)}
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto",
        num_labels=len(disease_names), id2label=id2label,
        label2id={n: i for i, n in enumerate(disease_names)},
        problem_type="multi_label_classification")
    base.resize_token_embeddings(len(tok))
    base.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    results = []
    with torch.no_grad():
        for item in valid:
            dh = "\n".join(item['dialogue_history'])
            text = f"Conversation:\n{dh}\n\nBased on the conversation, what diseases does the patient have?"
            enc = tok(text, truncation=True, max_length=2048, return_tensors="pt").to(model.device)
            logits = model(**enc).logits[0].float().cpu().numpy()
            top4 = np.argsort(-logits)[:4]
            results.append({"patient_id": item['patient_id'],
                            "true_labels": list(set(item['ground_truth'])),
                            "top4": [id2label[int(i)] for i in top4]})
    json.dump(results, open(out_path, 'w'), ensure_ascii=False)
    del model, base
    torch.cuda.empty_cache()
    return len(results)


exps = [e.strip() for e in args.exp.split(',') if e.strip()]
ratios = [int(r) for r in args.ratios.split(',') if r.strip()]
for exp, dn, vdir in [("exp6", aug_d, "data_exp6"), ("exp5", paper_d + ['Other'], "data_exp5")]:
    if exp not in exps:
        continue
    valid = json.load(open(f"{P.TRAIN_DATA_DIR}/{vdir}/valid_combined.json"))
    print(f"{exp}: validation {len(valid)} rows ({vdir}), {len(dn)} labels", flush=True)
    for R in ratios:
        mdir = f"{P.MODEL_DIR}/{exp}_hg_r{R:02d}_Qwen2.5-7B"
        if not os.path.exists(f"{mdir}/adapter_model.safetensors"):
            continue
        out = f"{mdir}/valid_predictions.json"
        if os.path.exists(out) and not args.force:
            print(f"  {exp} r{R}: skip (exists)", flush=True)
            continue
        n = run_model(mdir, dn, valid, out)
        print(f"  {exp} r{R}: valid {n} -> {out}", flush=True)
print("=== DONE valid eval ===", flush=True)
