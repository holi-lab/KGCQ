"""exp3 / exp4 HG 분류헤드 학습 (train_hg.py 의 exp3/exp4 변형, paper 구조 보존).

차이 (train_hg.py[exp1] 대비):
  exp3: 라벨 = paper 338 + 'Other' = 339,  data_exp3,  평가 OOD = gold 'Other'
  exp4: 라벨 = augmented_v3 528 (Other 없음),  data_exp4,  평가 OOD = gold ∉ paper(=rescued)
  데이터 = ratio sweep (train_r{R}.json) + valid_combined + test_combined
  출력 = model/{exp}_hg_r{R}_Qwen2.5-7B,  results/{exp}_r{R}.json
모델/LoRA/TrainingArguments/compute_metrics/adapter저장은 train_hg.py 와 동일(paper 보존).

사용:  python train_hg_exp34.py <exp> <ratio> <gpu>     # 예: python train_hg_exp34.py exp4 10 0
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import sys, os, random
import pandas as pd
from tqdm import tqdm
import numpy as np

EXP = sys.argv[1] if len(sys.argv) > 1 else "exp3"
RATIO = int(sys.argv[2]) if len(sys.argv) > 2 else 0
GPU = sys.argv[3] if len(sys.argv) > 3 else "0"
assert EXP in ("exp3", "exp4", "exp5", "exp6"), EXP
# exp5 = paper KG (exp3 로직, 신규 sweep) / exp6 = aug KG (exp4 로직, 신규 sweep)
_USE_PAPER = EXP in ("exp3", "exp5")

os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import torch
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
from peft import LoraConfig, get_peft_model, TaskType

RV2 = P.RV2
DATA = f'{P.TRAIN_DATA_DIR}/data_{EXP}'


def load_json(p):
    import json
    return json.load(open(p))


def save_to_json(obj, p):
    import json
    json.dump(obj, open(p, 'w'), ensure_ascii=False)


def disease_list(csv):
    df = pd.read_csv(csv)
    return [r['name'] for _, r in df.iterrows() if r['label'] == 'Disease']


# ========================================================
# 1. 라벨 — exp3: paper 338 + Other / exp4: aug 528
# ========================================================
if _USE_PAPER:   # exp3, exp5
    disease_names = disease_list(f"{P.KG_DIR}/paper/nodes.csv") + ['Other']     # 339
else:            # exp4, exp6
    disease_names = disease_list(f"{P.KG_DIR}/augmented_v3/nodes.csv")          # 528 (Other 없음)
id2label = {i: n for i, n in enumerate(disease_names)}
label2id = {n: i for i, n in enumerate(disease_names)}
num_labels = len(disease_names)
# 평가 OOD 판정용 (exp4 = paper 밖 질병 = rescued OOD)
PAPER_SET = set(str(n).strip().lower() for n in disease_list(f"{P.KG_DIR}/paper/nodes.csv"))
NONPAPER_IDX = set(i for i, n in id2label.items() if str(n).strip().lower() not in PAPER_SET and n != 'Other')
print(f"[{EXP}] 라벨 {num_labels} | ratio=r{RATIO} GPU={GPU}")

# ========================================================
# 2. softlabel → multilabel dataset
# ========================================================
train_raw = load_json(f"{DATA}/train_r{RATIO:02d}.json")
valid_raw = load_json(f"{DATA}/valid_combined.json")
test_raw = load_json(f"{DATA}/test_combined.json")


def make_examples(data):
    unique_d, all_samples = set(), []
    for item in data:
        dialogue_history = "\n".join(item['dialogue_history'])
        label = list(set(item['ground_truth']))
        unique_d.update(label)
        all_samples.append({
            "patient_id": item['patient_id'],
            "prompt": f"Conversation:\n{dialogue_history}\n\nBased on the conversation, what diseases does the patient have?",
            "labels": label})
    random.shuffle(all_samples)
    miss = [d for d in unique_d if d not in label2id]
    print(f"  samples={len(all_samples)}, unique={len(unique_d)}, 미매칭={len(miss)}" + (f" 예:{miss[:3]}" if miss else ""))
    return all_samples


train_examples = make_examples(train_raw)
val_examples = make_examples(valid_raw)
test_examples = make_examples(test_raw)
print(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")


def make_dataset(data, label2id_map, shuffle=True):
    prompts, labels = [], []
    for sample in tqdm(data):
        label_vector = [0] * num_labels
        for ds in sample['labels']:
            if ds in label2id_map:
                label_vector[label2id_map[ds]] = 1
        prompts.append(sample['prompt'])
        labels.append([float(x) for x in label_vector])
    dataset = Dataset.from_dict({"text": prompts, "labels": labels})
    return dataset.shuffle(seed=42) if shuffle else dataset


train_dataset = make_dataset(train_examples, label2id)
dev_dataset = make_dataset(val_examples, label2id)

model_name = f'{P.BASE_MODEL}'   # 로컬 base (이식용)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")


def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)


tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True).remove_columns(["text"])


# ========================================================
# 6. Metrics (recall@4 + AUROC) — paper 그대로
# ========================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    try:
        auroc_macro = roc_auc_score(labels, probs, average="macro")
        auroc_micro = roc_auc_score(labels, probs, average="micro")
    except ValueError:
        auroc_macro = auroc_micro = float("nan")
    total_recall, valid_samples = 0.0, 0
    for i in range(len(labels)):
        true_indices = np.where(labels[i] == 1)[0]
        if len(true_indices) == 0:
            continue
        top4 = np.argsort(-logits[i])[:4]
        total_recall += len(set(top4) & set(true_indices)) / len(true_indices)
        valid_samples += 1
    recall_at_4 = total_recall / valid_samples if valid_samples > 0 else 0.0
    return {"recall_at_4": recall_at_4, "auroc_macro": auroc_macro, "auroc_micro": auroc_micro}


# ========================================================
# 7~8. Model + LoRA — paper 그대로
# ========================================================
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, device_map="auto",
    num_labels=num_labels, id2label=id2label, label2id=label2id,
    problem_type="multi_label_classification")
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_CLS,
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'])
model = get_peft_model(model, lora_config)
model.train()

# ========================================================
# 9. TrainingArguments — paper 그대로
# ========================================================
run_name = f"{EXP}_hg_r{RATIO:02d}_Qwen2.5-7B"
output_dir = f"{P.MODEL_DIR}/{run_name}"
training_args = TrainingArguments(
    output_dir=output_dir, learning_rate=1e-5,
    per_device_train_batch_size=2, per_device_eval_batch_size=2,
    gradient_accumulation_steps=4, num_train_epochs=10, weight_decay=0.01,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    max_grad_norm=1.0, metric_for_best_model="recall_at_4", greater_is_better=True,
    logging_dir=os.path.join(output_dir, "runs"),
    report_to=os.environ.get("DGEN_REPORT_TO", "none"),
    run_name=run_name, save_total_limit=2, seed=42, bf16=True)

trainer = Trainer(
    model=model, args=training_args,
    train_dataset=tokenized_train_dataset, eval_dataset=tokenized_dev_dataset,
    processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])

last_checkpoint = None
if os.path.isdir(output_dir):
    cks = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if cks:
        last_checkpoint = os.path.join(output_dir, max(cks, key=lambda x: int(x.split('-')[1])))
        print(f"Resuming from {last_checkpoint}")

train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

# ========================================================
# 11. adapter only 저장 (디스크 절약) — train_hg.py 와 동일
# ========================================================
print("Saving LoRA adapter (not merged)...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
save_to_json({"disease_names": disease_names, "num_labels": num_labels, "exp": EXP},
             os.path.join(output_dir, "label_config.json"))
trainer.save_state()
trainer.save_metrics("train", train_result.metrics)

valid_metrics = trainer.evaluate()
valid_recall_at_4 = float(valid_metrics.get("eval_recall_at_4", float("nan")))
print(f"\n[{EXP} r{RATIO}] VALID recall@4 = {valid_recall_at_4:.4f}")

# ========================================================
# 12. Test eval (recall@1~4 + ID/OOD 분리)
# ========================================================
test_dataset = make_dataset(test_examples, label2id, shuffle=False)
tokenized_test = test_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
test_pred = trainer.predict(tokenized_test)
logits = test_pred.predictions
true_vecs = test_pred.label_ids
probs = 1 / (1 + np.exp(-logits))


def safe_roc_auc_macro(y_true, y_score):
    aucs = []
    for c in range(y_true.shape[1]):
        if len(np.unique(y_true[:, c])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y_true[:, c], y_score[:, c]))
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else float("nan")


try:
    auroc_micro = roc_auc_score(true_vecs, probs, average="micro")
except Exception:
    auroc_micro = float("nan")
auroc_macro = safe_roc_auc_macro(true_vecs, probs)


def recall_at_k(true_vec, topk_ids):
    ti = np.where(true_vec == 1)[0]
    if len(ti) == 0:
        return None
    return len(set(ti) & set(topk_ids)) / len(ti)


# OOD 판정: exp3 = gold 'Other' / exp4 = gold 에 paper 밖(rescued) 질병 포함
other_idx = label2id.get('Other')


def is_ood(tv):
    if _USE_PAPER:   # exp3, exp5: OOD = gold 'Other'
        return other_idx is not None and tv[other_idx] == 1
    return any(tv[i] == 1 for i in NONPAPER_IDX)   # exp4, exp6: rescued(nonpaper)


ranked = np.argsort(-logits, axis=1)
r1 = []; r2 = []; r3 = []; r4 = []
r4_id = []; r4_ood = []
results = []
for i, sample in enumerate(test_examples):
    tv = true_vecs[i]
    for store, k in [(r1, 1), (r2, 2), (r3, 3), (r4, 4)]:
        v = recall_at_k(tv, ranked[i, :k])
        if v is not None:
            store.append(v)
    v4 = recall_at_k(tv, ranked[i, :4])
    if v4 is not None:
        (r4_ood if is_ood(tv) else r4_id).append(v4)
    results.append({"patient_id": sample["patient_id"],
                    "true_labels": [id2label[x] for x in np.where(tv == 1)[0]],
                    "top4": [id2label[x] for x in ranked[i, :4]]})

label_OOD = "recall_at_4_OOD(Other)" if _USE_PAPER else "recall_at_4_OOD(rescued)"
test_metrics = {
    "exp": EXP, "ratio": RATIO, "num_labels": num_labels,
    "valid_recall_at_4": valid_recall_at_4,
    "auroc_micro": auroc_micro, "auroc_macro": auroc_macro,
    "recall_at_1": float(np.mean(r1)), "recall_at_2": float(np.mean(r2)),
    "recall_at_3": float(np.mean(r3)), "recall_at_4": float(np.mean(r4)),
    "recall_at_4_ID": float(np.mean(r4_id)) if r4_id else None,
    label_OOD: float(np.mean(r4_ood)) if r4_ood else None,
}
print(f"\n===== Metrics ({EXP} r{RATIO}) =====")
for k, v in test_metrics.items():
    print(f"  {k}: {v}")
save_to_json(results, os.path.join(output_dir, "test_predictions.json"))
save_to_json(test_metrics, os.path.join(output_dir, "test_metrics.json"))
res_dir = f"{P.HG_RESULTS_DIR}"
os.makedirs(res_dir, exist_ok=True)
save_to_json(test_metrics, f"{res_dir}/{EXP}_r{RATIO:02d}.json")
print(f"\nSaved → {output_dir}/  +  {res_dir}/{EXP}_r{RATIO:02d}.json")
