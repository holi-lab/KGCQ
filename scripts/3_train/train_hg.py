#!/usr/bin/env python
"""Train the Hypothesis Generator (HG): Qwen2.5-7B-Instruct + linear classification head, multi-label over the
disease nodes of the knowledge graph (paper Sec. 3.5.2; hyper-parameters unchanged from the original run).

Data: HG soft-label rows built by scripts/2_synth/build_hg_softlabel.py
      [{"patient_id", "symptom", "dialogue_history": [...], "ground_truth": [...], ...}]
      Only ``ground_truth`` (gold diseases) is used as the target.
Label order = order of Disease rows in --kg_nodes (+ "Other" with --add_other_label, used by the OOG experiments).

Hyper-parameters (paper): LoRA r=16, alpha=16, dropout=0.05 on q/k/v/o/up/down/gate; lr 1e-5; batch 2 x accum 4;
10 epochs; weight decay 0.01; max_grad_norm 1.0; bf16; eval/save each epoch; early stopping (patience 3) on
validation Recall@4; seed 42; max_length 2048.

Example
  python scripts/3_train/train_hg.py --output_dir models/hg_qwen2.5-7b_clf_head --gpu 0
"""
import argparse
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from kgcq import paths  # noqa: E402
from kgcq.utils import load_json, save_to_json  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default=str(paths.SOFTLABEL_DIR / "train.json"))
    p.add_argument("--valid", default=str(paths.SOFTLABEL_DIR / "valid.json"))
    p.add_argument("--test", default=str(paths.SOFTLABEL_DIR / "test.json"))
    p.add_argument("--kg_nodes", default=str(paths.KG_NODES))
    p.add_argument("--add_other_label", action="store_true", help="append an 'Other' label (out-of-graph abstention)")
    p.add_argument("--model_id", default=paths.BASE_LLM)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--save_adapter_only", action="store_true", help="save the LoRA adapter instead of the merged model")
    p.add_argument("--report_to", default="none", help="e.g. wandb")
    p.add_argument("--run_name", default=None)
    # hyper-parameters (defaults = paper)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_rows", type=int, default=0, help="debug: use only the first N rows of each split (0 = all)")
    return p


def make_examples(data, prompt_fn):
    samples = []
    for item in data:
        samples.append({"patient_id": item['patient_id'], "prompt": prompt_fn(item['dialogue_history']),
                        "labels": list(set(item['ground_truth']))})
    random.shuffle(samples)
    return samples


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    random.seed(args.seed)

    import numpy as np
    import torch
    from datasets import Dataset
    from sklearn.metrics import roc_auc_score
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding,
                              TrainingArguments, Trainer, EarlyStoppingCallback)
    from peft import LoraConfig, get_peft_model, TaskType
    from kgcq.models import disease_names_from_nodes

    disease_names = disease_names_from_nodes(args.kg_nodes)
    if args.add_other_label:
        disease_names = disease_names + ["Other"]
    id2label = {i: n for i, n in enumerate(disease_names)}
    label2id = {n: i for i, n in enumerate(disease_names)}
    num_labels = len(disease_names)
    print(f"labels: {num_labels}")

    def prompt_fn(dialogue_history):
        return f"Conversation:\n{chr(10).join(dialogue_history)}\n\nBased on the conversation, what diseases does the patient have?"

    def take(rows):
        return rows[:args.max_rows] if args.max_rows else rows
    train_examples = make_examples(take(load_json(args.train)), prompt_fn)
    val_examples = make_examples(take(load_json(args.valid)), prompt_fn)
    test_examples = make_examples(take(load_json(args.test)), prompt_fn) if args.test and os.path.exists(args.test) else []
    print(f"Train: {len(train_examples)}, Val: {len(val_examples)}, Test: {len(test_examples)}")

    def make_dataset(data, shuffle=True):
        prompts, labels = [], []
        for s in data:
            vec = [0.0] * num_labels
            for d in s['labels']:
                if d in label2id:
                    vec[label2id[d]] = 1.0
            prompts.append(s['prompt'])
            labels.append(vec)
        ds = Dataset.from_dict({"text": prompts, "labels": labels})
        return ds.shuffle(seed=args.seed) if shuffle else ds

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="right")

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    train_ds = make_dataset(train_examples).map(tokenize, batched=True).remove_columns(["text"])
    dev_ds = make_dataset(val_examples).map(tokenize, batched=True).remove_columns(["text"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        try:
            auroc_macro = roc_auc_score(labels, probs, average="macro")
            auroc_micro = roc_auc_score(labels, probs, average="micro")
        except ValueError:
            auroc_macro = auroc_micro = float("nan")
        total, n = 0.0, 0
        for i in range(len(labels)):
            true_idx = np.where(labels[i] == 1)[0]
            if len(true_idx) == 0:
                continue
            top4 = np.argsort(-logits[i])[:4]
            total += len(set(top4) & set(true_idx)) / len(true_idx)
            n += 1
        return {"recall_at_4": total / n if n else 0.0, "auroc_macro": auroc_macro, "auroc_micro": auroc_micro}

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto", num_labels=num_labels,
        id2label=id2label, label2id=label2id, problem_type="multi_label_classification")
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type=TaskType.SEQ_CLS,
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']))
    model.train()

    run_name = args.run_name or os.path.basename(os.path.normpath(args.output_dir))
    training_args = TrainingArguments(
        output_dir=args.output_dir, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs, weight_decay=args.weight_decay,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True, max_grad_norm=1.0,
        metric_for_best_model="recall_at_4", greater_is_better=True,
        logging_dir=os.path.join(args.output_dir, "runs"), report_to=args.report_to, run_name=run_name,
        save_total_limit=2, seed=args.seed, bf16=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds,
                      processing_class=tokenizer, data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics, callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)])

    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        cks = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if cks:
            last_checkpoint = os.path.join(args.output_dir, max(cks, key=lambda x: int(x.split('-')[1])))
            print(f"Resuming from {last_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    if args.save_adapter_only:
        model.save_pretrained(args.output_dir)
    else:
        print("Merging LoRA weights and saving the full classifier...")
        model.merge_and_unload().save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    save_to_json({"disease_names": disease_names, "num_labels": num_labels}, os.path.join(args.output_dir, "label_config.json"))
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()
    with torch.no_grad():
        torch.cuda.empty_cache()
        print("Final validation:", trainer.evaluate())

    if not test_examples:
        return
    # ---- standalone test evaluation (Recall@1-4 on truncated-dialogue rows; Fig. 3 "Qwen2.5 + Head")
    test_ds = make_dataset(test_examples, shuffle=False).map(tokenize, batched=True).remove_columns(["text"])
    pred = trainer.predict(test_ds)
    logits, true_vecs = pred.predictions, pred.label_ids
    probs = 1 / (1 + np.exp(-logits))

    def safe_macro_auc(y_true, y_score):
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
    ranked = np.argsort(-logits, axis=1)
    recalls = {k: [] for k in (1, 2, 3, 4)}
    rows = []
    for i, s in enumerate(test_examples):
        true_idx = np.where(true_vecs[i] == 1)[0]
        if len(true_idx) == 0:
            continue
        for k in recalls:
            recalls[k].append(len(set(true_idx) & set(ranked[i, :k])) / len(true_idx))
        rows.append({"patient_id": s["patient_id"], "prompt": s["prompt"],
                     "true_labels": [id2label[x] for x in true_idx],
                     "predicted_labels_threshold": [id2label[x] for x in np.where(probs[i] >= 0.001)[0]],
                     "top4_predictions": [id2label[x] for x in ranked[i, :4]]})
    metrics = {"auroc_micro": auroc_micro, "auroc_macro": safe_macro_auc(true_vecs, probs)}
    metrics.update({f"recall_at_{k}": float(np.mean(v)) for k, v in recalls.items()})
    print("Test metrics:", metrics)
    save_to_json(rows, os.path.join(args.output_dir, "test_predictions.json"))
    save_to_json(metrics, os.path.join(args.output_dir, "test_metrics.json"))


if __name__ == "__main__":
    main()
