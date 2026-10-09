#!/usr/bin/env python
"""Generative Hypothesis Generator baseline (Appendix C.1, Fig. 3 "Qwen2.5 (Gen., FT)").

Fine-tunes Qwen2.5-7B-Instruct (LoRA) to output the comma-separated disease names given the dialogue history and
the full list of KG disease names (prompts/hg_generative.txt-style "full" prompt: select 1-4 diseases).
Uses the same HG soft-label rows as the classification head; targets = ground_truth.

Hyper-parameters (original run): lr 1e-5, batch 4 x accum 4, 3 epochs, cosine/warmup 0.05, wd 0.1, LoRA r16,
early stopping on eval loss (patience 3), seed 42.

Example
  python scripts/3_train/train_hg_generative.py --output_dir models/hg_generative_qwen2.5-7b_lora --gpu 0,1
"""
import argparse
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from kgcq import paths  # noqa: E402
from kgcq.utils import load_json  # noqa: E402

FULL_PROMPT = """Act as a strictly logical medical expert.
Your task is to identify the most likely differential diagnoses from a provided list of candidate diseases, based on the patient's dialogue history.

### Input Data
1. ** Disease Candidates (Total {disease_names_len}):**
{disease_names}

2. **Dialogue History:**
{dialogue_history}

### Instructions
Step 1: Analyze the **Dialogue History**. Extract all reported symptoms (both positive and negative).
Step 2: Based **ONLY** on the extracted symptoms, reason about potential conditions.
Step 3: Scan the **Allowed Disease Candidates** list. Select exactly 1 to 4 diseases that match your reasoning.
    - **CRITICAL:** If there are more than one diseases, rank the selected diseases in descending order of probability (most likely first).
    - **CRITICAL:** You must select diseases **EXACTLY** as they appear in the list. Do not modify names or invent new ones.
    - If the symptoms are vague (e.g., just "headache"), select the most common/broad diseases from the list that fit.
    - If specific symptoms are present, prioritize diseases that explain those specific features.
 
### Output Format
Answer in the format ‘disease1, disease2, ..' separated by commas, **ordered from most likely to least likely**, without any additional explanation.
"""


def prepare(data, disease_names):
    samples = []
    for item in data:
        inp = {"dialogue_history": "\n".join(item['dialogue_history']), "disease_names": "; ".join(disease_names),
               "disease_names_len": len(disease_names)}
        samples.append({"prompt": FULL_PROMPT.format(**inp), "completion": ', '.join(list(set(item['ground_truth'])))})
    random.shuffle(samples)
    return samples


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default=str(paths.SOFTLABEL_DIR / "train.json"))
    p.add_argument("--valid", default=str(paths.SOFTLABEL_DIR / "valid.json"))
    p.add_argument("--kg_nodes", default=str(paths.KG_NODES))
    p.add_argument("--model_id", default=paths.BASE_LLM)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--report_to", default="none")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accumulation_steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)

    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback, DataCollatorForSeq2Seq
    from peft import LoraConfig, get_peft_model
    from kgcq.models import disease_names_from_nodes
    torch.utils.checkpoint.use_reentrant = False

    disease_names = disease_names_from_nodes(args.kg_nodes)
    train_ds = Dataset.from_list(prepare(load_json(args.train), disease_names))
    eval_ds = Dataset.from_list(prepare(load_json(args.valid), disease_names))

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.config.pad_token_id = tokenizer.pad_token_id
    model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                                             bias="none", task_type="CAUSAL_LM",
                                             target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']))
    model.train()

    def tokenize_and_mask(examples):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for prompt, completion in zip(examples['prompt'], examples['completion']):
            fp = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            fc = f"{completion}<|im_end|>\n"
            pids = tokenizer.encode(fp, add_special_tokens=False)
            cids = tokenizer.encode(fc, add_special_tokens=False)
            out["input_ids"].append(pids + cids)
            out["attention_mask"].append([1] * (len(pids) + len(cids)))
            out["labels"].append([-100] * len(pids) + cids)
        return out

    train_ds = train_ds.map(tokenize_and_mask, batched=True, num_proc=8, remove_columns=["prompt", "completion"])
    eval_ds = eval_ds.map(tokenize_and_mask, batched=True, num_proc=8, remove_columns=["prompt", "completion"])
    max_len = max(len(x) for x in train_ds['input_ids'] + eval_ds['input_ids'])

    cfg = SFTConfig(output_dir=args.output_dir, max_length=max_len, packing=False, do_train=True, do_eval=True,
                    per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation_steps,
                    per_device_eval_batch_size=args.batch_size, eval_accumulation_steps=args.accumulation_steps,
                    num_train_epochs=args.epochs, learning_rate=args.learning_rate, logging_steps=10,
                    optim="paged_adamw_32bit", warmup_ratio=0.05, lr_scheduler_type="cosine", weight_decay=0.1,
                    bf16=True, group_by_length=True, remove_unused_columns=False, eval_strategy="epoch",
                    save_strategy="epoch", gradient_checkpointing=True, overwrite_output_dir=True,
                    prediction_loss_only=True, load_best_model_at_end=True, ddp_find_unused_parameters=False,
                    logging_dir=os.path.join(args.output_dir, "runs"), report_to=args.report_to,
                    metric_for_best_model="eval_loss", greater_is_better=False, save_total_limit=2, seed=args.seed)
    trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=eval_ds, args=cfg,
                         data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8),
                         callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience, early_stopping_threshold=0.001)])
    torch.cuda.empty_cache()
    model.config.use_cache = False
    train_result = trainer.train()
    print("Final eval:", trainer.evaluate(eval_dataset=eval_ds))
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_model()
    trainer.save_state()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
