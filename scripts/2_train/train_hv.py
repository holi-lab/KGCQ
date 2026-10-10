#!/usr/bin/env python
"""Train the Hypothesis Verifier: LoRA SFT of Qwen2.5-7B-Instruct on synthetic dialogues (lr 1e-5, 2 epochs,
prompt tokens masked). Targets: a random 20% of the doctor turns plus the final turn of dialogues with
Recall@4 >= 0.5.

  python scripts/2_train/train_hv.py --train data/dialogues/hv_train_valid.json --output_dir models/hv_qwen2.5-7b_sft_lora --gpu 0
"""
import argparse
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from kgcq import paths  # noqa: E402
from kgcq.utils import load_json, file_to_string  # noqa: E402


def prepare_conversation_dataset(data, doctor_prompt, sample_ratio=5, min_recall4=0.5, verbose=True):
    """dialog.json -> [{"prompt", "completion"}] exactly as in the original sft_train.py."""
    all_samples, skipped, unique_d = [], 0, set()
    for pid, patient_data in data.items():
        unique_d.update(patient_data['ground_truth'])
        subgraph_text = "\n".join(patient_data['subgraph'])
        for symptom, symptom_data in patient_data.items():
            if symptom in ['ground_truth', 'subgraph', '_meta'] or symptom_data == {} or 'full_process' not in symptom_data:
                continue
            dialogue = symptom_data['full_process']
            if symptom_data['recall']['4'] < min_recall4:
                skipped += 1
                continue
            nums = [i for i in range(1, len(dialogue), 2)]
            nums = random.sample(nums, len(dialogue) // sample_ratio)
            nums.sort()
            nums.append(len(dialogue) - 1)
            for i in nums:
                if dialogue[i]["role"] != "assistant":
                    continue
                check = False
                history = []
                for turn in dialogue[:i]:
                    if turn["role"] == "assistant":
                        try:
                            question = turn['full_response'].split("<question>")[1].split("</question>")[0]
                        except Exception:
                            check = True
                            question = ""
                        history.append(f"Doctor: {question}")
                    elif turn["role"] == "user":
                        history.append(f"Patient: {turn['content']}")
                if check:
                    continue
                prompt_text = doctor_prompt.format(dialogue_text="\n".join(history), subgraph_text=subgraph_text)
                all_samples.append({"prompt": prompt_text, "completion": dialogue[i]['full_response']})
    if verbose:
        print(f"dialogues skipped (recall@4 < {min_recall4}): {skipped}; unique diseases: {len(unique_d)}; pairs: {len(all_samples)}")
    random.shuffle(all_samples)
    return all_samples


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default=str(paths.DIALOGUE_DIR / "hv_train_valid.json"))
    p.add_argument("--doctor_prompt", default=str(paths.HV_DOCTOR_PROMPT))
    p.add_argument("--model_id", default=paths.BASE_LLM)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu", default="0")
    p.add_argument("--report_to", default="none")
    p.add_argument("--run_name", default=None)
    p.add_argument("--use_quant", action="store_true")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--accumulation_steps", type=int, default=2)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--optim", default="paged_adamw_32bit")
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--save_strategy", default="epoch")
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_pairs", type=int, default=0, help="debug: train on only the first N prompt/completion pairs (0 = all)")
    return p


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)

    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorForSeq2Seq
    from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
    torch.utils.checkpoint.use_reentrant = False

    samples = prepare_conversation_dataset(load_json(args.train), file_to_string(args.doctor_prompt))
    if args.max_pairs:
        samples = samples[:args.max_pairs]
    train_dataset = Dataset.from_list(samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_quant_storage=torch.bfloat16,
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=False) if args.use_quant else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, attn_implementation=("sdpa" if args.use_quant else None),
        quantization_config=bnb, device_map="auto", use_cache=(False if args.use_quant else None))
    model.gradient_checkpointing_enable()
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.use_quant:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']))
    model.train()

    is_llama = 'llama' in args.model_id.lower()

    def tokenize_and_mask(examples):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for idx, (prompt, completion) in enumerate(zip(examples['prompt'], examples['completion'])):
            if is_llama:
                fp = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                fc = f"{completion}<|eot_id|>"
            else:  # Qwen / ChatML
                fp = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                fc = f"{completion}<|im_end|>\n"
            pids = tokenizer.encode(fp, add_special_tokens=False)
            cids = tokenizer.encode(fc, add_special_tokens=False)
            out["input_ids"].append(pids + cids)
            out["attention_mask"].append([1] * (len(pids) + len(cids)))
            out["labels"].append([-100] * len(pids) + cids)
        return out

    train_dataset = train_dataset.map(tokenize_and_mask, batched=True, num_proc=8, remove_columns=["prompt", "completion"])
    max_seq_len = max(len(x) for x in train_dataset['input_ids'])
    print("max seq len:", max_seq_len)

    run_name = args.run_name or os.path.basename(os.path.normpath(args.output_dir))
    sft_config = SFTConfig(
        output_dir=args.output_dir, max_length=max_seq_len, packing=False, do_train=True,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.accumulation_steps,
        per_device_eval_batch_size=args.batch_size, num_train_epochs=args.epochs, learning_rate=args.learning_rate,
        logging_steps=args.logging_steps, optim=args.optim, warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type, weight_decay=args.weight_decay, bf16=True, group_by_length=True,
        remove_unused_columns=False, save_strategy=args.save_strategy, gradient_checkpointing=True,
        overwrite_output_dir=True, prediction_loss_only=True, ddp_find_unused_parameters=False,
        logging_dir=os.path.join(args.output_dir, "runs"), report_to=args.report_to, run_name=run_name,
        save_total_limit=2, seed=args.seed)
    trainer = SFTTrainer(model=model, train_dataset=train_dataset, args=sft_config,
                         data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8))
    torch.cuda.empty_cache()
    model.config.use_cache = False

    last_checkpoint = None
    if not args.no_resume and os.path.isdir(args.output_dir):
        cks = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if cks:
            last_checkpoint = os.path.join(args.output_dir, max(cks, key=lambda x: int(x.split('-')[1])))
            print(f"Resuming from {last_checkpoint}")
    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_model()
    trainer.save_state()
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved LoRA adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
