"""exp3 / exp4 HV(SFT) 학습 (train_hv.py 의 exp3/exp4 변형, paper 구조 보존).

차이 (train_hv.py[exp1] 대비):
  데이터 = data_exp{3,4}_hv/train_r{R}_hv.json (exp3=OOD Other 진단 / exp4=OOD real 진단)
  출력 = model/{exp}_hv_r{R}_Qwen2.5-7B
모델/LoRA/SFTConfig/truncation/저장(adapter)은 train_hv.py 와 동일(paper 보존).
HV 자체는 생성형이라 라벨셋 구분 없음 — 대화(진단 타깃)만 exp3/exp4 다름.

사용:  python train_hv_exp34.py <exp> <ratio> <gpu>     # 예: python train_hv_exp34.py exp4 10 0
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import sys, os, random

EXP = sys.argv[1] if len(sys.argv) > 1 else "exp3"
RATIO = int(sys.argv[2]) if len(sys.argv) > 2 else 0
GPU = sys.argv[3] if len(sys.argv) > 3 else "0"
VARIANT = sys.argv[4] if len(sys.argv) > 4 else ""   # "symcentric" → train_r{R}_hv_symcentric.json + 출력 _symcentric
SUFFIX = "_symcentric" if VARIANT == "symcentric" else ""
assert EXP in ("exp3", "exp4", "exp5", "exp6"), EXP   # exp5=paper(exp3 데이터), exp6=aug(exp4 데이터)
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub"))
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

import torch
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

RV2 = P.RV2
# HV_DATA_DIR: 데이터 디렉토리 오버라이드 (예: v3 subgraph 변형). HV_TAG: 출력 모델명 접미사.
DATA = os.environ.get('HV_DATA_DIR', f'{P.TRAIN_DATA_DIR}/data_{EXP}_hv')
HV_TAG = os.environ.get('HV_TAG', '')
PROMPT_PATH = f'{P.PROMPT_DIR}/hv_doctor.txt'
MODEL_ID = f'{P.BASE_MODEL}'   # 로컬 base (이식용)


def load_json(p):
    import json
    return json.load(open(p))


def file_to_string(p):
    with open(p) as f:
        return f.read()


# prepare_conversation_dataset — paper sft_train.py 그대로 (train_hv.py 와 동일)
def prepare_conversation_dataset(data, question_generator_prompt):
    all_samples = []
    for pid, patient_data in data.items():
        if 'subgraph' not in patient_data:
            continue
        subgraph_text = "\n".join(patient_data['subgraph'])
        for symptom, symptom_data in patient_data.items():
            if symptom in ['ground_truth', 'subgraph', '_meta']:
                continue
            if symptom_data == {} or 'full_process' not in symptom_data:
                continue
            dialogue = symptom_data['full_process']
            if symptom_data['recall']['4'] >= 0.5:
                nums = [i for i in range(1, len(dialogue), 2)]
                nums = random.sample(nums, len(dialogue) // 5)
                nums.sort()
                nums.append(len(dialogue) - 1)
                for i in nums:
                    if dialogue[i]["role"] != "assistant":
                        continue
                    check = False
                    dialogue_history = dialogue[:i]
                    dialogue_history_str = []
                    for turn in dialogue_history:
                        if turn["role"] == "assistant":
                            try:
                                question = turn['full_response'].split("<question>")[1].split("</question>")[0]
                            except Exception:
                                check = True
                            dialogue_history_str.append(f"Doctor: {question}")
                        elif turn["role"] == "user":
                            dialogue_history_str.append(f"Patient: {turn['content']}")
                    if check:
                        continue
                    prompt_text = question_generator_prompt.format(
                        dialogue_text="\n".join(dialogue_history_str),
                        subgraph_text=subgraph_text,
                    )
                    all_samples.append({
                        "prompt": prompt_text,
                        "completion": dialogue[i]['full_response'],
                    })
    random.shuffle(all_samples)
    return all_samples


def load_model_for_train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", use_cache=False)
    model.gradient_checkpointing_enable()
    model.config.pad_token_id = tokenizer.pad_token_id
    lora_config = LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'])
    model = get_peft_model(model, lora_config)
    model.train()
    return tokenizer, model


def main():
    question_generator_prompt = file_to_string(PROMPT_PATH)
    raw = load_json(f"{DATA}/train_r{RATIO:02d}_hv{SUFFIX}.json")
    samples = prepare_conversation_dataset(raw, question_generator_prompt)
    print(f"[{EXP} r{RATIO}] dialogues {len(raw)} → SFT 페어 {len(samples)}  GPU={GPU}")
    train_dataset = Dataset.from_list(samples)

    tokenizer, model = load_model_for_train()

    def tokenize_and_mask(examples):
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for prompt, completion in zip(examples['prompt'], examples['completion']):
            fp = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            fc = f"{completion}<|im_end|>\n"
            pids = tokenizer.encode(fp, add_special_tokens=False)
            cids = tokenizer.encode(fc, add_special_tokens=False)
            ids = pids + cids
            out["input_ids"].append(ids)
            out["attention_mask"].append([1] * len(ids))
            out["labels"].append([-100] * len(pids) + cids)
        return out

    train_dataset = train_dataset.map(tokenize_and_mask, batched=True, num_proc=8,
                                      remove_columns=["prompt", "completion"])
    max_seq_len = max(len(x) for x in train_dataset['input_ids'])
    print(f"max seq len: {max_seq_len}")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8)

    run_name = f"{EXP}_hv_r{RATIO:02d}{SUFFIX}{HV_TAG}_Qwen2.5-7B"
    output_dir = f"{P.MODEL_DIR}/{run_name}"
    sft_config = SFTConfig(
        output_dir=output_dir, max_length=max_seq_len, packing=False, do_train=True,
        per_device_train_batch_size=2, gradient_accumulation_steps=2, num_train_epochs=2,
        learning_rate=1e-5, logging_steps=10, optim="paged_adamw_32bit",
        warmup_ratio=0.05, lr_scheduler_type="cosine", weight_decay=0.1, bf16=True,
        group_by_length=True, remove_unused_columns=False, save_strategy="epoch",
        gradient_checkpointing=True, prediction_loss_only=True,
        report_to=os.environ.get("DGEN_REPORT_TO", "none"),
        run_name=run_name, save_total_limit=2, seed=42)
    trainer = SFTTrainer(model=model, train_dataset=train_dataset, args=sft_config,
                         data_collator=data_collator)
    torch.cuda.empty_cache()
    model.config.use_cache = False

    last_checkpoint = None
    if os.path.isdir(output_dir):
        cks = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
        if cks:
            last_checkpoint = os.path.join(output_dir, max(cks, key=lambda x: int(x.split('-')[1])))
            print(f"Resuming from {last_checkpoint}")

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model()
    trainer.save_state()
    trainer.save_metrics("train", train_result.metrics)
    tokenizer.save_pretrained(output_dir)
    print(f"[{EXP} r{RATIO}] HV SFT done → {output_dir}")


if __name__ == "__main__":
    main()
