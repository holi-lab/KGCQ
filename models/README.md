# Model checkpoints (not tracked by git)

Final checkpoints of the paper, copied from the original training runs (training checkpoints `checkpoint-*` omitted).
Base models (`Qwen/Qwen2.5-7B-Instruct`, `sentence-transformers/all-MiniLM-L6-v2`) are loaded from the Hugging Face Hub.

| Path | What | Used by |
|---|---|---|
| `hg_qwen2.5-7b_clf_head/` | Hypothesis Generator: Qwen2.5-7B-Instruct + 338-way classification head, LoRA merged (full weights, ~15 GB) | `scripts/3_infer/run_dialogue.py`, `scripts/4_eval/eval_hg_standalone.py`, `scripts/oog/2_eval/eval_paper_hg.py` |
| `hv_qwen2.5-7b_sft_lora/` | Hypothesis Verifier: LoRA adapter (SFT on synthetic dialogues) | `run_dialogue.py --hv_backend local_finetuned` (Table 3 KGCQ, Table 4) |
| `oog/exp5_hg_r30_Qwen2.5-7B/` | "Other"-node HG adapter (338 + Other labels, training OOG ratio 30%) | `scripts/oog/2_eval/inference_exp34.py --exp exp5 --hg_ratio 30` |
| `oog/exp5_hv_r25_symcentric_Qwen2.5-7B/` | "Other"-node HV adapter (ratio 25%, symptom-centric subgraphs) | same, `--hv_ratio 25` (Table 5 / 8 "Other" node) |
| `oog/exp6_hg_r35_Qwen2.5-7B/` | KG-augmentation HG adapter (528 labels, ratio 35%) | `inference_exp34.py --exp exp6 --hg_ratio 35 --kg augmented` |
| `oog/exp6_hv_r30_symcentric_v3a_Qwen2.5-7B/` | KG-augmentation HV adapter (ratio 30%, subgraphs from `data/kg/augmented` and the HG's own predictions) | same, `--hv_ratio 30 --tag_suffix _v3a` (Table 5 / 8 KG augmentation) |

Not preserved: the generative-HG adapter (Fig. 3 "Qwen2.5 (Gen., FT)", retrainable with `scripts/2_train/train_hg_generative.py`),
the SapBERT retriever weights (`scripts/baselines/retriever/train_retriever.py`), and the Chain-of-Diagnosis assets.
Sources: `MedQA/model/1220_clf_head_Qwen2.5-7B-Instruct`, `MedQA/model/1220_sft_Qwen2.5-7B-Instruct_reverse`,
`MedQA/rebuttal_v2/model/{exp5_hg_r30,exp5_hv_r25_symcentric,exp6_hg_r35,exp6_hv_r30_symcentric_v3a}_Qwen2.5-7B`.
