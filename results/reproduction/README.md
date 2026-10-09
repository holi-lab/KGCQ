Reproduction check from a clean checkout (2026-10-09): `uv sync` environment, SLURM, one A100-80GB per job.
HG trained with scripts/3_train/train_hg.py (early-stopped at epoch 6, 1h32m), HV with scripts/3_train/train_hv.py
(2 epochs, 6h06m), then scripts/4_infer/run_dialogue.py --mode kgcq on the 275 evaluation profiles (288 dialogues,
2h09m, gpt-4o-mini patient simulator) and scripts/5_eval/summarize_run.py.
  kgcq_n2_tau0.005/metrics*.json  end-to-end metrics of the retrained system
  hg_test/test_metrics.json       standalone HG Recall@k on the 1,075 HG test rows
  hv_train_results.json           HV SFT training summary
