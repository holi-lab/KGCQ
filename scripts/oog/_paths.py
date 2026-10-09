"""Shared paths for the out-of-graph (OOG) experiments (paper Sec. 6.6 and Appendix A.2).

All paths are relative to the repository root; override with KGCQ_ROOT if needed.
Layout:
  data/kg/{paper,augmented}                    knowledge graphs (G: 338 diseases / G+: 528, attributes validated on MIMIC notes)
  data/oog/profiles                               OOG evaluation profiles and split jsonl files
  data/oog/preprocessed                           full profiles (exp1_id / exp1_ood / exp2_id)
  data/oog/train/data_exp{5,6}[_hv][_v3a]         ratio-sweep training sets and HG eval sets ({valid,test}_combined.json)
  models/oog                                      HG/HV adapters of the sweeps (+ base models under models/base)
  results/oog                                     metrics, CSV tables, per-case results
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("KGCQ_ROOT", HERE.parent.parent))
RV2 = str(ROOT)                                    # kept for scripts ported from rebuttal_v2
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
KG_DIR = ROOT / "data" / "kg"
OOG_DATA = ROOT / "data" / "oog"
PROFILE_DIR = OOG_DATA / "profiles"
PREPROCESSED_DIR = OOG_DATA / "preprocessed"
DIALOGUE_DIR = OOG_DATA / "dialogues"
OOD_SPLIT_DIR = OOG_DATA / "ood_split"
TRAIN_DATA_DIR = OOG_DATA / "train"
MODEL_DIR = ROOT / "models" / "oog"
_BASE_LOCAL = ROOT / "models" / "base" / "Qwen2.5-7B-Instruct"
_EMB_LOCAL = ROOT / "models" / "base" / "all-MiniLM-L6-v2"
BASE_MODEL = str(_BASE_LOCAL) if _BASE_LOCAL.exists() else "Qwen/Qwen2.5-7B-Instruct"   # local copy or HF id
EMB_MODEL = str(_EMB_LOCAL) if _EMB_LOCAL.exists() else "all-MiniLM-L6-v2"
RESULTS_DIR = ROOT / "results" / "oog"
PIPELINE_RESULTS_DIR = RESULTS_DIR / "pipeline"
HG_RESULTS_DIR = RESULTS_DIR / "hg_sweep"
FINAL_TABLES_DIR = RESULTS_DIR / "final_tables"
PROMPT_DIR = ROOT / "prompts"
ENV_FILE = ROOT / ".env"
KG = {name: {"nodes": str(KG_DIR / name / "nodes.csv"), "edges": str(KG_DIR / name / "edges.csv")}
      for name in ("original", "augmented")}

for d in (TRAIN_DATA_DIR, PIPELINE_RESULTS_DIR, HG_RESULTS_DIR, FINAL_TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)
