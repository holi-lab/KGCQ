"""Shared paths for the out-of-graph (OOG) experiments (paper Sec. 6.6 and Appendix A.2).

All paths are relative to the repository root; override with KGCQ_ROOT if needed.
Layout:
  data/kg/{paper,augmented,augmented_v3}          knowledge graphs (338 / 528 LLM-mined / 528 MIMIC-validated)
  data/oog/profiles                               OOG evaluation profiles and split jsonl files
  data/oog/preprocessed                           full profiles (exp1_id / exp1_ood / exp2_id)
  data/oog/train/data_exp{5,6}[_hv][_v3a]         ratio-sweep training sets and HG eval sets ({valid,test}_combined.json)
  data/oog/kg_v3                                  MIMIC-note validation artefacts for the augmented graph
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
KG_V3_DIR = OOG_DATA / "kg_v3"
SPLIT_INPUT_DIR = OOG_DATA / "profile_split_inputs"
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
      for name in ("paper", "augmented", "augmented_v3")}
# MIMIC-IV note sections (restricted; needed only to rebuild the MIMIC-validated graph augmented_v3)
MIMIC_NOTE_SECTION_CSV = os.environ.get("MIMIC_NOTE_SECTION_CSV", str(OOG_DATA / "mimic" / "note_section.csv"))
OOG_PROFILE_POOL_JSONL = str(SPLIT_INPUT_DIR / "profiles_paperKG_OOD_50289.jsonl")

for d in (TRAIN_DATA_DIR, PIPELINE_RESULTS_DIR, HG_RESULTS_DIR, FINAL_TABLES_DIR, KG_V3_DIR):
    d.mkdir(parents=True, exist_ok=True)
