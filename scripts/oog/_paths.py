"""Paths shared by the out-of-graph scripts (relative to the repository root; override with KGCQ_ROOT)."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("KGCQ_ROOT", HERE.parent.parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
KG_DIR = ROOT / "data" / "kg"
OOG_DATA = ROOT / "data" / "oog"
PROFILE_DIR = OOG_DATA / "profiles"           # evaluation profiles (ID 275 / 304, OOG 243 / 128)
TRAIN_DATA_DIR = OOG_DATA / "train"           # data_exp{5,6}[_hv][_v3a]: training sets and HG eval sets
MODEL_DIR = ROOT / "models" / "oog"
_BASE_LOCAL = ROOT / "models" / "base" / "Qwen2.5-7B-Instruct"
_EMB_LOCAL = ROOT / "models" / "base" / "all-MiniLM-L6-v2"
BASE_MODEL = str(_BASE_LOCAL) if _BASE_LOCAL.exists() else "Qwen/Qwen2.5-7B-Instruct"
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
