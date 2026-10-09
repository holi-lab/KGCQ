"""Repository paths. Everything is relative to the repository root so the folder can be moved freely.

Override the root with the environment variable ``KGCQ_ROOT`` if the package is installed elsewhere.
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("KGCQ_ROOT", Path(__file__).resolve().parent.parent))
DATA_DIR = ROOT / "data"
KG_DIR = DATA_DIR / "kg"
PROFILE_DIR = DATA_DIR / "profiles"
DIALOGUE_DIR = DATA_DIR / "dialogues"
SOFTLABEL_DIR = DATA_DIR / "hg_softlabel"
PROMPT_DIR = ROOT / "prompts"
MODEL_DIR = ROOT / "models"
RUN_DIR = ROOT / "runs"
RESULT_DIR = ROOT / "results"
ENV_FILE = ROOT / ".env"

# Default knowledge graph (the 338-disease graph used in the main experiments)
KG_NODES = KG_DIR / "original" / "nodes.csv"
KG_EDGES = KG_DIR / "original" / "edges.csv"

# Prompts
HV_DOCTOR_PROMPT = PROMPT_DIR / "hv_doctor.txt"                     # Hypothesis Verifier (inference + SFT)
HV_DOCTOR_NO_KG_PROMPT = PROMPT_DIR / "hv_doctor_no_kg.txt"         # parametric-only baseline
SYNTH_DOCTOR_GOLD_PROMPT = PROMPT_DIR / "synth_doctor_gold.txt"     # gold-conditioned clinician for synthetic dialogues
SYNTH_DOCTOR_OOG_PROMPT = PROMPT_DIR / "synth_doctor_gold_oog_other.txt"  # out-of-graph ("Other") synthetic dialogues
HG_GENERATIVE_PROMPT = PROMPT_DIR / "hg_generative.txt"             # generative HG baseline
PATIENT_PROMPT = PROMPT_DIR / "patientsim" / "patient_low_specificity.txt"   # ours (PatientSim + low specificity)
PATIENT_PROMPT_ORIGINAL = PROMPT_DIR / "patientsim" / "patient_original_patientsim.txt"
PERSONA_JSON = PROMPT_DIR / "patientsim" / "persona.json"

# Models (symlinks or real directories under models/)
HG_MODEL_DIR = MODEL_DIR / "hg_qwen2.5-7b_clf_head"
HV_MODEL_DIR = MODEL_DIR / "hv_qwen2.5-7b_sft_lora"
BASE_LLM = "Qwen/Qwen2.5-7B-Instruct"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_env():
    """Load API keys (OPENAI_API_KEY, OPENROUTER_API_KEY, WANDB_API_KEY) from, in order of precedence,
    the file named by $KGCQ_ENV_FILE, <ROOT>/.env, and ./.env. Existing environment variables are kept."""
    from dotenv import load_dotenv
    custom = os.environ.get("KGCQ_ENV_FILE")
    if custom:
        load_dotenv(custom)
    load_dotenv(ENV_FILE)
    load_dotenv()
