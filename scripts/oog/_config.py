"""Shared settings for the out-of-graph scripts."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _paths as P  # noqa: E402

ENV_FILE = str(P.ENV_FILE)
BASE_MODEL = P.BASE_MODEL
EMB_MODEL = P.EMB_MODEL
KG = P.KG
PATIENT_PROMPT_PATH = f'{P.PROMPT_DIR}/patientsim/patient_low_specificity.txt'
PATIENT_MODEL = "gpt-4o-mini-2024-07-18"
