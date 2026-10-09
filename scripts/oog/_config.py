"""Shared settings for the out-of-graph (OOG) training / inference / evaluation scripts."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _paths as P  # noqa: E402

RV2 = P.RV2
ROOT = str(P.ROOT)
ENV_FILE = str(P.ENV_FILE)
BASE_MODEL = P.BASE_MODEL
EMB_MODEL = P.EMB_MODEL
PROMPTS_DIR = str(P.PROMPT_DIR)
PATIENT_PROMPT_PATH = f'{PROMPTS_DIR}/patientsim/patient_low_specificity.txt'
PATIENT_MODEL = "gpt-4o-mini-2024-07-18"

KG = P.KG
PREPROCESSED = {k: str(P.PREPROCESSED_DIR / f'{k}.json') for k in ('exp1_id', 'exp1_ood', 'exp2_id')}
PROFILE_DIR = str(P.PROFILE_DIR)

# ratio-based subgraph builder settings (legacy `--subgraph_method ratio`; the paper runs use paper3hop + tau)
SUBGRAPH_CFG = dict(feature_mode='cap_own', cap=7, ratio=0.25)
SUBGRAPH_CFG_BY_KG = {
    'paper': dict(feature_mode='cap_own', cap=7, ratio=0.25),
    'augmented': dict(feature_mode='cap_own', cap=7, ratio=0.11),
    'augmented_v2': dict(feature_mode='cap_own', cap=7, ratio=0.18),
    'augmented_v3': dict(feature_mode='cap_own', cap=7, ratio=0.20),
}
