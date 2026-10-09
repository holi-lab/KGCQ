#!/usr/bin/env python
"""Offline smoke test of the KGCQ pipeline: real HG + HV models, a scripted (mock) patient, no API calls.

Checks model loading, HG ranking, 3-hop subgraph extraction with tau and the HV's <question>/<diagnosis>
generation on one profile. Prints the dialogue and the final Recall@k.

  python scripts/4_infer/smoke_test_offline.py --gpu 0
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from kgcq import paths  # noqa: E402
from kgcq.utils import load_json, load_csv, file_to_string, set_global_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--profiles", default=str(paths.PROFILE_DIR / "test_balanced_275.json"))
    ap.add_argument("--patient_index", type=int, default=0)
    ap.add_argument("--max_turns", type=int, default=8)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_global_seed(42)

    from kgcq.graph import DiagnosticKnowledgeGraph
    from kgcq.models import EmbeddingModel, DiseaseDetector, CausalLanguageModel, disease_names_from_nodes
    from kgcq.pipeline import ConversationalDiagnosis

    kg = DiagnosticKnowledgeGraph(load_csv(paths.KG_NODES), load_csv(paths.KG_EDGES), embedding_model=EmbeddingModel(paths.EMBEDDING_MODEL))
    profiles = load_json(args.profiles)
    pid, profile = list(profiles.items())[args.patient_index]
    profile['disease'] = profile['disease_mapped']
    print(f"patient {pid}: gold={profile['disease']}, chief complaints={profile['chiefcomplaint_new']}")

    # scripted patient: first the chief complaint, then answers drawn from the profile text
    pos = str(profile.get('present_illness_positive', ''))
    scripted = [f"I have {profile['chiefcomplaint_new'][0].lower()}, it just feels weird.",
                f"Hmm, I think so. {pos[:120]}", "No, I don't think so.", "I'm not sure, maybe.",
                "Not really.", "Yes, a bit.", "I don't know.", "No."]
    state = {"i": 0}

    def fake_patient(messages, temperature=0.0):
        r = scripted[min(state["i"], len(scripted) - 1)]
        state["i"] += 1
        return r

    hg = DiseaseDetector(str(paths.HG_MODEL_DIR), disease_names_from_nodes(paths.KG_NODES), base_model=paths.BASE_LLM)
    hv = CausalLanguageModel(paths.BASE_LLM)
    hv.load_model_for_inference(is_finetuned=True, finetuned_model_dir=str(paths.HV_MODEL_DIR))

    manager = ConversationalDiagnosis(
        kg=kg, doctor_prompt=file_to_string(paths.HV_DOCTOR_PROMPT), patient_prompt=file_to_string(paths.PATIENT_PROMPT),
        doctor_generate_func=lambda m, temperature=0.0: hv.get_response(m, temperature=temperature),
        patient_generate_func=fake_patient, kg_extract_type="3hop", top_k=2, threshold_tau=0.005,
        disease_detector=hg, max_turn=50)
    snap = manager.run(profile, max_turns=args.max_turns)
    for t in snap['dialogue']:
        print(f"  [{t['role']}] {t['content'][:160]}")
    probs = [t for t in snap['full_process'] if t['role'] == 'disease_prob']
    print("HG top-k at each turn:", [[d for d, _ in p['content']['top_k_diseases']] for p in probs])
    print("subgraph lines (last turn):", len([t for t in snap['full_process'] if t['role'] == 'subgraph_extractor'][-1]['content']))
    print("final diagnosis:", snap['diagnosis'], "| recall:", snap['recall'], "| turns:", snap['dialogue_length'])


if __name__ == "__main__":
    main()
