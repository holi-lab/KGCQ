#!/usr/bin/env python
"""Synthetic dialogue generation: a gold-conditioned clinician (prompts/synth_doctor_gold.txt) and the patient
simulator, both gpt-4o-mini. One dialogue per chief complaint; output <output_dir>/dialog.json (resumable).

  python scripts/1_synth/generate_dialogues.py --profiles data/profiles/hv_train.json \
      --oracle_subgraphs data/dialogues/oracle_subgraph_hv_train.json --output_dir data/dialogues/hv_train_gen
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from tqdm import tqdm  # noqa: E402

from kgcq import paths  # noqa: E402
from kgcq.utils import load_json, load_csv, file_to_string, save_to_json, set_global_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--oracle_subgraphs", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--doctor_model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--patient_model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--doctor_prompt", default=str(paths.SYNTH_DOCTOR_GOLD_PROMPT))
    ap.add_argument("--patient_prompt", default=str(paths.PATIENT_PROMPT))
    ap.add_argument("--kg_nodes", default=str(paths.KG_NODES))
    ap.add_argument("--kg_edges", default=str(paths.KG_EDGES))
    ap.add_argument("--embedding_model", default=paths.EMBEDDING_MODEL)
    ap.add_argument("--max_turns", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_patients", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    set_global_seed(args.seed)

    from kgcq.graph import DiagnosticKnowledgeGraph
    from kgcq.models import EmbeddingModel, get_openai_response
    from kgcq.pipeline import SyntheticDialogueGenerator
    from kgcq.metrics import aggregate_run

    profiles = load_json(args.profiles)
    subgraphs = load_json(args.oracle_subgraphs)
    kg = DiagnosticKnowledgeGraph(load_csv(args.kg_nodes), load_csv(args.kg_edges),
                                  embedding_model=EmbeddingModel(args.embedding_model))

    def generate_q(messages, temperature=0.0):
        return get_openai_response(args.doctor_model, messages, temperature=temperature)

    def generate_p(messages, temperature=0.0):
        return get_openai_response(args.patient_model, messages, temperature=temperature)

    manager = SyntheticDialogueGenerator(kg=kg, doctor_prompt=file_to_string(args.doctor_prompt),
                                         patient_prompt=file_to_string(args.patient_prompt),
                                         doctor_generate_func=generate_q, patient_generate_func=generate_p,
                                         max_turn=args.max_turns)

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "dialog.json")
    all_results = load_json(results_path) if os.path.exists(results_path) else {}
    todo = [(pid, p) for pid, p in profiles.items() if pid not in all_results and pid in subgraphs]
    skipped = sum(1 for pid in profiles if pid not in subgraphs)
    if args.max_patients:
        todo = todo[:args.max_patients]
    print(f"done {len(all_results)}, todo {len(todo)}, no-oracle-subgraph {skipped}")

    def save(pid, symptom, result):
        result[symptom] = manager.snapshot()
        all_results[pid] = result
        save_to_json(all_results, results_path)
        save_to_json(aggregate_run(all_results), os.path.join(args.output_dir, "metrics.json"))

    for pid, profile in tqdm(todo):
        profile['disease'] = profile['disease_mapped']
        profile['subgraph'] = subgraphs[pid]['subgraph']
        result = {"ground_truth": profile['disease'], "subgraph": profile['subgraph']}
        for symptom in profile['chiefcomplaint_new']:
            manager.run(profile, max_turns=args.max_turns, on_step=lambda m, s=symptom, r=result, p=pid: save(p, s, r))


if __name__ == "__main__":
    main()
