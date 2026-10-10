#!/usr/bin/env python
"""Run conversational diagnosis on a profile set.

Modes: kgcq (HG -> 3-hop subgraph -> HV), kg_only (symptom-anchored 2-hop, no HG), no_kg (parametric only),
hg_generative. HV backends: openai | openrouter | local | local_finetuned.

  python scripts/3_infer/run_dialogue.py --mode kgcq --hv_backend local_finetuned --tag kgcq_n2_tau0.005
  python scripts/3_infer/run_dialogue.py --mode no_kg --hv_backend openai --hv_model gpt-4.1-mini --tag gpt41mini_no_kg

Outputs runs/<tag>/dialog.json (resumable) and runs/<tag>/metrics.json.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from tqdm import tqdm  # noqa: E402

from kgcq import paths  # noqa: E402
from kgcq.utils import load_json, load_csv, file_to_string, save_to_json, set_global_seed  # noqa: E402


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="kgcq", choices=["kgcq", "kg_only", "no_kg", "hg_generative"])
    p.add_argument("--tag", required=True, help="output folder name under --runs_dir")
    p.add_argument("--runs_dir", default=str(paths.RUN_DIR))
    p.add_argument("--profiles", default=str(paths.PROFILE_DIR / "test_balanced_275.json"))
    p.add_argument("--kg_nodes", default=str(paths.KG_NODES))
    p.add_argument("--kg_edges", default=str(paths.KG_EDGES))
    p.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_patients", type=int, default=0, help="debug: stop after N patients (0 = all)")
    # patient simulator
    p.add_argument("--patient_backend", default="openai", choices=["openai", "openrouter", "local"])
    p.add_argument("--patient_model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--patient_prompt", default=str(paths.PATIENT_PROMPT))
    # hypothesis verifier
    p.add_argument("--hv_backend", default="local_finetuned", choices=["openai", "openrouter", "local", "local_finetuned"])
    p.add_argument("--hv_model", default=paths.BASE_LLM, help="API model name or HF id of the local base model")
    p.add_argument("--hv_adapter", default=str(paths.HV_MODEL_DIR), help="LoRA adapter dir for local_finetuned")
    p.add_argument("--doctor_prompt", default=None, help="defaults to prompts/hv_doctor.txt (hv_doctor_no_kg.txt for no_kg)")
    # hypothesis generator / subgraph
    p.add_argument("--hg_model", default=str(paths.HG_MODEL_DIR), help="HG checkpoint (merged) or LoRA adapter dir")
    p.add_argument("--hg_base", default=paths.BASE_LLM, help="base model when --hg_model is a LoRA adapter")
    p.add_argument("--top_k", type=int, default=2, help="n: number of anchor diseases from the HG")
    p.add_argument("--tau", type=float, default=0.005, help="probability threshold for competing diseases")
    p.add_argument("--subgraph_fmt", default="disease", choices=["disease", "symptom"])
    p.add_argument("--embedding_model", default=paths.EMBEDDING_MODEL)
    p.add_argument("--hg_generative_backend", default="local_finetuned", choices=["openai", "openrouter", "local", "local_finetuned"])
    p.add_argument("--hg_generative_model", default=paths.BASE_LLM)
    p.add_argument("--hg_generative_adapter", default=str(paths.MODEL_DIR / "hg_generative_qwen2.5-7b_lora"))
    p.add_argument("--hg_generative_prompt", default=str(paths.HG_GENERATIVE_PROMPT))
    p.add_argument("--max_turns", type=int, default=50)
    return p


def make_generate_func(backend, model_name, adapter=None, cache=None):
    """Return generate(messages, temperature) -> str for the given backend; local models are loaded lazily."""
    from kgcq.models import get_openai_response, get_openrouter_response, CausalLanguageModel
    cache = cache if cache is not None else {}

    def local():
        key = (model_name, adapter if backend == "local_finetuned" else None)
        if key not in cache:
            llm = CausalLanguageModel(model_name, use_quant=False)
            llm.load_model_for_inference(is_finetuned=(backend == "local_finetuned"), finetuned_model_dir=adapter)
            cache[key] = llm
        return cache[key]

    if backend == "openai":
        return lambda messages, temperature=0.0: get_openai_response(model_name, messages, temperature=temperature)
    if backend == "openrouter":
        return lambda messages, temperature=0.0: get_openrouter_response(model_name, messages, temperature=temperature)
    return lambda messages, temperature=0.0: local().get_response(messages, temperature=temperature)


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_global_seed(args.seed)

    from kgcq.graph import DiagnosticKnowledgeGraph
    from kgcq.models import EmbeddingModel, DiseaseDetector, disease_names_from_nodes, load_label_config
    from kgcq.pipeline import ConversationalDiagnosis, NoKGDiagnosis, GenerativeHGDiagnosis
    from kgcq.metrics import aggregate_run

    profiles = load_json(args.profiles)
    nodes_df, edges_df = load_csv(args.kg_nodes), load_csv(args.kg_edges)
    patient_prompt = file_to_string(args.patient_prompt)
    doctor_prompt = file_to_string(args.doctor_prompt or (paths.HV_DOCTOR_NO_KG_PROMPT if args.mode == "no_kg" else paths.HV_DOCTOR_PROMPT))

    kg = DiagnosticKnowledgeGraph(nodes_df, edges_df, embedding_model=EmbeddingModel(args.embedding_model))
    local_cache = {}
    generate_p = make_generate_func(args.patient_backend, args.patient_model, cache=local_cache)
    generate_q = make_generate_func(args.hv_backend, args.hv_model, adapter=args.hv_adapter, cache=local_cache)

    if args.mode in ("kgcq", "kg_only"):
        detector = None
        if args.mode == "kgcq":
            label_names = load_label_config(args.hg_model) or disease_names_from_nodes(args.kg_nodes)
            detector = DiseaseDetector(args.hg_model, label_names, base_model=args.hg_base)
        manager = ConversationalDiagnosis(
            kg=kg, doctor_prompt=doctor_prompt, patient_prompt=patient_prompt,
            doctor_generate_func=generate_q, patient_generate_func=generate_p,
            kg_extract_type=("3hop" if args.mode == "kgcq" else "2hop"), top_k=args.top_k,
            threshold_tau=args.tau, disease_detector=detector, symptom_extract_func=generate_p,
            max_turn=args.max_turns, subgraph_fmt=args.subgraph_fmt)
    elif args.mode == "no_kg":
        manager = NoKGDiagnosis(kg=kg, doctor_prompt=doctor_prompt, patient_prompt=patient_prompt,
                                doctor_generate_func=generate_q, patient_generate_func=generate_p, max_turn=args.max_turns)
    else:
        generate_d = make_generate_func(args.hg_generative_backend, args.hg_generative_model,
                                        adapter=args.hg_generative_adapter, cache=local_cache)
        manager = GenerativeHGDiagnosis(kg=kg, doctor_prompt=doctor_prompt, patient_prompt=patient_prompt,
                                        doctor_generate_func=generate_q, patient_generate_func=generate_p,
                                        disease_detector_func=generate_d,
                                        disease_detector_prompt=file_to_string(args.hg_generative_prompt),
                                        max_turn=args.max_turns, subgraph_fmt=args.subgraph_fmt)

    output_dir = os.path.join(args.runs_dir, args.tag)
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "dialog.json")
    all_results = load_json(results_path) if os.path.exists(results_path) else {}
    save_to_json(vars(args), os.path.join(output_dir, "args.json"))
    print(f"[{args.mode}] loaded {len(all_results)} finished patients from {results_path}")

    todo = [(pid, prof) for pid, prof in profiles.items() if pid not in all_results]
    if args.max_patients:
        todo = todo[:args.max_patients]
    print(f"patients to process: {len(todo)} / {len(profiles)}")

    def save(pid, symptom, result):
        result[symptom] = manager.snapshot()
        all_results[pid] = result
        save_to_json(all_results, results_path)
        save_to_json(aggregate_run(all_results), os.path.join(output_dir, "metrics.json"))

    for pid, profile in tqdm(todo):
        profile['disease'] = profile['disease_mapped']
        result = {"ground_truth": profile['disease']}
        for symptom in profile['chiefcomplaint_new']:   # one dialogue per chief complaint
            manager.run(profile, max_turns=args.max_turns, on_step=lambda m, s=symptom, r=result, p=pid: save(p, s, r))

    metrics = aggregate_run(all_results)
    print("Recall@1-4:", {k: round(v, 3) for k, v in metrics['recall'].items()}, "| avg turns:", round(metrics['avg_turns'], 2))


if __name__ == "__main__":
    main()
