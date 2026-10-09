#!/usr/bin/env python
"""Oracle subgraphs for synthetic dialogue generation.

Default (paper Sec. 3.5.1): full 3-hop expansion from each profile's gold disease(s), linearized in the
disease-centric format ("Disease 'd' has symptoms: [...]").

--hg_prune: gold-anchored, probability-pruned variant (gold + top-k HG anchors, 3-hop, competing diseases kept only
if HG probability >= tau). The HV training subgraphs released in data/dialogues/training_subgraph_hv_*.json are of
this pruned kind (about 26 diseases, 41 lines per profile); they were produced by an earlier version of the
pipeline and cannot be regenerated bit-exactly, see README.md (Known gaps). The HG is queried with the profile's chief
complaint as the first patient utterance.

Output {pid: {"ground_truth": [...], "subgraph": [lines]}} -- the file consumed by generate_dialogues.py.
"""
import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from tqdm import tqdm  # noqa: E402

from kgcq import paths  # noqa: E402
from kgcq.utils import load_json, load_csv, save_to_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--kg_nodes", default=str(paths.KG_NODES))
    ap.add_argument("--kg_edges", default=str(paths.KG_EDGES))
    ap.add_argument("--fmt", default="disease", choices=["disease", "symptom"])
    ap.add_argument("--hg_prune", action="store_true", help="gold-anchored, HG-probability-pruned variant")
    ap.add_argument("--hg_model", default=str(paths.HG_MODEL_DIR))
    ap.add_argument("--hg_base", default=paths.BASE_LLM)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--gpu", default="0")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.hg_prune else ""

    from kgcq.graph import DiagnosticKnowledgeGraph
    from kgcq.subgraph_extractor import extract_oracle_subgraph, extract_subgraph_3hop_with_gold

    kg = DiagnosticKnowledgeGraph(load_csv(args.kg_nodes), load_csv(args.kg_edges))
    profiles = load_json(args.profiles)
    detector = None
    if args.hg_prune:
        from kgcq.models import DiseaseDetector, disease_names_from_nodes, load_label_config
        detector = DiseaseDetector(args.hg_model, load_label_config(args.hg_model) or disease_names_from_nodes(args.kg_nodes),
                                   base_model=args.hg_base)
    out, missing = {}, 0
    for pid, prof in tqdm(profiles.items()):
        gold = prof['disease_mapped']
        if not all(kg.node_name2id(d) for d in gold):
            missing += 1
            continue
        if detector is None:
            _, lines = extract_oracle_subgraph(kg, gold, fmt=args.fmt)
        else:
            first_utt = "; ".join(prof.get('chiefcomplaint_new') or [prof.get('chiefcomplaint', '')])
            result, ranked = detector.rank_diseases([{"role": "user", "content": first_utt}], top_k=args.top_k)
            _, lines = extract_subgraph_3hop_with_gold(kg, [d for d, _ in result['ranked_list']], dict(ranked),
                                                       args.tau, gold, fmt=args.fmt)
        out[pid] = {"ground_truth": gold, "subgraph": lines if isinstance(lines, list) else [lines]}
    save_to_json(out, args.output)
    print(f"saved {len(out)} oracle subgraphs -> {args.output} (profiles without KG-mapped gold: {missing})")


if __name__ == "__main__":
    main()
