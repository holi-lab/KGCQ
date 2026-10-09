"""Hypothesis-driven subgraph extraction (paper Sec. 3.3.2).

extract_subgraph_3hop   : KGCQ. anchors = top-n HG diseases; 3-hop expansion; competing diseases kept only if
                          HG probability >= tau.
extract_subgraph_2hop   : "+KG" ablation without the HG. anchors = attribute nodes matched to the patient utterance;
                          diseases sharing >= threshold anchors are expanded to their attributes.
extract_subgraph_1hop   : anchors' direct attributes only.
extract_subgraph_3hop_clf_lm : generative-HG baseline (no probabilities): competing diseases kept if they share
                          >= 30% of an anchor's attributes (Appendix: Generative Hypothesis Generator).
extract_oracle_subgraph : 3-hop expansion from the gold disease(s) (synthetic dialogue generation).
"""


def extract_subgraph_3hop(kg, curr_disease_list, total_disease_probs, prob_threshold, fmt="disease"):
    nodes_in_subgraph = [kg.node_name2id(d)['id'] for d in curr_disease_list]
    subgraph = kg.get_subgraph(nodes_in_subgraph, hops=3).copy()
    constraint = [disease for disease, prob in total_disease_probs.items() if prob >= prob_threshold]
    subgraph_lines = kg.get_subgraph_text(subgraph, constraint, fmt=fmt)
    return subgraph, subgraph_lines


def extract_subgraph_1hop(kg, curr_disease_list, fmt="disease"):
    nodes_in_subgraph = [kg.node_name2id(d)['id'] for d in curr_disease_list]
    subgraph = kg.get_subgraph(nodes_in_subgraph, hops=1).copy()
    subgraph_lines = kg.get_subgraph_text(subgraph, kg.disease_names, fmt=fmt)
    return subgraph, subgraph_lines


def extract_subgraph_2hop(kg, curr_symptom_nodes, fmt="disease"):
    n = len(curr_symptom_nodes)
    if n == 1:
        threshold = 1
    elif n <= 4:
        threshold = 2
    else:
        threshold = n // 5
    subgraph = kg.get_subgraph_with_shared_disease(curr_symptom_nodes, min_shared=threshold).copy()
    subgraph_lines = kg.get_subgraph_text(subgraph, kg.disease_names, fmt=fmt)
    return subgraph, subgraph_lines


def extract_subgraph_3hop_clf_lm(kg, curr_disease_list, overlap_ratio=0.3, fmt="disease"):
    nodes_in_subgraph = kg.node_name2id_disease(curr_disease_list)

    def features(disease_id):
        return {p for p in kg.graph.predecessors(disease_id)
                if kg.graph.nodes[p].get('label') in ['Symptom', 'Risk_Factor', 'Cause']}

    disease2features = {d: features(d) for d in nodes_in_subgraph}
    subgraph = kg.get_subgraph(nodes_in_subgraph, hops=3).copy()
    candidate_disease_ids = [n for n in subgraph.nodes()
                             if kg.graph.nodes[n].get('label') == 'Disease' and n not in nodes_in_subgraph]
    selected_disease_ids = set()
    for cand_id in candidate_disease_ids:
        cand_features = features(cand_id)
        for base_id in nodes_in_subgraph:
            shared = disease2features[base_id] & cand_features
            threshold = 1 if len(disease2features[base_id]) <= 3 else len(disease2features[base_id]) * overlap_ratio
            if len(shared) >= threshold:
                selected_disease_ids.add(cand_id)
                break
    final_nodes = set(nodes_in_subgraph) | selected_disease_ids
    subgraph = kg.get_subgraph(final_nodes, hops=1).copy()
    subgraph_lines = kg.get_subgraph_text(subgraph, kg.disease_names, fmt=fmt)
    return subgraph, subgraph_lines


def extract_oracle_subgraph(kg, gold_disease, fmt="disease"):
    nodes_in_subgraph = [kg.node_name2id(d)['id'] for d in gold_disease]
    subgraph = kg.get_subgraph(nodes_in_subgraph, hops=3).copy()
    subgraph_lines = kg.get_subgraph_text(subgraph, kg.disease_names, fmt=fmt)
    return subgraph, subgraph_lines


def extract_subgraph_3hop_with_gold(kg, curr_disease_list, total_disease_probs, prob_threshold, gold_disease, fmt="disease"):
    """Gold-anchored, probability-pruned 3-hop subgraph (training-time variant).

    The gold disease(s) replace the lowest-ranked anchors, the 3-hop neighbourhood is expanded, and edges of
    competing diseases whose HG probability is below ``prob_threshold`` are dropped (gold edges are always kept).
    This mirrors the extraction used when the released HV training subgraphs were produced (see README.md (Known gaps)).
    """
    curr_disease_list = list(curr_disease_list)
    gold_disease = list(gold_disease)
    new_list = [g for g in gold_disease if g not in curr_disease_list]
    curr_disease_list = new_list + curr_disease_list[:max(0, len(curr_disease_list) - len(new_list))]
    nodes_in_subgraph = [kg.node_name2id(d)['id'] for d in curr_disease_list if kg.node_name2id(d)]
    subgraph = kg.get_subgraph(nodes_in_subgraph, hops=3).copy()
    keep = {d for d, p in total_disease_probs.items() if p >= prob_threshold} | set(gold_disease)
    edges_to_remove = []
    for u, v, _ in subgraph.edges(data=True):
        u_node, v_node = kg.node_id2name(u), kg.node_id2name(v)
        if u_node['label'] == 'Disease' and u_node['name'] not in keep:
            edges_to_remove.append((u, v))
        elif v_node['label'] == 'Disease' and v_node['name'] not in keep:
            edges_to_remove.append((u, v))
    subgraph.remove_edges_from(edges_to_remove)
    subgraph_lines = kg.get_subgraph_text(subgraph, list(keep), fmt=fmt)
    return subgraph, subgraph_lines
