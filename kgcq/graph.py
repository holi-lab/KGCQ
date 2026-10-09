"""Diagnostic knowledge graph.

nodes.csv columns: id, name, label (Disease | Symptom | Cause | Risk_Factor)
edges.csv columns: start, end, type   (attribute -> disease; caused_by | can_cause | is_a_risk_factor_of)
"""
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd


class DiagnosticKnowledgeGraph:
    def __init__(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame, embedding_model=None):
        self.nodes_df = nodes_df
        self.edges_df = edges_df
        self.graph = self._build_graph()
        self.node_names = self._prepare_node_texts()
        self.embedding_model = embedding_model
        if self.embedding_model is not None:
            self.node_embeddings = self.embedding_model.encode(self.node_names)
        else:
            self.node_embeddings = None
        self.disease_names = [row['name'] for _, row in nodes_df.iterrows() if row['label'] == 'Disease']
        self.symptom_names = [row['name'] for _, row in nodes_df.iterrows() if row['label'] != 'Disease']
        self.disease_ids = [row['id'] for _, row in nodes_df.iterrows() if row['label'] == 'Disease']
        self.symptom_ids = [row['id'] for _, row in nodes_df.iterrows() if row['label'] != 'Disease']
        self._id2row = {row['id']: {'id': row['id'], 'name': row['name'], 'label': row['label']}
                        for _, row in nodes_df.iterrows()}
        self._name2row = {}
        for _, row in nodes_df.iterrows():
            self._name2row.setdefault(row['name'], {'id': row['id'], 'name': row['name'], 'label': row['label']})

    @classmethod
    def from_csv(cls, nodes_csv, edges_csv, embedding_model=None):
        return cls(pd.read_csv(nodes_csv), pd.read_csv(edges_csv), embedding_model=embedding_model)

    def _build_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for _, node in self.nodes_df.iterrows():
            G.add_node(node['id'], label=node['label'], name=node['name'])
        for _, row in self.edges_df.iterrows():
            G.add_edge(row['start'], row['end'], type=row['type'])
        return G

    def _prepare_node_texts(self) -> List[str]:
        return [node['name'] for _, node in self.nodes_df.iterrows()]

    # ------------------------------------------------------------------ lookups
    def node_id2name(self, node_id: str) -> Dict:
        return dict(self._id2row.get(node_id, {}))

    def node_name2id(self, node_name: str) -> Dict:
        return dict(self._name2row.get(node_name, {}))

    def node_name2id_symptom(self, disease_names, symptom_or_risk_names):
        disease_ids = []
        for disease_name in disease_names:
            for node_id, data in self.graph.nodes(data=True):
                if data.get('label') == 'Disease' and data.get('name') == disease_name:
                    disease_ids.append(node_id)
        related_ids = []
        for disease_id in disease_ids:
            for neighbor in self.graph.predecessors(disease_id):
                neighbor_data = self.graph.nodes[neighbor]
                if neighbor_data.get('label') in ['Symptom', 'Risk_Factor', 'Cause']:
                    if neighbor_data.get('name') in symptom_or_risk_names:
                        related_ids.append(neighbor)
        return related_ids

    def node_name2id_disease(self, disease_names, subgraph=None):
        target_graph = subgraph if subgraph is not None else self.graph
        disease_ids = []
        for disease_name in disease_names:
            for node_id, data in target_graph.nodes(data=True):
                if data.get('label') == 'Disease' and data.get('name') == disease_name:
                    disease_ids.append(node_id)
        return disease_ids

    def get_subgraph_disease_names(self, subgraph):
        return [data.get('name') for _, data in subgraph.nodes(data=True) if data.get('label') == 'Disease']

    # ------------------------------------------------------------------ embedding search
    def find_similar_nodes(self, text, top_k, threshold, node_subset) -> List[Tuple[str, str, float]]:
        """Return [(node_id, node_name, cosine_similarity)] among ``node_subset`` (list of node ids)."""
        if self.embedding_model is None or self.node_embeddings is None:
            raise ValueError("Embedding model not initialized. Pass embedding_model to constructor.")
        query_embedding = self.embedding_model.encode([text])
        if node_subset is None:
            subset_indices = list(range(len(self.nodes_df)))
        else:
            subset = set(node_subset)
            subset_indices = [i for i, row in self.nodes_df.iterrows() if row['id'] in subset]
        similar_results = self.embedding_model.get_similar_indices(
            query_embedding, self.node_embeddings[subset_indices], threshold, top_k)
        if not similar_results:
            return []
        results = []
        for idx, similarity in similar_results:
            global_idx = subset_indices[idx]
            results.append((self.nodes_df.iloc[global_idx]['id'], self.nodes_df.iloc[global_idx]['name'], similarity))
        return results

    def find_matching_disease(self, disease_query):
        """Map a free-text diagnosis to a disease node: exact match, else nearest disease by embedding."""
        if disease_query in self.disease_names:
            return disease_query, 1.0
        similar_diseases = self.find_similar_nodes(disease_query, top_k=1, threshold=0.0, node_subset=self.disease_ids)
        _, node_name, sim = similar_diseases[0]
        return node_name, sim

    # ------------------------------------------------------------------ subgraphs
    def get_subgraph(self, node_ids: List[str], hops: int = 2) -> nx.Graph:
        """Undirected k-hop expansion around ``node_ids`` (returns a node-induced subgraph view)."""
        if not node_ids:
            return nx.Graph()
        subgraph_nodes = set(node_ids)
        current_nodes = set(node_ids)
        for _ in range(hops):
            next_nodes = set()
            for node in current_nodes:
                if node in self.graph:
                    next_nodes.update(self.graph.successors(node))
                    next_nodes.update(self.graph.predecessors(node))
            current_nodes = next_nodes - subgraph_nodes
            subgraph_nodes.update(current_nodes)
        return self.graph.subgraph(subgraph_nodes)

    def get_subgraph_with_shared_disease(self, node_ids: List[str], min_shared: int = 2) -> nx.Graph:
        """2-hop expansion from attribute nodes: keep diseases connected to >= min_shared of ``node_ids``
        and add all attributes of those diseases (used by the HG-free "+KG" ablation)."""
        if not node_ids:
            return nx.Graph()
        disease_to_nodes = {}
        for node_id in node_ids:
            for neighbor in self.graph.successors(node_id):
                if self.graph.nodes[neighbor].get('label') == 'Disease':
                    disease_to_nodes.setdefault(neighbor, set()).add(node_id)
        selected_diseases = [d for d, nodes in disease_to_nodes.items() if len(nodes) >= min_shared]
        subgraph_nodes = set(node_ids) | set(selected_diseases)
        for disease_id in selected_diseases:
            subgraph_nodes.update(self.graph.predecessors(disease_id))
            subgraph_nodes.update(self.graph.successors(disease_id))
        return self.graph.subgraph(subgraph_nodes)

    def get_subgraph_text(self, subgraph: nx.Graph, disease_constraint: List[str] = (), fmt: str = "disease"):
        """Linearize a subgraph into natural-language statements for the HV prompt.

        Only edges whose disease is in ``disease_constraint`` are kept (probability filtering with tau).
        fmt="disease" (default, used in all main-paper runs; historically called "reverse"):
            Disease 'd' has symptoms: ['s1', 's2'].
        fmt="symptom" (attribute-centric, used in the out-of-graph experiments):
            's1' is a symptom of ['d1', 'd2'].
        Returns a list of lines, or the string "No connections found." for an empty subgraph.
        """
        if not subgraph.nodes():
            return "No connections found."
        constraint = set(disease_constraint)
        s2d, r2d, c2d = {}, {}, {}
        d2s, d2r, d2c = {}, {}, {}
        for start_id, end_id, _ in subgraph.edges(data=True):
            start_info = self.node_id2name(start_id)
            end_info = self.node_id2name(end_id)
            assert end_info['label'] == 'Disease'
            if end_info['name'] not in constraint:
                continue
            if start_info['label'] == "Symptom":
                s2d.setdefault(start_info['name'], []).append(end_info['name'])
                d2s.setdefault(end_info['name'], []).append(start_info['name'])
            elif start_info['label'] == "Risk_Factor":
                r2d.setdefault(start_info['name'], []).append(end_info['name'])
                d2r.setdefault(end_info['name'], []).append(start_info['name'])
            elif start_info['label'] == "Cause":
                c2d.setdefault(start_info['name'], []).append(end_info['name'])
                d2c.setdefault(end_info['name'], []).append(start_info['name'])

        if fmt == "symptom":
            lines = [f"'{s}' is a symptom of {d_list}." for s, d_list in s2d.items()]
            lines += [f"'{r}' is a risk factor for {d_list}." for r, d_list in r2d.items()]
            lines += [f"'{c}' is a cause of {d_list}." for c, d_list in c2d.items()]
            return lines
        lines = [f"Disease '{d}' has symptoms: {s_list}." for d, s_list in d2s.items()]
        lines += [f"Disease '{d}' has risk factors: {r_list}." for d, r_list in d2r.items()]
        lines += [f"Disease '{d}' has causes: {c_list}." for d, c_list in d2c.items()]
        return lines

    def get_subgraph_text_json(self, subgraph: nx.Graph):
        """Alternative block format (one block per disease). Not used in the paper's experiments."""
        if not subgraph.nodes():
            return [], ""
        d2s = {}
        for start_id, end_id, _ in subgraph.edges(data=True):
            start_info = self.node_id2name(start_id)
            end_info = self.node_id2name(end_id)
            assert end_info['label'] == 'Disease', "Disease node expected as end node."
            rel = {"Symptom": "Symptom", "Risk_Factor": "Risk Factor", "Cause": "Cause"}.get(start_info['label'])
            if rel is None:
                continue
            d2s.setdefault(end_info['name'], {"Symptom": [], "Risk Factor": [], "Cause": []})[rel].append(start_info['name'])
        blocks = []
        for d, relations in d2s.items():
            text = f"Disease: {d}\n"
            for rel_type, items in relations.items():
                if items:
                    text += f"  - {rel_type}: {', '.join(items)}\n"
            blocks.append(text.strip())
        return blocks, "\n\n".join(blocks)

    def visualize_subgraph(self, subgraph: nx.Graph, figsize: tuple = (15, 10), save_path: str = None):
        import matplotlib.pyplot as plt
        if not subgraph.nodes():
            print("Empty subgraph - nothing to visualize")
            return
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(subgraph, k=2, iterations=100)
        color_map = {'Symptom': '#FF6B6B', 'Disease': '#4ECDC4', 'Risk_Factor': '#FFD93D', 'Cause': '#A0A0FF', 'default': '#B8B8B8'}
        node_colors, node_labels = [], {}
        for node in subgraph.nodes():
            info = self.node_id2name(node)
            node_labels[node] = info.get('name', node)
            node_colors.append(color_map.get(info.get('label', 'default'), color_map['default']))
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', alpha=0.6, width=1)
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, node_size=1000, alpha=0.8)
        nx.draw_networkx_labels(subgraph, pos, node_labels, font_size=8, font_weight='bold')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
