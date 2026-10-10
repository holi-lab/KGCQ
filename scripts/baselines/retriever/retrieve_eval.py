"""Standalone Recall@k of an encoder retriever over the KG disease documents (Fig. 3 SapBERT curve)."""
import os
import json
import torch
import argparse
import pandas as pd
from retrieve_utils import encoder
from glob import glob
from datetime import datetime

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repository root
class KnowledgeGraph:
    """Symptoms linked to each disease in the KG."""
    def __init__(self, nodes_path, edges_path):
        self.nodes_df = pd.read_csv(nodes_path)
        self.edges_df = pd.read_csv(edges_path)
        
        self.disease_to_id = {}
        for _, row in self.nodes_df[self.nodes_df['label'] == 'Disease'].iterrows():
            self.disease_to_id[row['name'].lower()] = row['id']
        
        self.id_to_name = dict(zip(self.nodes_df['id'], self.nodes_df['name']))
        
    def get_disease_symptoms(self, disease_name):
        """Return the symptom nodes linked to a disease."""
        disease_name_lower = disease_name.lower()
        
        if disease_name_lower not in self.disease_to_id:
            return []
        
        disease_id = self.disease_to_id[disease_name_lower]
        
        connected_nodes = self.edges_df[
            (self.edges_df['end'] == disease_id) & 
            (self.edges_df['type'] == 'caused_by')
        ]['start'].tolist()
        
        symptom_names = [self.id_to_name.get(node_id, '') for node_id in connected_nodes]
        return [s for s in symptom_names if s]
    
    def get_disease_connected_nodes(self, disease_name):
        """Return all attribute nodes linked to a disease."""
        disease_name_lower = disease_name.lower()
        
        if disease_name_lower not in self.disease_to_id:
            return []
        
        disease_id = self.disease_to_id[disease_name_lower]
        
        connected_nodes = self.edges_df[
            (self.edges_df['end'] == disease_id) & 
            (self.edges_df['type'] == 'caused_by')
        ]['start'].tolist()
        
        node_names = [self.id_to_name.get(node_id, '') for node_id in connected_nodes]
        return [n for n in node_names if n]

def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.model_path and os.path.exists(args.model_path):
        model_path = os.path.abspath(args.model_path)
        print(f"Loading trained model from: {model_path}")
        model = encoder(model_path)
    else:
        print("Trained model not found. Using base model for evaluation.")
        base_model = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
        model = encoder(base_model)
    model.to(device)
    model.eval()
    
    tokenizer = model.encoder_tokenizer
    
    print(f"Loading KG from: {args.kg_nodes_path}")
    kg = KnowledgeGraph(args.kg_nodes_path, args.kg_edges_path)
    
    disease_nodes = kg.nodes_df[kg.nodes_df['label'] == 'Disease']
    all_diseases = []
    all_disease_descriptions = []
    
    for _, row in disease_nodes.iterrows():
        disease_name = row['name']
        connected_nodes = kg.get_disease_connected_nodes(disease_name)
        disease_description = "; ".join(connected_nodes) if connected_nodes else disease_name
        
        all_diseases.append(disease_name.lower())
        all_disease_descriptions.append(disease_description)
    
    print(f"Total diseases in KG: {len(all_diseases)}")
    
    print("Encoding all disease descriptions...")
    all_disease_tokens = tokenizer(
        all_disease_descriptions,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=256
    )
    all_disease_ids = all_disease_tokens['input_ids'].to(device)
    all_disease_mask = all_disease_tokens['attention_mask'].to(device)
    
    with torch.no_grad():
        all_disease_embs = model.grad_encode(all_disease_ids, all_disease_mask)
    
    print(f"Disease embeddings shape: {all_disease_embs.shape}")
    
    print(f"Loading test data from: {args.test_path}")
    with open(args.test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print("Evaluating...")
    recalls = {1: [], 2: [], 3: [], 4: []}
    total_samples = 0
    
    for patient_id, patient_data in test_data.items():
        mapped_symptoms = patient_data.get('mapped_symptoms', {})
        chiefcomplaint = mapped_symptoms.get('chiefcomplaint', '')
        present_illness_positive = mapped_symptoms.get('present_illness_positive', '')
        
        symptoms_parts = []
        if chiefcomplaint:
            symptoms_parts.append(chiefcomplaint)
        if present_illness_positive:
            symptoms_parts.append(present_illness_positive)
        
        if not symptoms_parts:
            continue
        
        symptoms_text = "; ".join(symptoms_parts)
        
        ground_truth_diseases = patient_data.get('ground_truth', [])
        if not ground_truth_diseases:
            continue
        
        ground_truth = set([d.lower() for d in ground_truth_diseases])
        
        symptom_tokens = tokenizer(
            [symptoms_text],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=256
        )
        symptom_ids = symptom_tokens['input_ids'].to(device)
        symptom_mask = symptom_tokens['attention_mask'].to(device)
        
        with torch.no_grad():
            symptom_emb = model.grad_encode(symptom_ids, symptom_mask)
        
        similarity = torch.matmul(symptom_emb, all_disease_embs.t()).squeeze(0)
        
        top_k_values, top_k_indices = torch.topk(similarity, k=max(recalls.keys()), largest=True)
        
        for k in recalls.keys():
            top_k_diseases = [all_diseases[idx] for idx in top_k_indices[:k].cpu().tolist()]
            
            hit = any(gt in top_k_diseases for gt in ground_truth)
            recalls[k].append(1 if hit else 0)
        
        total_samples += 1
        
        if total_samples % 10 == 0:
            print(f"Processed {total_samples} samples...")
    
    results = {
        'total_samples': total_samples,
        'recalls': {}
    }
    
    for k in sorted(recalls.keys()):
        recall_k = sum(recalls[k]) / len(recalls[k]) * 100 if recalls[k] else 0
        results['recalls'][k] = recall_k
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Disease Retriever')
    
    parser.add_argument('--model_path', type=str,
                        default=None,
                        help='Path to trained model (if None, evaluates all models in train/)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU number to use')
    
    parser.add_argument('--test_path', type=str,
                        default=os.path.join(REPO, 'data/retriever/test_augmented_symptoms.json'),
                        help='Path to test data')
    parser.add_argument('--kg_nodes_path', type=str,
                        default=os.path.join(REPO, 'data/kg/original/nodes.csv'),
                        help='Path to KG nodes file')
    parser.add_argument('--kg_edges_path', type=str,
                        default=os.path.join(REPO, 'data/kg/original/edges.csv'),
                        help='Path to KG edges file')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(REPO, 'results/retriever'),
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.model_path:
        model_paths = [args.model_path]
    else:
        model_paths = glob('./train/trained_models/trained_*')
        model_paths = [p for p in model_paths if os.path.isdir(p)]
        print(f"Found {len(model_paths)} models in train/trained_models/")
    
    output_file = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.txt")
    
    all_results = []
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Evaluation Results - {timestamp}\n")
        f.write("="*70 + "\n\n")
        
        for model_path in model_paths:
            model_name = os.path.basename(model_path)
            print(f"\n{'='*70}")
            print(f"Evaluating: {model_name}")
            print(f"{'='*70}")
            
            f.write(f"Model: {model_name}\n")
            f.write(f"Path: {model_path}\n")
            f.write("-"*70 + "\n")
            
            args.model_path = model_path
            try:
                results = evaluate(args)
                
                print(f"\nResults on {results['total_samples']} samples:")
                for k in sorted(results['recalls'].keys()):
                    recall_k = results['recalls'][k]
                    print(f"Recall@{k}: {recall_k:.2f}%")
                
                f.write(f"Total samples: {results['total_samples']}\n")
                for k in sorted(results['recalls'].keys()):
                    recall_k = results['recalls'][k]
                    f.write(f"Recall@{k}: {recall_k:.2f}%\n")
                f.write("\n")
                
                all_results.append({
                    'model': model_name,
                    'results': results
                })
                
            except Exception as e:
                error_msg = f"Error evaluating {model_name}: {str(e)}"
                print(error_msg)
                f.write(f"ERROR: {str(e)}\n\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Summary\n")
        f.write("="*70 + "\n\n")
        
        for result in all_results:
            f.write(f"{result['model']:30s} | ")
            for k in sorted(result['results']['recalls'].keys()):
                recall_k = result['results']['recalls'][k]
                f.write(f"R@{k}: {recall_k:5.2f}% | ")
            f.write("\n")
    
    print(f"\n{'='*70}")
    print(f"Evaluation complete! Results saved to: {output_file}")
    print(f"{'='*70}")
