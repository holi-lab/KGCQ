"""Retriever test queries: extract the patient's symptoms from each HG test dialogue with GPT-4o-mini."""
import json
import os
import sys
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd

# GPT calls go through Azure OpenAI (kgcq.models); credentials from .env
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repository root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from kgcq.models import get_openai_response  # noqa: E402

file_lock = threading.Lock()

class KnowledgeGraph:
    """Attribute nodes linked to each disease in the KG."""
    def __init__(self, nodes_path, edges_path):
        self.nodes_df = pd.read_csv(nodes_path)
        self.edges_df = pd.read_csv(edges_path)
        
        self.disease_to_id = {}
        for _, row in self.nodes_df[self.nodes_df['label'] == 'Disease'].iterrows():
            self.disease_to_id[row['name'].lower()] = row['id']
        
        self.id_to_name = dict(zip(self.nodes_df['id'], self.nodes_df['name']))
        
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

def extract_symptoms_with_llm(conversation_text):
    """Extract the patient's symptoms from a dialogue with the LLM."""
    
    prompt = f"""Below is a conversation between a doctor and a patient. Please extract the symptoms that the patient is experiencing from this conversation.

Conversation:
{conversation_text}

Requirements:
1. Extract the chief complaints (symptoms explicitly mentioned by the patient).
2. Extract the present illness positive (symptoms implicitly mentioned by the patient).
3. Separate each symptom with a semicolon (;).

Please respond in JSON format:
{{
  "chiefcomplaint": "symptom1; symptom2; symptom3",
  "present_illness_positive": "symptom1; symptom2; symptom3"
}}
"""
    
    try:
        content = get_openai_response(
            "gpt-4o-mini",
            [
                {"role": "system", "content": "You are a medical expert who extracts symptoms from patient-doctor conversations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(content)
        return result.get("chiefcomplaint", ""), result.get("present_illness_positive", "")
    
    except Exception as e:
        print(f"Error extracting symptoms: {e}")
        return "", ""

def load_existing_results(output_path):
    """Load existing results."""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_result(output_path, data):
    """Save results (thread-safe)."""
    with file_lock:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def process_test_sample(args):
    """Process one test sample."""
    sample, kg, existing_results = args
    
    patient_id = sample['patient_id']
    
    if patient_id in existing_results:
        return None
    
    prompt = sample['prompt']
    conversation_start = prompt.find("Conversation:\n")
    conversation_end = prompt.find("\n\nBased on")
    
    if conversation_start == -1 or conversation_end == -1:
        return None
    
    conversation_text = prompt[conversation_start + len("Conversation:\n"):conversation_end]
    
    chiefcomplaint, present_illness_positive = extract_symptoms_with_llm(conversation_text)
    
    if not chiefcomplaint and not present_illness_positive:
        return None
    
    ground_truth_diseases = sample.get('true_labels', [])
    
    ground_truth_symptoms = set()
    for disease in ground_truth_diseases:
        connected_nodes = kg.get_disease_connected_nodes(disease)
        ground_truth_symptoms.update(connected_nodes)
    
    return {
        "patient_id": patient_id,
        "ground_truth": ground_truth_diseases,
        "chiefcomplaint": chiefcomplaint,
        "present_illness_positive": present_illness_positive,
        "ground_truth_symptoms": sorted(list(ground_truth_symptoms))
    }

def create_test_dataset(input_path, output_path, kg_nodes_path, kg_edges_path, max_workers=10, max_samples=None):
    """Extract the dialogues from test_predictions.json and build the test queries with the LLM."""
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loading KG from {kg_nodes_path}, {kg_edges_path}...")
    kg = KnowledgeGraph(kg_nodes_path, kg_edges_path)
    
    print(f"Loading existing results from {output_path}...")
    existing_results = load_existing_results(output_path)
    
    tasks = []
    total_existing = len(existing_results)
    
    print(f"Preparing tasks for {len(data)} samples...")
    for sample in data:
        patient_id = sample.get('patient_id')
        
        if not patient_id:
            continue
        
        if patient_id in existing_results:
            continue
        
        tasks.append((sample, kg, existing_results))
        
        if max_samples is not None and len(tasks) >= max_samples:
            break
    
    print(f"Already processed: {total_existing} samples")
    print(f"Remaining tasks to process: {len(tasks)}")
    
    if len(tasks) == 0:
        print("All samples already processed!")
        return
    
    print(f"Processing with {max_workers} workers...")
    processed_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_test_sample, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting symptoms"):
            result = future.result()
            if result is not None:
                processed_results.append(result)
    
    print(f"\nMerging {len(processed_results)} new results with existing data...")
    final_results = existing_results.copy()
    
    for result in processed_results:
        patient_id = result['patient_id']
        final_results[patient_id] = {
            "ground_truth": result['ground_truth'],
            "mapped_symptoms": {
                "chiefcomplaint": result['chiefcomplaint'],
                "present_illness_positive": result['present_illness_positive']
            },
            "ground_truth_symptoms": result['ground_truth_symptoms']
        }
    
    print(f"Saving final results to {output_path}...")
    save_result(output_path, final_results)
    
    print(f"\n\nTotal processed in this run: {len(processed_results)}")
    print(f"Total samples in file: {total_existing + len(processed_results)}")
    print(f"Results saved to: {output_path}")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create test dataset with symptom extraction')
    
    parser.add_argument('--input', type=str,
                        default=os.path.join(REPO, 'models/hg_qwen2.5-7b_clf_head/test_predictions.json'),
                        help='Path to input test_predictions.json')
    parser.add_argument('--output', type=str,
                        default=os.path.join(REPO, 'data/retriever/test_augmented_symptoms.json'),
                        help='Path to output test dataset JSON file')
    parser.add_argument('--kg_nodes_path', type=str,
                        default=os.path.join(REPO, 'data/kg/original/nodes.csv'),
                        help='Path to KG nodes file')
    parser.add_argument('--kg_edges_path', type=str,
                        default=os.path.join(REPO, 'data/kg/original/edges.csv'),
                        help='Path to KG edges file')
    parser.add_argument('--max_workers', type=int, default=10,
                        help='Maximum number of parallel workers')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    create_test_dataset(
        args.input,
        args.output,
        args.kg_nodes_path,
        args.kg_edges_path,
        args.max_workers,
        args.max_samples
    )
