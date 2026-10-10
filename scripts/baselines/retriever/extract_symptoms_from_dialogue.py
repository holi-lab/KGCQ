"""Retriever training queries: truncate synthetic dialogues and extract the symptoms mentioned so far with GPT-4o-mini."""
import json
import random
import os
import sys
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# GPT calls go through Azure OpenAI (kgcq.models); credentials from .env
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repository root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
from kgcq.models import get_openai_response  # noqa: E402

file_lock = threading.Lock()

def extract_symptoms_with_llm(dialogue_turns):
    """Extract the patient's symptoms from a dialogue with the LLM."""
    dialogue_text = "\n".join([f"{turn['role']}: {turn['content']}" for turn in dialogue_turns])
    
    prompt = f"""Below is a conversation between a doctor and a patient. Please extract the symptoms that the patient is experiencing from this conversation.

Conversation:
{dialogue_text}

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

def save_individual_result(temp_dir, worker_id, data):
    """Save one worker's result to a temporary file."""
    temp_file = os.path.join(temp_dir, f"worker_{worker_id}.json")
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def is_sample_processed(results, patient_id, symptom_key, cut_idx):
    """Check whether a sample was already processed."""
    if patient_id not in results:
        return False
    if symptom_key not in results[patient_id]:
        return False
    dialogue_key = f"dialogue_{cut_idx}"
    return dialogue_key in results[patient_id][symptom_key]

def process_dialogue_sample(args):
    """Process one dialogue sample."""
    patient_id, symptom_key, symptom_data, patient_ground_truth, cut_idx, end_idx, dialogue, temp_dir, worker_id, existing_results = args
    
    if is_sample_processed(existing_results, patient_id, symptom_key, cut_idx):
        return None
    
    cut_dialogue = dialogue[:end_idx + 1]
    
    chiefcomplaint, present_illness_positive = extract_symptoms_with_llm(cut_dialogue)
    
    if not chiefcomplaint and not present_illness_positive:
        return None
    
    dialogue_key = f"dialogue_{cut_idx}"
    
    return {
        "patient_id": patient_id,
        "symptom_key": symptom_key,
        "dialogue_key": dialogue_key,
        "ground_truth": patient_ground_truth,
        "dialogue": cut_dialogue,
        "dialogue_length": len(cut_dialogue),
        "chiefcomplaint": chiefcomplaint,
        "present_illness_positive": present_illness_positive
    }

def create_augmented_dataset(input_path, output_path, augment_ratio=5, max_workers=4, max_samples=None):
    """Truncate dialogues and extract the symptoms mentioned so far to build the training queries."""
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loading existing results from {output_path}...")
    existing_results = load_existing_results(output_path)
    
    temp_dir = os.path.join(os.path.dirname(output_path), 'temp_results')
    os.makedirs(temp_dir, exist_ok=True)
    
    tasks = []
    total_existing = 0
    total_tasks_added = 0
    worker_id = 0
    
    print(f"Preparing tasks for {len(data)} patients...")
    for patient_id, patient_data in data.items():
        if max_samples is not None and total_tasks_added >= max_samples:
            break
        symptom_keys = [k for k in patient_data.keys() if k not in ['ground_truth', 'subgraph']]
        
        for symptom_key in symptom_keys:
            symptom_data = patient_data[symptom_key]
            
            if 'dialogue' not in symptom_data:
                continue
            
            if 'recall' in symptom_data and '4' in symptom_data['recall']:
                if symptom_data['recall']['4'] < 0.5:
                    continue
            
            dialogue = symptom_data['dialogue']
            dialogue_length = len(dialogue)
            
            if dialogue_length < 2:
                continue
            
            user_indices = [i for i, turn in enumerate(dialogue) if turn['role'] == 'user']
            
            if len(user_indices) < 2:
                continue
            
            num_cuts = max(1, len(user_indices) // augment_ratio)
            sampled_indices = random.sample(user_indices, num_cuts)
            
            for cut_idx, end_idx in enumerate(sampled_indices):
                if max_samples is not None and total_tasks_added >= max_samples:
                    break
                    
                if is_sample_processed(existing_results, patient_id, symptom_key, cut_idx):
                    total_existing += 1
                    continue
                
                tasks.append((
                    patient_id, symptom_key, symptom_data, 
                    patient_data['ground_truth'], cut_idx, end_idx, dialogue,
                    temp_dir, worker_id, existing_results
                ))
                total_tasks_added += 1
                worker_id += 1
    
    print(f"Already processed: {total_existing} samples")
    print(f"Remaining tasks to process: {len(tasks)}")
    
    if len(tasks) == 0:
        print("All samples already processed!")
        return
    
    print(f"Processing with {max_workers} workers...")
    processed_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_dialogue_sample, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting symptoms"):
            result = future.result()
            if result is not None:
                processed_results.append(result)
    
    print(f"\nMerging {len(processed_results)} new results with existing data...")
    final_results = existing_results.copy()
    
    for result in processed_results:
        patient_id = result['patient_id']
        symptom_key = result['symptom_key']
        dialogue_key = result['dialogue_key']
        
        if patient_id not in final_results:
            final_results[patient_id] = {
                "ground_truth": result['ground_truth']
            }
        
        if symptom_key not in final_results[patient_id]:
            final_results[patient_id][symptom_key] = {}
        
        final_results[patient_id][symptom_key][dialogue_key] = {
            "dialogue": result['dialogue'],
            "dialogue_length": result['dialogue_length'],
            "mapped_symptoms": {
                "chiefcomplaint": result['chiefcomplaint'],
                "present_illness_positive": result['present_illness_positive']
            }
        }
    
    print(f"Saving final results to {output_path}...")
    save_result(output_path, final_results)
    
    import shutil
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    print(f"\n\nTotal processed in this run: {len(processed_results)}")
    print(f"Total samples in file: {total_existing + len(processed_results)}")
    print(f"Results saved to: {output_path}")
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract symptoms from dialogue using LLM')
    
    parser.add_argument('--input', type=str,
                        default=os.path.join(REPO, 'data/dialogues/hg_train.json'),
                        help='Path to input train_dialog.json')
    parser.add_argument('--output', type=str,
                        default=os.path.join(REPO, 'data/retriever/train_augmented_symptoms.json'),
                        help='Path to output augmented JSON file')
    parser.add_argument('--augment_ratio', type=int, default=5,
                        help='Dialogue length division ratio for augmentation')
    parser.add_argument('--max_workers', type=int, default=50,
                        help='Maximum number of parallel workers')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process (for testing)')
    
    args = parser.parse_args()
    
    create_augmented_dataset(args.input, args.output, args.augment_ratio, args.max_workers, args.max_samples)
