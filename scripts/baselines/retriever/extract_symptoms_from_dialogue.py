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

# 파일 저장용 lock
file_lock = threading.Lock()

def extract_symptoms_with_llm(dialogue_turns):
    """LLM을 사용하여 대화에서 환자 증상 추출"""
    # 대화를 텍스트로 변환
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
    """기존 결과 파일 로드"""
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_result(output_path, data):
    """결과를 파일에 저장 (thread-safe)"""
    with file_lock:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def save_individual_result(temp_dir, worker_id, data):
    """개별 워커의 결과를 임시 파일에 저장"""
    temp_file = os.path.join(temp_dir, f"worker_{worker_id}.json")
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def is_sample_processed(results, patient_id, symptom_key, cut_idx):
    """해당 샘플이 이미 처리되었는지 확인"""
    if patient_id not in results:
        return False
    if symptom_key not in results[patient_id]:
        return False
    dialogue_key = f"dialogue_{cut_idx}"
    return dialogue_key in results[patient_id][symptom_key]

def process_dialogue_sample(args):
    """단일 대화 샘플 처리 (멀티프로세싱용)"""
    patient_id, symptom_key, symptom_data, patient_ground_truth, cut_idx, end_idx, dialogue, temp_dir, worker_id, existing_results = args
    
    # 이미 처리된 샘플인지 확인
    if is_sample_processed(existing_results, patient_id, symptom_key, cut_idx):
        return None
    
    # 대화를 처음부터 해당 인덱스까지 자르기
    cut_dialogue = dialogue[:end_idx + 1]
    
    # LLM으로 증상 추출
    chiefcomplaint, present_illness_positive = extract_symptoms_with_llm(cut_dialogue)
    
    if not chiefcomplaint and not present_illness_positive:
        return None
    
    # 결과 데이터 구조 생성
    dialogue_key = f"dialogue_{cut_idx}"
    
    # 개별 결과 반환 (dict 수정 없이)
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
    """
    대화를 자르고 LLM으로 증상을 추출하여 증강된 데이터셋 생성
    
    Args:
        input_path: train_dialog.json 경로
        output_path: 출력 JSON 파일 경로
        augment_ratio: 대화 길이를 나누는 비율 (기본 5)
        max_workers: 병렬 처리 워커 수 (기본 4)
        max_samples: 처리할 최대 샘플 수 (None이면 제한 없음)
    """
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 기존 결과 로드
    print(f"Loading existing results from {output_path}...")
    existing_results = load_existing_results(output_path)
    
    # 임시 디렉토리 생성
    temp_dir = os.path.join(os.path.dirname(output_path), 'temp_results')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 처리할 작업 목록 생성
    tasks = []
    total_existing = 0
    total_tasks_added = 0
    worker_id = 0
    
    print(f"Preparing tasks for {len(data)} patients...")
    for patient_id, patient_data in data.items():
        if max_samples is not None and total_tasks_added >= max_samples:
            break
        # ground_truth와 subgraph 제외한 증상 키들
        symptom_keys = [k for k in patient_data.keys() if k not in ['ground_truth', 'subgraph']]
        
        for symptom_key in symptom_keys:
            symptom_data = patient_data[symptom_key]
            
            if 'dialogue' not in symptom_data:
                continue
            
            # recall 필터링
            if 'recall' in symptom_data and '4' in symptom_data['recall']:
                if symptom_data['recall']['4'] < 0.5:
                    continue
            
            dialogue = symptom_data['dialogue']
            dialogue_length = len(dialogue)
            
            if dialogue_length < 2:
                continue
            
            # 대화 인덱스 생성 (user만)
            user_indices = [i for i, turn in enumerate(dialogue) if turn['role'] == 'user']
            
            if len(user_indices) < 2:
                continue
            
            # 대화를 잘라서 여러 샘플 생성
            num_cuts = max(1, len(user_indices) // augment_ratio)
            sampled_indices = random.sample(user_indices, num_cuts)
            
            for cut_idx, end_idx in enumerate(sampled_indices):
                if max_samples is not None and total_tasks_added >= max_samples:
                    break
                    
                # 이미 처리된 샘플인지 확인
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
    
    # 멀티스레딩으로 병렬 처리
    print(f"Processing with {max_workers} workers...")
    processed_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_dialogue_sample, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting symptoms"):
            result = future.result()
            if result is not None:
                processed_results.append(result)
    
    # 기존 결과와 새 결과 병합
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
    
    # 최종 결과 저장
    print(f"Saving final results to {output_path}...")
    save_result(output_path, final_results)
    
    # 임시 디렉토리 삭제
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
