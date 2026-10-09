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

# 파일 저장용 lock
file_lock = threading.Lock()

class KnowledgeGraph:
    """KG에서 질병과 연결된 노드들을 가져오는 클래스"""
    def __init__(self, nodes_path, edges_path):
        self.nodes_df = pd.read_csv(nodes_path)
        self.edges_df = pd.read_csv(edges_path)
        
        # disease name -> node id 매핑
        self.disease_to_id = {}
        for _, row in self.nodes_df[self.nodes_df['label'] == 'Disease'].iterrows():
            self.disease_to_id[row['name'].lower()] = row['id']
        
        # node id -> name 매핑
        self.id_to_name = dict(zip(self.nodes_df['id'], self.nodes_df['name']))
        
    def get_disease_connected_nodes(self, disease_name):
        """질병과 KG상에서 연결된 모든 노드들을 반환"""
        disease_name_lower = disease_name.lower()
        
        if disease_name_lower not in self.disease_to_id:
            return []
        
        disease_id = self.disease_to_id[disease_name_lower]
        
        # 해당 질병에 연결된 모든 노드 찾기 (caused_by 관계)
        connected_nodes = self.edges_df[
            (self.edges_df['end'] == disease_id) & 
            (self.edges_df['type'] == 'caused_by')
        ]['start'].tolist()
        
        # 노드 이름으로 변환
        node_names = [self.id_to_name.get(node_id, '') for node_id in connected_nodes]
        return [n for n in node_names if n]

def extract_symptoms_with_llm(conversation_text):
    """LLM을 사용하여 대화에서 환자 증상 추출"""
    
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

def process_test_sample(args):
    """단일 테스트 샘플 처리"""
    sample, kg, existing_results = args
    
    patient_id = sample['patient_id']
    
    # 이미 처리된 샘플인지 확인
    if patient_id in existing_results:
        return None
    
    # prompt에서 대화 부분 추출
    prompt = sample['prompt']
    # "Conversation:\n" 이후, "\n\nBased on" 이전까지가 대화
    conversation_start = prompt.find("Conversation:\n")
    conversation_end = prompt.find("\n\nBased on")
    
    if conversation_start == -1 or conversation_end == -1:
        return None
    
    conversation_text = prompt[conversation_start + len("Conversation:\n"):conversation_end]
    
    # LLM으로 증상 추출
    chiefcomplaint, present_illness_positive = extract_symptoms_with_llm(conversation_text)
    
    if not chiefcomplaint and not present_illness_positive:
        return None
    
    # ground_truth 질병들
    ground_truth_diseases = sample.get('true_labels', [])
    
    # ground_truth 질병들에 연결된 KG 노드들 수집
    ground_truth_symptoms = set()
    for disease in ground_truth_diseases:
        connected_nodes = kg.get_disease_connected_nodes(disease)
        ground_truth_symptoms.update(connected_nodes)
    
    # 결과 데이터 반환 (dict 수정하지 않음)
    return {
        "patient_id": patient_id,
        "ground_truth": ground_truth_diseases,
        "chiefcomplaint": chiefcomplaint,
        "present_illness_positive": present_illness_positive,
        "ground_truth_symptoms": sorted(list(ground_truth_symptoms))
    }

def create_test_dataset(input_path, output_path, kg_nodes_path, kg_edges_path, max_workers=10, max_samples=None):
    """
    test_predictions.json에서 대화를 추출하고 LLM으로 증상을 뽑아 테스트 데이터셋 생성
    
    Args:
        input_path: test_predictions.json 경로
        output_path: 출력 JSON 파일 경로
        kg_nodes_path: KG nodes.csv 경로
        kg_edges_path: KG edges.csv 경로
        max_workers: 병렬 처리 워커 수
        max_samples: 처리할 최대 샘플 수 (None이면 제한 없음)
    """
    print(f"Loading data from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loading KG from {kg_nodes_path}, {kg_edges_path}...")
    kg = KnowledgeGraph(kg_nodes_path, kg_edges_path)
    
    # 기존 결과 로드
    print(f"Loading existing results from {output_path}...")
    existing_results = load_existing_results(output_path)
    
    # 처리할 작업 목록 생성
    tasks = []
    total_existing = len(existing_results)
    
    print(f"Preparing tasks for {len(data)} samples...")
    for sample in data:
        patient_id = sample.get('patient_id')
        
        if not patient_id:
            continue
        
        # 이미 처리된 샘플 건너뛰기
        if patient_id in existing_results:
            continue
        
        tasks.append((sample, kg, existing_results))
        
        # max_samples 제한
        if max_samples is not None and len(tasks) >= max_samples:
            break
    
    print(f"Already processed: {total_existing} samples")
    print(f"Remaining tasks to process: {len(tasks)}")
    
    if len(tasks) == 0:
        print("All samples already processed!")
        return
    
    # 멀티스레딩으로 병렬 처리
    print(f"Processing with {max_workers} workers...")
    processed_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_test_sample, task) for task in tasks]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting symptoms"):
            result = future.result()
            if result is not None:
                processed_results.append(result)
    
    # 기존 결과와 새 결과 병합
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
    
    # 최종 결과 저장
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
                        default=os.path.join(REPO, 'data/kg/paper/nodes.csv'),
                        help='Path to KG nodes file')
    parser.add_argument('--kg_edges_path', type=str,
                        default=os.path.join(REPO, 'data/kg/paper/edges.csv'),
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
