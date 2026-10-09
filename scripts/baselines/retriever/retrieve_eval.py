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
    """KG에서 질병과 연결된 증상들을 가져오는 클래스"""
    def __init__(self, nodes_path, edges_path):
        self.nodes_df = pd.read_csv(nodes_path)
        self.edges_df = pd.read_csv(edges_path)
        
        # disease name -> node id 매핑
        self.disease_to_id = {}
        for _, row in self.nodes_df[self.nodes_df['label'] == 'Disease'].iterrows():
            self.disease_to_id[row['name'].lower()] = row['id']
        
        # node id -> name 매핑
        self.id_to_name = dict(zip(self.nodes_df['id'], self.nodes_df['name']))
        
    def get_disease_symptoms(self, disease_name):
        """질병과 연결된 모든 증상 노드를 반환"""
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
        symptom_names = [self.id_to_name.get(node_id, '') for node_id in connected_nodes]
        return [s for s in symptom_names if s]
    
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

def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. 모델 로드
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
    
    # 2. KG 로드 및 모든 disease 노드에 대해 연결된 노드들의 텍스트 합 생성
    print(f"Loading KG from: {args.kg_nodes_path}")
    kg = KnowledgeGraph(args.kg_nodes_path, args.kg_edges_path)
    
    disease_nodes = kg.nodes_df[kg.nodes_df['label'] == 'Disease']
    all_diseases = []
    all_disease_descriptions = []
    
    for _, row in disease_nodes.iterrows():
        disease_name = row['name']
        # 질병에 연결된 모든 노드들 가져오기
        connected_nodes = kg.get_disease_connected_nodes(disease_name)
        # 연결된 노드들을 텍스트로 합치기
        disease_description = "; ".join(connected_nodes) if connected_nodes else disease_name
        
        all_diseases.append(disease_name.lower())
        all_disease_descriptions.append(disease_description)
    
    print(f"Total diseases in KG: {len(all_diseases)}")
    
    # 모든 disease description 임베딩
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
    
    # 3. Test 데이터 로드 (test_augmented_symptoms.json)
    print(f"Loading test data from: {args.test_path}")
    with open(args.test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 4. 평가
    print("Evaluating...")
    recalls = {1: [], 2: [], 3: [], 4: []}
    total_samples = 0
    
    for patient_id, patient_data in test_data.items():
        # mapped_symptoms에서 증상 추출
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
        
        # 정답 질병 (ground_truth)
        ground_truth_diseases = patient_data.get('ground_truth', [])
        if not ground_truth_diseases:
            continue
        
        # 정답 질병을 소문자로 변환
        ground_truth = set([d.lower() for d in ground_truth_diseases])
        
        # 증상 임베딩
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
        
        # 모든 disease와의 유사도 계산
        similarity = torch.matmul(symptom_emb, all_disease_embs.t()).squeeze(0)
        
        # Top-k 추출
        top_k_values, top_k_indices = torch.topk(similarity, k=max(recalls.keys()), largest=True)
        
        # Recall 계산
        for k in recalls.keys():
            top_k_diseases = [all_diseases[idx] for idx in top_k_indices[:k].cpu().tolist()]
            
            # 정답이 top-k 안에 있는지 확인
            hit = any(gt in top_k_diseases for gt in ground_truth)
            recalls[k].append(1 if hit else 0)
        
        total_samples += 1
        
        if total_samples % 10 == 0:
            print(f"Processed {total_samples} samples...")
    
    # 5. 결과 반환
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
    
    # 모델 설정
    parser.add_argument('--model_path', type=str,
                        default=None,
                        help='Path to trained model (if None, evaluates all models in train/)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU number to use')
    
    # 데이터 설정
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
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 타임스탬프 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 평가할 모델 리스트 생성
    if args.model_path:
        model_paths = [args.model_path]
    else:
        # train/trained_models/ 폴더에서만 모든 trained_* 디렉토리 찾기
        model_paths = glob('./train/trained_models/trained_*')
        model_paths = [p for p in model_paths if os.path.isdir(p)]
        print(f"Found {len(model_paths)} models in train/trained_models/")
    
    # 결과 저장할 파일
    output_file = os.path.join(args.output_dir, f"evaluation_results_{timestamp}.txt")
    
    # 모든 모델 평가
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
            
            # 평가 실행
            args.model_path = model_path
            try:
                results = evaluate(args)
                
                # 콘솔 출력
                print(f"\nResults on {results['total_samples']} samples:")
                for k in sorted(results['recalls'].keys()):
                    recall_k = results['recalls'][k]
                    print(f"Recall@{k}: {recall_k:.2f}%")
                
                # 파일 출력
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
        
        # 요약 출력
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
