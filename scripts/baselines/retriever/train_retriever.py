"""Fine-tune an encoder (e.g. SapBERT) on (symptom query, disease document) pairs for the retriever HG."""
import sys
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import pandas as pd
from pathlib import Path
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))  # repository root
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: peft not installed. LoRA training will not be available.")
    print("Install with: pip install peft")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retrieve_utils import encoder

def get_model_save_path(model_name, base_dir='./trained_models', lr=None, epochs=None, use_lora=False, lora_r=None):
    """Derive the save path from the model name, lr, epochs and LoRA flag."""
    model_path = Path(model_name)
    model_short_name = model_path.name.lower()
    model_short_name = re.sub(r'[^a-z0-9_-]', '_', model_short_name)
    model_short_name = re.sub(r'_+', '_', model_short_name).strip('_')
    
    lr_str = f"_lr{lr:.0e}" if lr is not None else ""
    epochs_str = f"_ep{epochs}" if epochs is not None else ""
    lora_str = f"_lora_r{lora_r}" if use_lora and lora_r is not None else ""
    
    save_path = Path(base_dir) / f"trained_{model_short_name}{lr_str}{epochs_str}{lora_str}"
    return str(save_path)

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

class DiseaseDataset(Dataset):
    def __init__(self, data_path, kg_nodes_path, kg_edges_path, tokenizer):
        self.tokenizer = tokenizer
        
        self.kg = KnowledgeGraph(kg_nodes_path, kg_edges_path)
        
        disease_nodes = self.kg.nodes_df[self.kg.nodes_df['label'] == 'Disease']
        self.all_diseases = []
        self.all_disease_descriptions = []
        
        for _, row in disease_nodes.iterrows():
            disease_name = row['name']
            kg_symptoms = self.kg.get_disease_symptoms(disease_name)
            disease_description = f"{disease_name}: {', '.join(kg_symptoms)}" if kg_symptoms else disease_name
            
            self.all_diseases.append(disease_name)
            self.all_disease_descriptions.append(disease_description)
        
        print(f"Total diseases in KG: {len(self.all_diseases)}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        self.data = []
        
        for patient_id, patient_data in raw_data.items():
            ground_truth_diseases = patient_data.get('ground_truth', [])
            
            if not ground_truth_diseases:
                continue
            
            all_connected_nodes = set()
            for disease in ground_truth_diseases:
                connected_nodes = self.kg.get_disease_connected_nodes(disease)
                all_connected_nodes.update(connected_nodes)
            
            connected_nodes_text = "; ".join(sorted(all_connected_nodes))
            
            if not connected_nodes_text:
                continue
            
            for symptom_key, symptom_data in patient_data.items():
                if symptom_key in ['ground_truth', 'subgraph']:
                    continue
                
                for dialogue_key, dialogue_data in symptom_data.items():
                    if not isinstance(dialogue_data, dict):
                        continue
                    
                    mapped_symptoms = dialogue_data.get('mapped_symptoms', {})
                    chiefcomplaint = mapped_symptoms.get('chiefcomplaint', '')
                    present_illness_positive = mapped_symptoms.get('present_illness_positive', '')
                    
                    symptoms_parts = []
                    if chiefcomplaint:
                        symptoms_parts.append(chiefcomplaint)
                    if present_illness_positive:
                        symptoms_parts.append(present_illness_positive)
                    
                    symptoms_text = "; ".join(symptoms_parts)
                    
                    if symptoms_text:
                        self.data.append({
                            "symptoms": symptoms_text,
                            "positive_nodes": connected_nodes_text
                        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "symptoms": item['symptoms'],
            "positive_nodes": item['positive_nodes']
        }

    def collate_fn(self, batch):
        symptoms = [item['symptoms'] for item in batch]
        pos_nodes = [item['positive_nodes'] for item in batch]
        
        q_tokens = self.tokenizer(symptoms, return_tensors='pt', padding=True, truncation=True, max_length=256)
        
        return q_tokens, pos_nodes

def train_single_model(model_name, args, device, lr=None, epochs=None):
    """Train one model."""
    actual_lr = lr if lr is not None else args.lr
    actual_epochs = epochs if epochs is not None else args.epochs
    
    print(f"\n{'='*80}")
    print(f"Training model: {model_name}")
    print(f"LR: {actual_lr}, Epochs: {actual_epochs}")
    print(f"{'='*80}")
    
    if args.auto_save_path:
        save_path = get_model_save_path(model_name, args.save_dir, lr=actual_lr, epochs=actual_epochs, 
                                       use_lora=args.use_lora, lora_r=args.lora_r if args.use_lora else None)
    else:
        save_path = args.save_path
    
    print(f"Model will be saved to: {save_path}")
    
    print(f"Loading model: {model_name}")
    model = encoder(model_name)
    model.to(device)
    
    if args.use_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("peft library is required for LoRA training. Install with: pip install peft")
        
        print(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        
        try:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules="all-linear",
                bias="none",
            )
        except:
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", 
                               "query", "key", "value", "dense"],
                bias="none",
            )
        
        model.encoder_model = get_peft_model(model.encoder_model, peft_config)
        print("="*80)
        model.encoder_model.print_trainable_parameters()
        print("="*80)
    
    model.train()

    tokenizer = model.encoder_tokenizer
    
    print(f"Loading data from: {args.data_path}")
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        return
    
    print(f"Loading KG from: {args.kg_nodes_path}, {args.kg_edges_path}")
    if not os.path.exists(args.kg_nodes_path) or not os.path.exists(args.kg_edges_path):
        print("Error: KG files not found")
        return

    dataset = DiseaseDataset(args.data_path, args.kg_nodes_path, args.kg_edges_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    
    print(f"Total samples: {len(dataset)}")
    
    print("Preparing all KG nodes as candidates...")
    all_nodes_df = dataset.kg.nodes_df[dataset.kg.nodes_df['label'] != 'Disease']
    all_node_names = all_nodes_df['name'].tolist()
    
    print(f"Total candidate nodes: {len(all_node_names)}")
    
    print("Encoding all KG node embeddings...")
    all_node_tokens = tokenizer(
        all_node_names, 
        return_tensors='pt', 
        padding=True, 
        truncation=True, 
        max_length=256
    )
    all_node_ids = all_node_tokens['input_ids'].to(device)
    all_node_mask = all_node_tokens['attention_mask'].to(device)
    
    with torch.no_grad():
        all_node_embs = model.grad_encode(all_node_ids, all_node_mask)  # [num_nodes, hidden_dim]
    
    print(f"All node embeddings shape: {all_node_embs.shape}")

    effective_lr = actual_lr * 5 if args.use_lora else actual_lr
    if args.use_lora:
        print(f"LoRA mode: Adjusting LR from {actual_lr:.0e} to {effective_lr:.0e}")
    
    optimizer = AdamW(model.parameters(), lr=effective_lr)

    print("Start Training...")
    for epoch in range(actual_epochs):
        total_loss = 0
        total_acc = 0
        
        for step, (q_tokens, pos_nodes_list) in enumerate(dataloader):
            optimizer.zero_grad()
            
            q_input_ids = q_tokens['input_ids'].to(device)
            q_attention_mask = q_tokens['attention_mask'].to(device)
            query_embs = model.grad_encode(q_input_ids, q_attention_mask)  # [batch_size, hidden_dim]
            
            # query_embs: [batch_size, hidden_dim]
            # all_node_embs: [num_nodes, hidden_dim]
            similarity = torch.matmul(query_embs, all_node_embs.t())  # [batch_size, num_nodes]
            
            labels = []
            for pos_nodes_text in pos_nodes_list:
                pos_nodes = [n.strip() for n in pos_nodes_text.split(';')]
                label_idx = 0
                for node in pos_nodes:
                    if node in all_node_names:
                        label_idx = all_node_names.index(node)
                        break
                labels.append(label_idx)
            
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            
            loss = torch.nn.functional.cross_entropy(similarity, labels)
            
            _, predicted = torch.max(similarity, dim=1)
            acc = (predicted == labels).float().mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_acc += acc.item()
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        avg_acc = total_acc / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f} | Average Acc: {avg_acc:.4f}")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if args.use_lora:
        print("Saving LoRA adapter...")
        model.encoder_model.save_pretrained(save_path)
        with open(os.path.join(save_path, "base_model_info.json"), 'w') as f:
            json.dump({"base_model": model_name, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha}, f)
    else:
        model.save_pretrained(save_path)
    
    print(f"Model saved to {os.path.abspath(save_path)}")
    print(f"Training completed for {model_name}\n")
    
    return save_path

def train(args):
    """Train the requested models in sequence."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    models_to_train = []
    
    if args.config_file:
        print(f"Loading model configurations from: {args.config_file}")
        with open(args.config_file, 'r') as f:
            config = json.load(f)
            models_to_train = config.get('models', [])
        print(f"Found {len(models_to_train)} models in config file")
    elif args.models:
        models_to_train = [m.strip() for m in args.models.split(',')]
        print(f"Training {len(models_to_train)} models from command line")
    else:
        models_to_train = [args.model_name]
        print(f"Training single model: {args.model_name}")
    
    lr_list = args.lr_list if args.lr_list else [args.lr]
    epochs_list = args.epochs_list if args.epochs_list else [args.epochs]
    
    total_combinations = len(models_to_train) * len(lr_list) * len(epochs_list)
    if len(lr_list) > 1 or len(epochs_list) > 1:
        print(f"\nGrid Search Mode: {len(lr_list)} LRs × {len(epochs_list)} Epochs = {len(lr_list) * len(epochs_list)} combinations per model")
        print(f"Total training runs: {total_combinations}")
        print(f"LR values: {lr_list}")
        print(f"Epochs values: {epochs_list}")
    
    trained_models = []
    current_run = 0
    
    for model_name in models_to_train:
        for lr in lr_list:
            for epochs in epochs_list:
                current_run += 1
                print(f"\n[Run {current_run}/{total_combinations}]")
                try:
                    save_path = train_single_model(model_name, args, device, lr=lr, epochs=epochs)
                    trained_models.append((model_name, lr, epochs, save_path))
                except Exception as e:
                    print(f"Error training model {model_name} (lr={lr}, epochs={epochs}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print(f"\n{'='*80}")
    print("Training Summary")
    print(f"{'='*80}")
    print(f"Successfully trained {len(trained_models)} / {total_combinations} configurations:")
    for model_name, lr, epochs, save_path in trained_models:
        print(f"  - {model_name} (lr={lr:.0e}, epochs={epochs})")
        print(f"    Saved to: {save_path}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Disease Retriever with multiple embedding models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_retriever.py --model_name cambridgeltl/SapBERT-from-PubMedBERT-fulltext
  
  python train_retriever.py --models "cambridgeltl/SapBERT-from-PubMedBERT-fulltext,sentence-transformers/all-mpnet-base-v2"
  
  python train_retriever.py --config_file model_configs.json
  
  python train_retriever.py --models "model1,model2" --auto_save_path
        """)
    
    parser.add_argument('--model_name', type=str, 
                        default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext',
                        help='Pre-trained model name or path (for single model training)')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated list of model names to train sequentially')
    parser.add_argument('--config_file', type=str, default=None,
                        help='JSON config file with list of models to train')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU number to use')
    
    parser.add_argument('--data_path', type=str,
                        default=os.path.join(REPO, 'data/retriever/train_augmented_symptoms.json'),
                        help='Path to training data (train_augmented_symptoms.json)')
    parser.add_argument('--kg_nodes_path', type=str,
                        default=os.path.join(REPO, 'data/kg/original/nodes.csv'),
                        help='Path to KG nodes file')
    parser.add_argument('--kg_edges_path', type=str,
                        default=os.path.join(REPO, 'data/kg/original/edges.csv'),
                        help='Path to KG edges file')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training (default: 16, use 8 for large models like 4B)')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate (single value, ignored if --lr_list is provided)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (single value, ignored if --epochs_list is provided)')
    parser.add_argument('--lr_list', type=float, nargs='+', default=None,
                        help='List of learning rates for grid search (e.g., --lr_list 1e-5 2e-5 5e-5)')
    parser.add_argument('--epochs_list', type=int, nargs='+', default=None,
                        help='List of epochs for grid search (e.g., --epochs_list 5 10 15)')
    
    parser.add_argument('--use_lora', action='store_true',
                        help='Use LoRA for parameter-efficient fine-tuning')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank (default: 16, higher = more parameters)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha (default: 32, typically 2*r)')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout (default: 0.1)')
    
    parser.add_argument('--save_path', type=str, default=os.path.join(REPO, 'models/retriever'),
                        help='Path to save trained model (for single model, ignored if --auto_save_path is used)')
    parser.add_argument('--save_dir', type=str, default=os.path.join(REPO, 'models/retriever'),
                        help='Base directory for saving models when using --auto_save_path')
    parser.add_argument('--auto_save_path', action='store_true',
                        help='Automatically generate save path from model name')
    
    args = parser.parse_args()
    train(args)