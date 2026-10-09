"""LLM wrappers used by the pipeline.

get_openai_response / get_openrouter_response : API chat models (patient simulator, baselines, judges);
                                                 GPT models are called through Azure OpenAI Service
CausalLanguageModel                            : local HF causal LM (+ optional LoRA adapter) for the HV
EmbeddingModel                                 : SentenceTransformer wrapper (node matching)
DiseaseDetector                                : Hypothesis Generator = Qwen2.5-7B-Instruct + classification head
"""
import os
import json

import numpy as np
import requests
import torch
from sklearn.metrics.pairwise import cosine_similarity

from .paths import load_env

load_env()

_openai_client = None

# ----------------------------------------------------------------------------- Azure OpenAI
# All GPT calls go through Azure OpenAI Service (the paper used Azure for PhysioNet compliance).
#   AZURE_OPENAI_API_KEY        key of the Azure OpenAI resource
#   AZURE_OPENAI_ENDPOINT       https://<resource>.openai.azure.com/
#   AZURE_OPENAI_API_VERSION    e.g. 2024-10-21 (default)
#   AZURE_OPENAI_DEPLOYMENTS    optional JSON map {model name used in the code: Azure deployment name}
#                               e.g. {"gpt-4o-mini-2024-07-18": "gpt-4o-mini", "gpt-4.1-mini": "gpt-4-1-mini"}
#                               (unmapped names are passed to Azure as the deployment name)
# Set KGCQ_OPENAI_PROVIDER=openai to call api.openai.com (OPENAI_API_KEY) instead.
DEFAULT_AZURE_API_VERSION = "2024-10-21"


def openai_provider():
    return os.environ.get("KGCQ_OPENAI_PROVIDER", "azure").lower()


def resolve_deployment(model_name):
    """Map a model name used in the code to the Azure deployment name (identity if not mapped)."""
    mapping = os.environ.get("AZURE_OPENAI_DEPLOYMENTS", "").strip()
    if mapping:
        try:
            table = json.loads(mapping)
        except json.JSONDecodeError as e:
            raise ValueError(f"AZURE_OPENAI_DEPLOYMENTS must be a JSON object: {e}")
        if model_name in table:
            return table[model_name]
    return model_name


def make_openai_client(**kwargs):
    """Create a chat-completions client (AzureOpenAI by default). kwargs: timeout, max_retries, ..."""
    if openai_provider() == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), **kwargs)
    from openai import AzureOpenAI
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise RuntimeError("Azure OpenAI is not configured: set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT in .env "
                           "(or KGCQ_OPENAI_PROVIDER=openai with OPENAI_API_KEY).")
    return AzureOpenAI(api_key=api_key, azure_endpoint=endpoint,
                       api_version=os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION), **kwargs)


def openai_client(force_new=False, **kwargs):
    """Lazily-created shared client (so importing this module never requires credentials).
    force_new=True rebuilds it, e.g. per forked worker with its own timeout/max_retries."""
    global _openai_client
    if _openai_client is None or force_new:
        _openai_client = make_openai_client(**kwargs)
    return _openai_client


def get_openrouter_response(model_name, messages, temperature=0.0):
    """Non-OpenAI API models (Claude, Gemini, Llama, Qwen-72B baselines) via OpenRouter."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}", "Content-Type": "application/json"}
    payload = {"model": model_name, "messages": messages, "temperature": temperature}
    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']


def get_openai_response(model_name, messages, temperature=0.0, **kwargs):
    """GPT chat completion through Azure OpenAI. ``model_name`` is the model id used throughout the code
    (e.g. gpt-4o-mini-2024-07-18); it is translated to the Azure deployment name by resolve_deployment().
    Extra kwargs (e.g. response_format) are passed to chat.completions.create."""
    client = openai_client()
    deployment = resolve_deployment(model_name)
    if 'gpt-5' in model_name:  # gpt-5 family does not accept a temperature
        response = client.chat.completions.create(model=deployment, messages=messages, **kwargs)
    else:
        response = client.chat.completions.create(model=deployment, messages=messages, temperature=temperature, **kwargs)
    return response.choices[0].message.content.strip()


class CausalLanguageModel:
    def __init__(self, model_name, use_quant: bool = False):
        self.model_name = model_name
        self.use_quant = use_quant
        self._load_tokenizer()
        self._load_model()

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self):
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        bnb_config = None
        if self.use_quant:
            bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_quant_storage=torch.bfloat16,
                                            bnb_4bit_compute_dtype=torch.bfloat16,
                                            bnb_4bit_use_double_quant=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16,
            attn_implementation=("sdpa" if self.use_quant else None),
            quantization_config=bnb_config, device_map="auto",
            use_cache=(False if self.use_quant else None))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if self.use_quant:
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model)

    def load_model_for_inference(self, is_finetuned: bool = False, finetuned_model_dir=None):
        if is_finetuned:
            from peft import PeftModel
            assert finetuned_model_dir is not None, "finetuned_model_dir is required to load a LoRA adapter."
            self.model = PeftModel.from_pretrained(self.model, str(finetuned_model_dir))
            self.model = self.model.merge_and_unload()
            print("Loaded fine-tuned adapter:", finetuned_model_dir)
        self.model.eval()

    def _prompt(self, messages):
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, include_date=False)

    def get_response(self, messages: list, max_new_tokens: int = 512, temperature: float = 0) -> str:
        inputs = self.tokenizer(self._prompt(messages), return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                                          pad_token_id=self.tokenizer.pad_token_id,
                                          eos_token_id=self.tokenizer.eos_token_id)
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    def generate_w_prob(self, messages: list, max_new_tokens: int = 512, temperature: float = 0):
        """Generate and return (text, average token log-probability)."""
        inputs = self.tokenizer(self._prompt(messages), return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                                          pad_token_id=self.tokenizer.pad_token_id,
                                          eos_token_id=self.tokenizer.eos_token_id,
                                          return_dict_in_generate=True, output_scores=True)
        gen_ids = outputs.sequences[:, inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        log_probs = [torch.log_softmax(logits, dim=-1)[0, gen_ids[0, t]] for t, logits in enumerate(outputs.scores)]
        avg_logprob = torch.stack(log_probs).sum().item() / gen_ids.size(1)
        return response, avg_logprob

    def get_responses_batch(self, messages_list: list, max_new_tokens: int = 512, temperature: float = 0.01) -> list:
        prompts = [self._prompt(m) for m in messages_list]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=temperature,
                                          pad_token_id=self.tokenizer.pad_token_id,
                                          eos_token_id=self.tokenizer.eos_token_id)
        generated_tokens = outputs[:, inputs['input_ids'].shape[1]:]
        return [r.strip() for r in self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)]


class EmbeddingModel:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode(self, text_list: list, **kwargs):
        return self.model.encode(text_list, **kwargs)

    @staticmethod
    def get_similar_indices(query_embedding, candidates_embedding, threshold=0, top_k=10):
        similarities = cosine_similarity(query_embedding, candidates_embedding).flatten()
        indices_above_threshold = np.where(similarities >= threshold)[0]
        if len(indices_above_threshold) == 0:
            return []
        scores_above_threshold = similarities[indices_above_threshold]
        sorted_order = np.argsort(scores_above_threshold)[::-1]
        top_indices = indices_above_threshold[sorted_order][:top_k]
        top_scores = scores_above_threshold[sorted_order][:top_k]
        return list(zip(top_indices, top_scores))


class DiseaseDetector:
    """Hypothesis Generator (HG): multi-label disease classifier over the dialogue history.

    P(disease | H_t) = sigmoid(W . LLM_HG(H_t) + b), one logit per disease node (paper Eq. 1).

    ``finetuned_model_dir`` may be either
      * a merged AutoModelForSequenceClassification checkpoint (the released HG), or
      * a LoRA adapter directory (contains adapter_config.json); then ``base_model`` is loaded first.
    ``disease_names`` fixes the label order (index i -> disease_names[i]); it must match training.
    """

    def __init__(self, finetuned_model_dir, disease_names, base_model=None):
        self.finetuned_model_dir = str(finetuned_model_dir)
        self.base_model = base_model
        self.disease_names = list(disease_names)
        self.id2label = {i: name for i, name in enumerate(self.disease_names)}
        self.label2id = {name: i for i, name in enumerate(self.disease_names)}
        self.num_labels = len(self.disease_names)
        self._load_tokenizer()
        self._load_model()

    @property
    def is_adapter(self):
        return os.path.exists(os.path.join(self.finetuned_model_dir, "adapter_config.json"))

    def _load_tokenizer(self):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model_dir, padding_side="left")

    def _load_model(self):
        from transformers import AutoModelForSequenceClassification
        kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto", num_labels=self.num_labels,
                      id2label=self.id2label, label2id=self.label2id, problem_type="multi_label_classification")
        if self.is_adapter:
            from peft import PeftModel
            assert self.base_model is not None, "base_model is required when loading a LoRA adapter HG."
            base = AutoModelForSequenceClassification.from_pretrained(self.base_model, **kwargs)
            base.resize_token_embeddings(len(self.tokenizer))
            base.config.pad_token_id = self.tokenizer.pad_token_id
            self.model = PeftModel.from_pretrained(base, self.finetuned_model_dir)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.finetuned_model_dir, **kwargs)
        self.model.eval()

    @staticmethod
    def create_prompt(messages):
        """Same template as HG training (prompts/hg_classifier_template.txt)."""
        dialogue_turns = []
        for turn in messages:
            if turn['role'] == "user":
                dialogue_turns.append(f"Patient: {turn['content']}")
            elif turn['role'] == "assistant":
                dialogue_turns.append(f"Doctor: {turn['content']}")
        dialogue_text = "\n".join(dialogue_turns)
        return f"Conversation:\n{dialogue_text}\n\nBased on the conversation, what diseases does the patient have?"

    def predict_all_probs(self, messages):
        text = self.create_prompt(messages)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits.float().cpu().numpy()[0]
        return 1 / (1 + np.exp(-logits))

    def rank_diseases(self, messages, gold_disease_list=None, top_k=10):
        """Return (result, ranked). result has ranked_list (top-k), entropy, all_probs, and gold_prob/gold_rank
        when ``gold_disease_list`` is given (diagnostics only; never used for decisions)."""
        probs = self.predict_all_probs(messages)
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            normalized = probs / probs_sum
            entropy = -np.sum(normalized * np.log(normalized + 1e-9))
        else:
            entropy = 0.0
        ranked = sorted([(self.id2label[i], float(probs[i])) for i in range(self.num_labels)],
                        key=lambda x: x[1], reverse=True)
        result = {"ranked_list": ranked[:top_k], "entropy": float(entropy), "all_probs": dict(ranked)}
        if gold_disease_list is not None:
            gold = [g.strip() for g in gold_disease_list]
            gold_probs = [prob for name, prob in ranked if name in gold]
            gold_ranks = [idx + 1 for idx, (name, _) in enumerate(ranked) if name in gold]
            result["gold_prob"] = sum(gold_probs) / len(gold_probs) if gold_probs else 0.0
            result["gold_rank"] = sum(gold_ranks) / len(gold_ranks) if gold_ranks else float('inf')
        return result, ranked


def disease_names_from_nodes(nodes_csv):
    """Label order of the HG = order of Disease rows in nodes.csv."""
    import pandas as pd
    df = pd.read_csv(nodes_csv)
    return [row['name'] for _, row in df.iterrows() if row['label'] == 'Disease']


def load_label_config(model_dir):
    """If the HG directory stores label_config.json (adapter HGs), return its disease_names."""
    p = os.path.join(str(model_dir), "label_config.json")
    if os.path.exists(p):
        return json.load(open(p))["disease_names"]
    return None
