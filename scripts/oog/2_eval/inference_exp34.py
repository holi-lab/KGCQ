"""End-to-end out-of-graph inference: HG adapter -> 3-hop subgraph (tau) -> HV adapter.
In-graph (ID) and out-of-graph (OOD) profiles are scored separately.

  python inference_exp34.py --exp exp5 --hg_ratio 30 --hv_ratio 25 --gpu 0 --id_set balanced --ood_set clean243 \
      --hv_dir ../../models/oog/exp5_hv_r25_symcentric_Qwen2.5-7B --tau 0.005 --tag_suffix _sweep
  python inference_exp34.py --exp exp6 --hg_ratio 35 --hv_ratio 30 --gpu 0 --kg augmented ... --tag_suffix _v3a
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import os, sys
# CUDA_VISIBLE_DEVICES must be set before torch is imported
_gpu = sys.argv[sys.argv.index('--gpu') + 1] if '--gpu' in sys.argv else '0'
os.environ['CUDA_VISIBLE_DEVICES'] = _gpu
os.environ.setdefault('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json, argparse, random

import _config as C
import _subgraph as SG
from dotenv import load_dotenv
load_dotenv(C.ENV_FILE)

import pandas as pd
import numpy as np
import torch
from kgcq.utils import file_to_string
from kgcq.models import DiseaseDetector, CausalLanguageModel, get_openai_response, EmbeddingModel
from kgcq.graph import DiagnosticKnowledgeGraph
from kgcq.simulator import generate_doctor_response
from kgcq import simulator as our_sim
from transformers import AutoModelForSequenceClassification
from peft import PeftModel

OTHER_NOTE = ("Note: The actual disease may not be among the diseases listed above. "
              "In that case, it should be classified as 'Other'.")
ID_SETS = {'balanced': 'balanced_rating_w_persona.json', 'valid': 'id_valid_304.json'}        # 275 test / 304 valid
OOD_SETS = {'clean243': 'ood_test_clean243.json', 'valid128': 'ood_valid_128.json'}             # 243 test / 128 valid


class HGAdapterDetector(DiseaseDetector):
    """HG = base classification model + LoRA adapter (as saved by train_hg_exp34.py)."""
    def __init__(self, base_id, adapter_dir, disease_names):
        self.base_id = base_id
        self.adapter_dir = adapter_dir
        self.finetuned_model_dir = adapter_dir
        self.disease_names = disease_names
        self.id2label = {i: n for i, n in enumerate(disease_names)}
        self.label2id = {n: i for i, n in enumerate(disease_names)}
        self.num_labels = len(disease_names)
        self._load_tokenizer()
        self._load_model()

    def _load_model(self):
        base = AutoModelForSequenceClassification.from_pretrained(
            self.base_id, torch_dtype=torch.bfloat16, device_map="auto",
            num_labels=self.num_labels, id2label=self.id2label, label2id=self.label2id,
            problem_type="multi_label_classification")
        base.resize_token_embeddings(len(self.tokenizer))
        base.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = PeftModel.from_pretrained(base, self.adapter_dir)
        self.model.eval()


def recall_at_k(gold, pred, k):
    if not pred:
        return 0.0
    g = set(gold)
    return len(g & set(pred[:k])) / len(g) if g else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, choices=['exp5', 'exp6'], help='exp5 = "Other" node, exp6 = KG augmentation')
    ap.add_argument('--hg_ratio', type=int, required=True)
    ap.add_argument('--hv_ratio', type=int, required=True)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--top_k', type=int, default=2)
    ap.add_argument('--tau', type=float, default=0.005)
    ap.add_argument('--max_turns', type=int, default=50)
    ap.add_argument('--id_set', choices=list(ID_SETS), default='balanced')
    ap.add_argument('--ood_set', choices=list(OOD_SETS), default='clean243')
    ap.add_argument('--id_only', action='store_true')
    ap.add_argument('--ood_only', action='store_true')
    ap.add_argument('--hv_dir', default=None, help='HV adapter dir (default models/oog/{exp}_hv_r{ratio}_Qwen2.5-7B)')
    ap.add_argument('--kg', default=None, help='graph name (default: original for exp5, augmented for exp6)')
    ap.add_argument('--subgraph_format', choices=['disease', 'symptom'], default='symptom')
    ap.add_argument('--subgraph_method', choices=['paper3hop'], default='paper3hop')
    ap.add_argument('--tag_suffix', default='')
    ap.add_argument('--no_save_dialogue', action='store_true')
    args = ap.parse_args()

    EXP = args.exp
    kg_name = args.kg or ('original' if EXP == 'exp5' else 'augmented')
    nodes = pd.read_csv(C.KG[kg_name]['nodes'])
    edges = pd.read_csv(C.KG[kg_name]['edges'])
    disease_names = [r['name'] for _, r in nodes.iterrows() if r['label'] == 'Disease']
    if EXP == 'exp5':
        disease_names = disease_names + ['Other']
    print(f"[{EXP}] KG={kg_name}, HG labels {len(disease_names)}", flush=True)

    kg = DiagnosticKnowledgeGraph(nodes, edges, embedding_model=EmbeddingModel(C.EMB_MODEL))
    meta = {
        'name2id': {str(r['name']).strip().lower(): r['id'] for _, r in nodes.iterrows() if r['label'] == 'Disease'},
        'id2name': dict(zip(nodes['id'], nodes['name'])),
        'id2label': dict(zip(nodes['id'], nodes['label'])),
    }

    hg_dir = f'{P.MODEL_DIR}/{EXP}_hg_r{args.hg_ratio:02d}_Qwen2.5-7B'
    hv_dir = args.hv_dir or f'{P.MODEL_DIR}/{EXP}_hv_r{args.hv_ratio:02d}_Qwen2.5-7B'
    print(f"HG: {hg_dir}\nHV: {hv_dir}", flush=True)
    hg = HGAdapterDetector(C.BASE_MODEL, hg_dir, disease_names)
    hv = CausalLanguageModel(C.BASE_MODEL, use_quant=False)
    hv.load_model_for_inference(is_finetuned=True, finetuned_model_dir=hv_dir)

    DOCTOR_PROMPT = file_to_string(f'{P.PROMPT_DIR}/hv_doctor.txt')
    PATIENT_PROMPT = file_to_string(C.PATIENT_PROMPT_PATH)

    def gen_p(messages, temperature=0.0):
        return get_openai_response(C.PATIENT_MODEL, messages, temperature=temperature)

    def hv_gen(messages, temperature=0.0):
        return hv.get_response(messages, temperature=temperature)

    # evaluation profiles: (hadm, profile, gold, kind)
    targets = []
    if not args.ood_only:
        prof = json.load(open(f'{P.PROFILE_DIR}/{ID_SETS[args.id_set]}'))
        for h, p in prof.items():
            gold = [str(d).strip().lower() for d in (p.get('disease_mapped') or [])]
            if gold:
                targets.append((str(h), dict(p), gold, 'ID'))
    if not args.id_only:
        prof = json.load(open(f'{P.PROFILE_DIR}/{OOD_SETS[args.ood_set]}'))
        for h, p in prof.items():
            real = [str(d).strip().lower() for d in (p.get('disease_mapped') or [])]
            gold = ['other'] if EXP == 'exp5' else real
            if gold:
                targets.append((str(h), dict(p), gold, 'OOD'))
    nID = sum(1 for t in targets if t[3] == 'ID')
    nOOD = sum(1 for t in targets if t[3] == 'OOD')
    print(f"targets: ID {nID} + OOD {nOOD} = {len(targets)}  top_k={args.top_k}", flush=True)

    def build_inference_subgraph(top_diseases, all_probs):
        seeds = [d for d in top_diseases if d.lower() != 'other']
        has_other = any(d.lower() == 'other' for d in top_diseases)
        if not seeds:
            return [OTHER_NOTE] if has_other else []
        lines, _ = SG.build_subgraph_3hop(kg, meta, seeds, all_probs or {}, tau=args.tau, fmt=args.subgraph_format)
        if has_other:
            lines = lines + [OTHER_NOTE]
        return lines

    def detect_subgraph(pure_messages):
        res = hg.rank_diseases(messages=pure_messages[1:], gold_disease_list=None, top_k=args.top_k)
        res = res if isinstance(res, dict) else res[0]
        top = [d for d, _ in res['ranked_list']]
        return build_inference_subgraph(top, res.get('all_probs')), top

    agg = {'ID': {1: [], 2: [], 3: [], 4: []}, 'OOD': {1: [], 2: [], 3: [], 4: []}}
    results = []
    for idx, (h, prof, gt, kind) in enumerate(targets):
        prof['disease'] = gt
        pure_messages = [{"role": "assistant", "content": "Hello, how can I help you today?"}]
        lr = prof.get('likelihood_rating') or 3
        max_turn = random.randint(max(1, 10 - lr), 10 - lr + 3)
        try:
            resp = our_sim.generate_patient_response_w_persona(PATIENT_PROMPT, pure_messages, gen_p, prof)
            pure_messages.append({"role": "user", "content": resp})
        except Exception as e:
            results.append({"hadm": h, "kind": kind, "error": f"patient:{str(e)[:50]}"})
            continue
        subgraph_lines, top = detect_subgraph(pure_messages)
        final_diagnosis, recall = [], {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
        curr_turn = 0
        try:
            while curr_turn <= args.max_turns:
                inputs = {"subgraph_text": "\n".join(subgraph_lines), "max_turn": max_turn, "curr_turn": curr_turn + 1}
                action, full_response, utterance = generate_doctor_response(DOCTOR_PROMPT, pure_messages, hv_gen, inputs)
                pure_messages.append({"role": "assistant", "content": utterance})
                if action == 'diagnosis':
                    for u in utterance.lower().split(','):
                        u = u.strip()
                        if u in ('other', "'other'", '"other"'):
                            final_diagnosis.append('other')
                        else:
                            name, sim = kg.find_matching_disease(u)
                            final_diagnosis.append(name.lower() if sim > 0.9 else "")
                    for k in (1, 2, 3, 4):
                        recall[k] = recall_at_k(gt, final_diagnosis, k)
                    break
                resp = our_sim.generate_patient_response_w_persona(PATIENT_PROMPT, pure_messages, gen_p, prof)
                pure_messages.append({"role": "user", "content": resp})
                subgraph_lines, top = detect_subgraph(pure_messages)
                curr_turn += 1
        except Exception as e:
            results.append({"hadm": h, "kind": kind, "error": f"loop:{str(e)[:50]}"})
            continue
        for k in (1, 2, 3, 4):
            agg[kind][k].append(recall[k])
        row = {"hadm": h, "kind": kind, "gt": gt, "final_diagnosis": final_diagnosis,
               "recall": recall, "turns": curr_turn, "top_k": top}
        if not args.no_save_dialogue:
            row["dialogue"] = pure_messages
            row["subgraph_lines"] = subgraph_lines
            row["n_subgraph_lines"] = len(subgraph_lines)
        results.append(row)
        if (idx + 1) % 10 == 0:
            r4i = np.mean(agg['ID'][4]) if agg['ID'][4] else 0
            r4o = np.mean(agg['OOD'][4]) if agg['OOD'][4] else 0
            print(f"  [{idx+1}/{len(targets)}] ID@4={r4i:.3f} OOD@4={r4o:.3f}", flush=True)

    metrics = {"exp": EXP, "hg_ratio": args.hg_ratio, "hv_ratio": args.hv_ratio, "top_k": args.top_k,
               "n_ID": len(agg['ID'][4]), "n_OOD": len(agg['OOD'][4])}
    for kind in ('ID', 'OOD'):
        for k in (1, 2, 3, 4):
            v = agg[kind][k]
            metrics[f"{kind}_recall@{k}"] = round(float(np.mean(v)), 4) if v else None
    # output tag: {exp}_hg{HG}_hv{HV}_{bal|validid}[_validood|_oodclean]{suffix}
    tag = f'{EXP}_hg{args.hg_ratio}_hv{args.hv_ratio}'
    tag += '_bal' if args.id_set == 'balanced' else '_validid'
    if args.ood_only:
        tag += '_validood' if args.ood_set == 'valid128' else '_oodclean'
    tag += args.tag_suffix
    metrics['id_set'] = args.id_set
    metrics['hv_dir'] = hv_dir
    out_dir = str(P.PIPELINE_RESULTS_DIR)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(metrics, open(f'{out_dir}/{tag}_metrics.json', 'w'), ensure_ascii=False, indent=2)
    json.dump(results, open(f'{out_dir}/{tag}_results.json', 'w'), ensure_ascii=False)
    print("\n===== Pipeline Metrics =====", flush=True)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nSaved -> {out_dir}/{tag}_*", flush=True)


if __name__ == '__main__':
    main()
