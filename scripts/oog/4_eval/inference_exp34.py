"""4_eval/inference_exp34.py — exp3/exp4 full 파이프라인 eval (HG+HV 인퍼런스).

inference.py 의 exp3/exp4 변형:
  - HG = adapter-only → base+adapter 로딩(HGAdapterDetector)
  - 라벨/KG: exp3 = paper(338+Other=339) / exp4 = augmented_v3(528)
  - 프로필: ID = paper split_1210/clf/test_w_persona.json (172) / OOD = exp2_id (exp2 test strict-OOD 74)
  - gold: ID = disease_mapped(paper) / OOD: exp3='Other', exp4=disease_mapped(aug rescued)
  - 추론 루프(환자sim → HG top_k → 우리 subgraph(+exp3 OOD시 Other note) → HV 대화 → 진단) = inference.py 그대로
  - recall@1~4 를 ID/OOD 분리 집계

사용: python inference_exp34.py --exp exp3 --hg_ratio 10 --hv_ratio 10 --gpu 1 [--n N]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import os, sys
# ★ CUDA_VISIBLE_DEVICES 는 torch import 전에 설정해야 함 (안 그러면 CPU 로딩)
_gpu = sys.argv[sys.argv.index('--gpu') + 1] if '--gpu' in sys.argv else '0'
os.environ['CUDA_VISIBLE_DEVICES'] = _gpu
os.environ.setdefault('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import json, argparse, random

HERE = os.path.dirname(os.path.abspath(__file__))
import _config as C      # scripts/oog/_config.py
import _subgraph as SG   # scripts/oog/_subgraph.py
OTHER_NOTE = ("Note: The actual disease may not be among the diseases listed above. "
              "In that case, it should be classified as 'Other'.")

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

PAPER_PROFILE = f'{P.PROFILE_DIR}/clf_test_w_persona.json'   # eval ID test 172 프로필 (로컬 복사본)


class HGAdapterDetector(DiseaseDetector):
    """adapter-only HG: base AutoModelForSequenceClassification + PeftModel adapter."""
    def __init__(self, base_id, adapter_dir, disease_names):
        self.base_id = base_id
        self.adapter_dir = adapter_dir
        self.finetuned_model_dir = adapter_dir       # tokenizer 는 adapter dir 에 저장됨
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


def dis_set(csv):
    n = pd.read_csv(csv)
    return set(str(r['name']).strip().lower() for _, r in n.iterrows() if r['label'] == 'Disease')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, choices=['exp3', 'exp4', 'exp5', 'exp6'])
    ap.add_argument('--hg_ratio', type=int, default=10)
    ap.add_argument('--hv_ratio', type=int, default=10)
    ap.add_argument('--gpu', default='0')
    ap.add_argument('--top_k', type=int, default=2)
    ap.add_argument('--max_turns', type=int, default=50)
    ap.add_argument('--n', type=int, default=0, help='ID/OOD 각 N명 제한(0=전체)')
    ap.add_argument('--id_set', choices=['clf', 'balanced', 'valid'], default='clf',
                    help="ID 평가셋: clf=clf/test(172) / balanced=balanced_rating(275,test) / valid=sft/valid(304,검증)")
    ap.add_argument('--id_only', action='store_true',
                    help='OOD 건너뛰고 ID만 평가 (balanced_rating은 전부 ID — OOD는 기존 결과 재사용)')
    ap.add_argument('--ood_only', action='store_true',
                    help='ID 건너뛰고 OOD만 평가 — OOD ratio sweep 채울 때')
    ap.add_argument('--ood_set', choices=['strict74', 'exp275', 'clean243', 'valid128'], default='strict74',
                    help='OOD 평가셋: strict74(74) / exp275(275,test) / clean243(243,누수제거 test) / valid128(128,검증)')
    ap.add_argument('--save_dialogue', action='store_true', help='(구) 대화/subgraph 저장 — 이제 기본 저장이라 무의미')
    ap.add_argument('--no_save_dialogue', action='store_true', help='대화/subgraph 저장 안 함 (기본=저장; 비싼 인퍼런스 대화 보존)')
    ap.add_argument('--hv_dir', default=None, help='HV adapter 경로 직접 지정(예: paper-faithful HV). 미지정시 model/{exp}_hv_r{R}')
    ap.add_argument('--subgraph_format', choices=['disease', 'symptom'], default='disease',
                    help='subgraph linearization: disease(기본,질병-중심) / symptom(paper 인퍼런스 증상-중심 reverse)')
    ap.add_argument('--subgraph_method', choices=['ratio', 'paper3hop'], default='ratio',
                    help='질병 선택: ratio(기본,우리 build_subgraph) / paper3hop(paper extract_subgraph_3hop, 3-hop+tau)')
    ap.add_argument('--tau', type=float, default=0.005, help='paper3hop 의 prob_threshold(tau)')
    ap.add_argument('--tag_suffix', default='', help='출력 태그 접미사(예: _paperhv) — 기존 결과 보존용')
    ap.add_argument('--kg', default=None, help='KG 오버라이드(예: augmented_v3); 미지정시 exp 기본(paper/augmented)')
    args = ap.parse_args()
    # CUDA_VISIBLE_DEVICES 는 파일 상단에서 이미 설정됨 (torch import 전)

    EXP = args.exp
    kg_name = args.kg if args.kg else ('paper' if EXP in ('exp3', 'exp5') else 'augmented_v3')
    sg_cfg = C.SUBGRAPH_CFG_BY_KG.get(kg_name, C.SUBGRAPH_CFG)
    PAPER = dis_set(C.KG['paper']['nodes'])
    AUG = dis_set(C.KG['augmented_v3']['nodes'])

    nodes = pd.read_csv(C.KG[kg_name]['nodes'])
    edges = pd.read_csv(C.KG[kg_name]['edges'])
    disease_names = [r['name'] for _, r in nodes.iterrows() if r['label'] == 'Disease']
    if EXP in ('exp3', 'exp5'):
        disease_names = disease_names + ['Other']
    print(f"[{EXP}] KG={kg_name}, HG 라벨 {len(disease_names)}", flush=True)

    emb = EmbeddingModel(C.EMB_MODEL)
    kg = DiagnosticKnowledgeGraph(nodes, edges, embedding_model=emb)
    meta = {
        'name2id': {str(r['name']).strip().lower(): r['id'] for _, r in nodes.iterrows() if r['label'] == 'Disease'},
        'id2name': dict(zip(nodes['id'], nodes['name'])),
        'id2label': dict(zip(nodes['id'], nodes['label'])),
        'disease_ids': [r['id'] for _, r in nodes.iterrows() if r['label'] == 'Disease'],
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

    # --- 평가 환자: ID(paper test) + OOD(exp2 test strict) ---
    def aood(dm):
        dm = [str(d).strip().lower() for d in (dm or [])]
        return bool(dm) and not any(d in PAPER for d in dm) and all(d in AUG for d in dm)

    targets = []   # (hadm, profile, gold, kind)
    if not args.ood_only:
        if args.id_set == 'balanced':
            # paper 파이프라인 테스트셋(test.py 가 쓰는 그 파일) = 275 프로필, 전부 ID
            paper_prof = json.load(open(f'{P.PROFILE_DIR}/balanced_rating_w_persona.json'))
            id_keys = set(paper_prof.keys())
            print(f"[ID 평가셋] balanced_rating (paper 파이프라인 테스트) {len(id_keys)}명", flush=True)
        elif args.id_set == 'valid':
            # HV ID validation = sft/valid 304 (ratio 선택용)
            paper_prof = json.load(open(f'{P.PROFILE_DIR}/id_valid_304.json'))
            id_keys = set(paper_prof.keys())
            print(f"[ID 평가셋] sft/valid (검증) {len(id_keys)}명", flush=True)
        else:
            id_keys = set(json.load(open(f'{P.DIALOGUE_DIR}/exp_paper/test/hg_dialogue.json')).keys())
            paper_prof = json.load(open(PAPER_PROFILE))
            print(f"[ID 평가셋] clf/test {len(id_keys)}명", flush=True)
        for h in id_keys:
            if str(h) in paper_prof:
                prof = dict(paper_prof[str(h)])
                gold = [str(d).strip().lower() for d in (prof.get('disease_mapped') or [])]
                if gold:
                    targets.append((str(h), prof, gold, 'ID'))
    if not args.id_only:
        if args.ood_set == 'exp275':
            ood_prof = json.load(open(f'{P.PROFILE_DIR}/ood_test_275.json'))
            ood_keys = list(ood_prof.keys())   # 전부 clean rescued-OOD (누수 검증 완료)
            print(f"[OOD 평가셋] 확장 275 (clean rescued-OOD)", flush=True)
        elif args.ood_set == 'clean243':
            ood_prof = json.load(open(f'{P.PROFILE_DIR}/ood_test_clean243.json'))
            ood_keys = list(ood_prof.keys())   # exp275 - 누수32 = 243 (전 ratio 공정 test)
            print(f"[OOD 평가셋] clean 243 (exp275-누수)", flush=True)
        elif args.ood_set == 'valid128':
            ood_prof = json.load(open(f'{P.PROFILE_DIR}/ood_valid_128.json'))
            ood_keys = list(ood_prof.keys())   # OOD validation (ratio 선택용), train·test 누수0
            print(f"[OOD 평가셋] valid 128 (검증)", flush=True)
        else:
            ood_prof = json.load(open(C.PREPROCESSED['exp2_id']))
            hv2_test = json.load(open(f'{P.DIALOGUE_DIR}/exp2/test/hg_dialogue.json'))
            ood_keys = [h for h, v in hv2_test.items() if aood(v.get('ground_truth'))]
            print(f"[OOD 평가셋] exp2 strict-OOD test 74", flush=True)
        for h in ood_keys:
            if str(h) in ood_prof:
                prof = dict(ood_prof[str(h)])
                real = [str(d).strip().lower() for d in (prof.get('disease_mapped') or [])]
                gold = ['other'] if EXP in ('exp3', 'exp5') else real   # 진단은 소문자 'other'로 들어가므로 맞춤
                if gold:
                    targets.append((str(h), prof, gold, 'OOD'))
    if args.n:
        ids = [t for t in targets if t[3] == 'ID'][:args.n]
        oods = [t for t in targets if t[3] == 'OOD'][:args.n]
        targets = ids + oods
    nID = sum(1 for t in targets if t[3] == 'ID')
    nOOD = sum(1 for t in targets if t[3] == 'OOD')
    print(f"평가 대상: ID {nID} + OOD {nOOD} = {len(targets)}  top_k={args.top_k}", flush=True)

    def build_inference_subgraph(top_diseases, all_probs=None):
        seeds = [d for d in top_diseases if d.lower() != 'other']
        has_other = any(d.lower() == 'other' for d in top_diseases)
        if not seeds:
            return [OTHER_NOTE] if has_other else []
        if args.subgraph_method == 'paper3hop':
            lines, _ = SG.build_subgraph_3hop(kg, meta, seeds, all_probs or {}, tau=args.tau, fmt=args.subgraph_format)
        else:
            seed = hash(tuple(seeds)) & 0xFFFFFFFF
            lines, _, _ = SG.build_subgraph(kg, meta, seeds, seed=seed, fmt=args.subgraph_format, **sg_cfg)
        if has_other:
            lines = lines + [OTHER_NOTE]
        return lines

    def detect_subgraph(pure_messages):
        res = hg.rank_diseases(messages=pure_messages[1:], gold_disease_list=None, top_k=args.top_k)
        res = res if isinstance(res, dict) else res[0]
        ranked = res['ranked_list']
        top = [d for d, _ in ranked]
        all_probs = res.get('all_probs')
        return build_inference_subgraph(top, all_probs), top

    agg = {'ID': {1: [], 2: [], 3: [], 4: []}, 'OOD': {1: [], 2: [], 3: [], 4: []}}
    results = []
    for idx, (h, prof, gt, kind) in enumerate(targets):
        prof['disease'] = gt
        ccs = prof.get('chiefcomplaint_new') or ['']
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
                else:
                    resp = our_sim.generate_patient_response_w_persona(PATIENT_PROMPT, pure_messages, gen_p, prof)
                    pure_messages.append({"role": "user", "content": resp})
                    subgraph_lines, top = detect_subgraph(pure_messages)
                curr_turn += 1
        except Exception as e:
            results.append({"hadm": h, "kind": kind, "error": f"loop:{str(e)[:50]}"})
            continue
        for k in (1, 2, 3, 4):
            agg[kind][k].append(recall[k])
        _r = {"hadm": h, "kind": kind, "gt": gt, "final_diagnosis": final_diagnosis,
              "recall": recall, "turns": curr_turn, "top_k": top}
        if not args.no_save_dialogue:   # 기본: 항상 대화+subgraph 저장 (재인퍼런스 비용 방지)
            _r["dialogue"] = pure_messages
            _r["subgraph_lines"] = subgraph_lines
            _r["n_subgraph_lines"] = len(subgraph_lines)
        results.append(_r)
        if (idx + 1) % 10 == 0:
            done = len(agg['ID'][4]) + len(agg['OOD'][4])
            r4i = np.mean(agg['ID'][4]) if agg['ID'][4] else 0
            r4o = np.mean(agg['OOD'][4]) if agg['OOD'][4] else 0
            print(f"  [{idx+1}/{len(targets)}] done {done}  ID@4={r4i:.3f} OOD@4={r4o:.3f}", flush=True)

    metrics = {"exp": EXP, "hg_ratio": args.hg_ratio, "hv_ratio": args.hv_ratio, "top_k": args.top_k,
               "n_ID": len(agg['ID'][4]), "n_OOD": len(agg['OOD'][4])}
    for kind in ('ID', 'OOD'):
        for k in (1, 2, 3, 4):
            v = agg[kind][k]
            metrics[f"{kind}_recall@{k}"] = round(float(np.mean(v)), 4) if v else None
    out_dir = f'{P.PIPELINE_RESULTS_DIR}'
    os.makedirs(out_dir, exist_ok=True)
    tag = f'{EXP}_hg{args.hg_ratio}_hv{args.hv_ratio}'
    if args.id_set == 'balanced':
        tag += '_bal'   # paper balanced_rating ID (ID-only)
    elif args.id_set == 'valid':
        tag += '_validid'   # sft/valid ID 검증
    if args.ood_only:
        if args.ood_set == 'valid128':
            tag += '_validood'   # OOD 검증
        elif args.ood_set == 'clean243':
            tag += '_oodclean'   # 누수제거 OOD test
        else:
            tag += '_ood'   # exp275 OOD test
    tag += args.tag_suffix   # 예: _paperfaithful
    metrics['id_set'] = args.id_set
    metrics['hv_dir'] = hv_dir
    json.dump(metrics, open(f'{out_dir}/{tag}_metrics.json', 'w'), ensure_ascii=False, indent=2)
    json.dump(results, open(f'{out_dir}/{tag}_results.json', 'w'), ensure_ascii=False)
    print("\n===== Pipeline Metrics =====", flush=True)
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nSaved → {out_dir}/{tag}_*", flush=True)


if __name__ == '__main__':
    main()
