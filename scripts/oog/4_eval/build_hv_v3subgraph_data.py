"""HV 학습데이터의 subgraph 필드를 v3 KG 기반으로 재생성 (2 변형, Phase B).

기존 exp6_hv 학습 subgraph = gold 3-hop oracle(v1 KG, disease-centric) → 인퍼런스(HG top-k, v3, symptom)와
구성·포맷·KG 모두 불일치(= exp6-rescue-failure-analysis 의 'oracle-subgraph mismatch').

두 변형 모두 인퍼런스와 100% 동일한 구성으로 subgraph 재생성:
  build_subgraph_3hop(v3_KG, seeds, probs, tau=0.005, fmt='symptom')
차이는 seed/prob 출처뿐 →  A vs B 가 oracle-mismatch 가설을 정확히 격리:
  A (hgpred, realistic): seeds=HG top-k, probs=HG all_probs        → gold 는 HG 가 맞출 때만 존재(~40%)
  B (gold,   oracle):    seeds=[gold]+HG top-k, probs=HG∪{gold:1.0} → gold 항상 존재(recall=1.0)

HG(exp6_hg_r35) 예측은 r35 superset 대화에 1회 실행 후 hadm 별 캐시(모든 ratio 공유).
출력: 3_train/data_exp6_hv_v3a/  ,  3_train/data_exp6_hv_v3b/  (train_r{R}_hv_symcentric.json)
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import os, sys, json, argparse, copy

ap = argparse.ArgumentParser()
ap.add_argument('--gpu', default='0')
ap.add_argument('--top_k', type=int, default=2)      # 인퍼런스 default 와 동일
ap.add_argument('--tau', type=float, default=0.005)  # 인퍼런스와 동일
ap.add_argument('--ratios', default='0,5,10,15,20,25,30,35')
ap.add_argument('--hg_ratio', type=int, default=35)
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu       # torch import 전에
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('HF_HUB_CACHE', os.path.expanduser('~/.cache/huggingface/hub'))
os.environ.setdefault('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

HERE = os.path.dirname(os.path.abspath(__file__))
import _config as C          # noqa  (scripts/oog/_config.py)
import _subgraph as SG       # noqa  (scripts/oog/_subgraph.py)
import pandas as pd          # noqa
from kgcq.graph import DiagnosticKnowledgeGraph      # noqa
from kgcq.models import EmbeddingModel               # noqa
sys.path.insert(0, HERE)
from inference_exp34 import HGAdapterDetector    # noqa  (인퍼런스와 동일 HG 로더)

RATIOS = [int(x) for x in args.ratios.split(',')]

# --- KG: HG 라벨 = augmented_v3 의 528 질환 (학습과 동일 순서), subgraph 도 v3 ---
hg_nodes = pd.read_csv(C.KG['augmented_v3']['nodes'])
disease_names_hg = [r['name'] for _, r in hg_nodes.iterrows() if r['label'] == 'Disease']   # 528, HG head 순서

v3_nodes = pd.read_csv(C.KG['augmented_v3']['nodes'])
v3_edges = pd.read_csv(C.KG['augmented_v3']['edges'])
# v3 Disease 순서가 v1 과 동일한지 확인(HG 재사용 불변식)
v3_dis = [r['name'] for _, r in v3_nodes.iterrows() if r['label'] == 'Disease']
assert v3_dis == disease_names_hg, "v3 Disease 순서가 v1 과 다름 — HG 라벨 정합 깨짐!"
print(f"[KG] HG 라벨 {len(disease_names_hg)} (v1==v3 순서 확인)  v3 nodes {len(v3_nodes)} edges {len(v3_edges)}", flush=True)

emb = EmbeddingModel(C.EMB_MODEL)
kg = DiagnosticKnowledgeGraph(v3_nodes, v3_edges, embedding_model=emb)
meta = {
    'name2id': {str(r['name']).strip().lower(): r['id'] for _, r in v3_nodes.iterrows() if r['label'] == 'Disease'},
    'id2name': dict(zip(v3_nodes['id'], v3_nodes['name'])),
    'id2label': dict(zip(v3_nodes['id'], v3_nodes['label'])),
    'disease_ids': [r['id'] for _, r in v3_nodes.iterrows() if r['label'] == 'Disease'],
}

# --- HG 로드 ---
hg_dir = f'{P.MODEL_DIR}/exp6_hg_r{args.hg_ratio:02d}_Qwen2.5-7B'
print(f"[HG] {hg_dir}", flush=True)
hg = HGAdapterDetector(C.BASE_MODEL, hg_dir, disease_names_hg)


def hg_predict(dialogue):
    """인퍼런스 detect_subgraph 와 동일: dialogue[1:](greeting 제외) → HG top_k + all_probs."""
    msgs = dialogue[1:] if dialogue and len(dialogue) > 1 else dialogue
    res = hg.rank_diseases(messages=msgs, gold_disease_list=None, top_k=args.top_k)
    res = res if isinstance(res, dict) else res[0]
    seeds = [d for d, _ in res['ranked_list']]
    probs = res.get('all_probs') or {}
    return seeds, probs


def item_dialogue(item):
    for k, v in item.items():
        if k in ('ground_truth', 'subgraph', '_meta'):
            continue
        if isinstance(v, dict) and v.get('dialogue') and len(v['dialogue']) > 1:
            return v['dialogue']
    return None


# --- 1) HG 예측 캐시 (r35 superset) ---
sup_path = f'{P.TRAIN_DATA_DIR}/data_exp6_hv/train_r{max(RATIOS):02d}_hv_symcentric.json'
superset = json.load(open(sup_path))
print(f"[cache] r{max(RATIOS)} superset {len(superset)} hadm 에 HG 예측 …", flush=True)
cache = {}
n_nodlg = 0
for i, (hadm, item) in enumerate(superset.items()):
    dlg = item_dialogue(item)
    if dlg is None:
        cache[hadm] = None
        n_nodlg += 1
        continue
    seeds, probs = hg_predict(dlg)
    cache[hadm] = {'seeds': seeds, 'probs': probs}
    if (i + 1) % 300 == 0:
        print(f"    {i+1}/{len(superset)}", flush=True)
print(f"[cache] 완료. dialogue 없음 {n_nodlg}", flush=True)


def build_sg(seeds, probs):
    if not seeds:
        return [], set()
    return SG.build_subgraph_3hop(kg, meta, seeds, probs, tau=args.tau, fmt='symptom')


# --- 2) 각 ratio×변형 subgraph 재생성 ---
for variant in ('a', 'b'):
    out = f'{P.TRAIN_DATA_DIR}/data_exp6_hv_v3{variant}'
    os.makedirs(out, exist_ok=True)
for R in RATIOS:
    data = json.load(open(f'{P.TRAIN_DATA_DIR}/data_exp6_hv/train_r{R:02d}_hv_symcentric.json'))
    outA, outB = {}, {}
    statA = {'gold_in': 0, 'empty': 0, 'lines': 0, 'n': 0}
    statB = {'gold_in': 0, 'empty': 0, 'lines': 0, 'n': 0}
    for hadm, item in data.items():
        c = cache.get(hadm)
        gt = item.get('ground_truth') or []
        gold = str(gt[0]).strip().lower() if gt else None
        seedsA = c['seeds'] if c else []
        probsA = c['probs'] if c else {}
        # A: HG-pred (realistic)
        linesA, dzA = build_sg(seedsA, probsA)
        # B: gold oracle (gold 강제 seed+prob)
        seedsB = ([gold] if gold else []) + seedsA
        probsB = dict(probsA)
        if gold:
            probsB[gold] = 1.0
        linesB, dzB = build_sg(seedsB, probsB)

        ia = copy.copy(item); ia['subgraph'] = linesA; outA[hadm] = ia
        ib = copy.copy(item); ib['subgraph'] = linesB; outB[hadm] = ib
        for st, lines, dz in ((statA, linesA, dzA), (statB, linesB, dzB)):
            st['n'] += 1
            st['lines'] += len(lines)
            if not lines:
                st['empty'] += 1
            if gold and gold in dz:
                st['gold_in'] += 1
    json.dump(outA, open(f'{P.TRAIN_DATA_DIR}/data_exp6_hv_v3a/train_r{R:02d}_hv_symcentric.json', 'w'), ensure_ascii=False)
    json.dump(outB, open(f'{P.TRAIN_DATA_DIR}/data_exp6_hv_v3b/train_r{R:02d}_hv_symcentric.json', 'w'), ensure_ascii=False)

    def fmt(s):
        n = max(1, s['n'])
        return f"gold_in {s['gold_in']}/{s['n']} ({100*s['gold_in']/n:.0f}%)  empty {s['empty']}  avg_lines {s['lines']/n:.1f}"
    print(f"  r{R:02d} ({len(data)})  A[{fmt(statA)}]  B[{fmt(statB)}]", flush=True)

print("\n완료. → data_exp6_hv_v3a/ (realistic)  data_exp6_hv_v3b/ (gold oracle)")
