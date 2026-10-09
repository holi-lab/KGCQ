"""False-negative soft-coverage (rebuttal §g_coverage line 260):
OOD 프로필(gold ∉ paper 338)에서, paper-KG HG가 'gold의 가장 가까운 in-graph 친척'을
top-k 안에 지명하는 비율. 친척 = aug KG attribute(증상+cause+rf) Jaccard 최대 paper 질병.
exp5(paper+Other, HG r15)을 저장된 OOD HV 대화에 직접 적용해 full ranking 산출.
read-only(저장 대화 재사용)+HG forward만. 사용: python soft_coverage.py [gpu]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import os, sys, json
from collections import defaultdict
GPU = sys.argv[1] if len(sys.argv) > 1 else '0'
os.environ['CUDA_VISIBLE_DEVICES'] = GPU
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RV2 = P.RV2
from inference_exp34 import HGAdapterDetector
import _config as C

def norm(x): return str(x).strip().lower()
# attribute sets (augmented KG, gold가 거기에만 있으므로)
an = pd.read_csv(C.KG['augmented']['nodes']); ae = pd.read_csv(C.KG['augmented']['edges'])
pn = pd.read_csv(C.KG['paper']['nodes'])
paper_dis = [norm(r['name']) for _, r in pn.iterrows() if r['label'] == 'Disease']
lab = {r['id']: r['label'] for _, r in an.iterrows()}; id2n = {r['id']: norm(r['name']) for _, r in an.iterrows()}
attr = defaultdict(set)
for _, e in ae.iterrows():
    s, t = e['start'], e['end']
    if lab.get(s) == 'Disease' and lab.get(t) in ('Symptom', 'Cause', 'Risk_Factor'): attr[id2n[s]].add(id2n[t])
    if lab.get(t) == 'Disease' and lab.get(s) in ('Symptom', 'Cause', 'Risk_Factor'): attr[id2n[t]].add(id2n[s])
def closest_paper(gold):
    ga = attr.get(gold, set())
    if not ga: return None
    best, bj = None, -1
    for p in paper_dis:
        pa = attr.get(p, set())
        if not pa: continue
        j = len(ga & pa) / len(ga | pa)
        if j > bj: bj, best = j, p
    return best

# paper-KG HG (exp5 = 338+Other, r15)
disease_names = [r['name'] for _, r in pn.iterrows() if r['label'] == 'Disease'] + ['Other']
hg = HGAdapterDetector(C.BASE_MODEL, f'{P.MODEL_DIR}/exp5_hg_r15_Qwen2.5-7B', disease_names)
print(f'HG 로드 (paper+Other {len(disease_names)})', flush=True)

# exp5 gt는 'other'로 저장됨 → 실제 OOD 질병명은 exp6 결과(rescued gt)에서 hadm으로 매핑
exp6 = json.load(open(f'{P.PIPELINE_RESULTS_DIR}/exp6_hg20_hv20_bal_sweep_results.json'))
e6items = exp6 if isinstance(exp6, list) else list(exp6.values())
real_gt = {it['hadm']: [norm(g) for g in it['gt']] for it in e6items if it.get('kind') == 'OOD' and 'hadm' in it}

res = json.load(open(f'{P.PIPELINE_RESULTS_DIR}/exp5_hg15_hv20_bal_sweep_results.json'))
items = res if isinstance(res, list) else list(res.values())
ood = [it for it in items if it.get('kind') == 'OOD' and it.get('dialogue') and it.get('hadm') in real_gt]

K = [1, 2, 4, 10]
hit = {k: 0 for k in K}; n = 0
for i, it in enumerate(ood):
    gt = real_gt[it['hadm']]                       # 실제 OOD 질병명
    rels = [closest_paper(g) for g in gt]; rels = [r for r in rels if r]
    if not rels: continue
    n += 1
    msgs = [t for t in it['dialogue'] if isinstance(t, dict) and 'role' in t]
    if msgs and msgs[-1]['role'] == 'assistant': msgs = msgs[:-1]
    r = hg.rank_diseases(messages=msgs, gold_disease_list=None, top_k=len(disease_names))
    r = r if isinstance(r, dict) else r[0]
    ranked = [norm(d) for d, _ in r['ranked_list'] if norm(d) != 'other']   # Other 제외 (실제 질병 순위)
    for k in K:
        if any(rel in ranked[:k] for rel in rels): hit[k] += 1
    if (i + 1) % 60 == 0: print(f'  {i+1}/{len(ood)}', flush=True)

print(f'\n=== Soft-coverage: paper-KG HG가 OOD gold의 closest in-graph 친척을 top-k 지명 (n={n}) ===')
for k in K:
    print(f'  top-{k}: {hit[k]}/{n} = {hit[k]/n:.0%}')
print('\n대표 OOD → closest in-graph 친척:')
for g in ['deep vein thrombosis', 'bacteremia', 'ulcerative colitis', 'complete atrioventricular block',
          'hydronephrosis', 'intra-abdominal abscess']:
    print(f'  {g:32s} → {closest_paper(g)}')
