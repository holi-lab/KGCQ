"""OOG 질병별 실제 MIMIC HPI(History of Present Illness) 추출 (note_section.csv에서).
질병 매핑: profiles_paperKG_OOD_50289(hadm→raw disease) + catalog(raw→kg_disease, same/child) → oog190.
train/eval 태깅. 출력: data/kg_v3/oog_hpi_by_disease.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import json, csv
from collections import defaultdict
import pandas as pd
RV2 = P.RV2
MED = P.RV2

# 190 OOG 질병
pn = list(csv.DictReader(open(f"{P.KG_DIR}/paper/nodes.csv")))
an = list(csv.DictReader(open(f"{P.KG_DIR}/augmented/nodes.csv")))
paper_dis = {x['name'].strip().lower() for x in pn if x['label'] == 'Disease'}
oog190 = {x['name'].strip().lower() for x in an if x['label'] == 'Disease' and x['name'].strip().lower() not in paper_dis}

# catalog: raw disease → kg_disease (same/child)
raw2kg = defaultdict(set)
for r in csv.DictReader(open(f"{P.SPLIT_INPUT_DIR}/catalog_with_relationship.csv")):
    if r.get('disease_relationship', '').strip() in ('same', 'child'):
        kg = (r.get('kg_disease') or '').strip().lower().lstrip('[new]').strip()
        if kg in oog190:
            raw2kg[(r.get('disease') or '').strip().lower()].add(kg)

# hadm → oog diseases
hadm2dis = defaultdict(set)
for line in open(P.OOG_PROFILE_POOL_JSONL):
    p = json.loads(line); h = str(p['hadm_id'])
    for rd in (p.get('disease') or []):
        for kg in raw2kg.get(str(rd).strip().lower(), ()):
            hadm2dis[h].add(kg)
oog_hadms = set(hadm2dis)
print(f"OOG 질병 {len(oog190)}, 매핑된 hadm {len(oog_hadms)}, 커버 질병 {len(set().union(*hadm2dis.values()) if hadm2dis else set())}")

# train/eval 태깅
hg = json.load(open(f"{P.OOD_SPLIT_DIR}/hg_train_ood_ordered.json"))['order']
hv = json.load(open(f"{P.OOD_SPLIT_DIR}/hv_train_ood_ordered.json"))['order']
train = set(map(str, hg)) | set(map(str, hv))
evalids = set()
for f in ['ood_valid_128', 'ood_test_clean243', 'ood_test_275']:
    evalids |= set(map(str, json.load(open(f"{P.PROFILE_DIR}/{f}.json")).keys()))
def split_of(h): return 'train' if h in train else ('eval' if h in evalids else 'other')

# note_section.csv → hadm HPI (청크)
cols = ['hadm_id', 'History of Present Illness', 'Complaint']
hadm_hpi = {}
for chunk in pd.read_csv(P.MIMIC_NOTE_SECTION_CSV, usecols=cols, chunksize=100000, dtype=str):
    for _, row in chunk.iterrows():
        h = str(row['hadm_id']).split('.')[0]
        if h in oog_hadms and h not in hadm_hpi:
            hadm_hpi[h] = {'hpi': (row['History of Present Illness'] or ''), 'complaint': (row['Complaint'] or '')}
print(f"note에서 매칭된 hadm {len(hadm_hpi)}/{len(oog_hadms)}")

# 질병별 집계
dis = defaultdict(lambda: {'train': [], 'other': [], 'eval': []})
for h, ds in hadm2dis.items():
    if h not in hadm_hpi: continue
    rec = {'hadm': h, 'hpi': hadm_hpi[h]['hpi'], 'complaint': hadm_hpi[h]['complaint']}
    if len((rec['hpi'] or '').strip()) < 15: continue   # 빈/deid 제외
    for d in ds:
        dis[d][split_of(h)].append(rec)

import os
os.makedirs(P.KG_V3_DIR, exist_ok=True)
out = {d: dict(v) for d, v in dis.items()}
json.dump(out, open(f"{P.KG_V3_DIR}/oog_hpi_by_disease.json", 'w'), ensure_ascii=False)
# 커버리지 리포트
cov_any = sum(1 for d in oog190 if d in out and (out[d]['train'] or out[d]['other'] or out[d]['eval']))
cov_train = sum(1 for d in oog190 if d in out and out[d]['train'])
cov_ne = sum(1 for d in oog190 if d in out and (out[d]['train'] or out[d]['other']))
print(f"\n★ 실 HPI 커버: 전체 {cov_any}/190, train {cov_train}/190, non-eval(train+other) {cov_ne}/190")
print(f"저장: data/kg_v3/oog_hpi_by_disease.json")
