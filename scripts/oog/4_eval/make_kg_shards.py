"""190 OOG 질병을 10 shard로 분할. 각 shard = self-contained 입력(질병+KG증상+실HPI non-eval).
출력: data/kg_v3/shards/shard_00_input.json .. shard_09_input.json
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import json, csv, os, random
from collections import defaultdict
RV2 = P.RV2
N_SHARD, NOTE_CAP = 10, 25   # 질병당 노트 최대 25개(에이전트 입력 크기 제한)

pn = list(csv.DictReader(open(f"{P.KG_DIR}/paper/nodes.csv")))
an = list(csv.DictReader(open(f"{P.KG_DIR}/augmented/nodes.csv")))
ae = list(csv.DictReader(open(f"{P.KG_DIR}/augmented/edges.csv")))
aid = {x['id']: (x['name'].strip(), x['label']) for x in an}
paper_dis = {x['name'].strip().lower() for x in pn if x['label'] == 'Disease'}
oog = {x['name'].strip().lower(): x['id'] for x in an if x['label'] == 'Disease' and x['name'].strip().lower() not in paper_dis}
# KG 증상 per OOG 질병
kg_sym = defaultdict(list)
oog_ids = {v: k for k, v in oog.items()}
for e in ae:
    if e['end'] in oog_ids and e['start'] in aid and aid[e['start']][1] == 'Symptom':
        kg_sym[oog_ids[e['end']]].append(aid[e['start']][0])

hpi = json.load(open(f"{P.KG_V3_DIR}/oog_hpi_by_disease.json"))

diseases = sorted(oog)   # 190
random.seed(42); random.shuffle(diseases)
shards = [diseases[i::N_SHARD] for i in range(N_SHARD)]
os.makedirs(f"{P.KG_V3_DIR}/shards", exist_ok=True)
for si, ds in enumerate(shards):
    obj = {}
    for d in ds:
        rec = hpi.get(d, {})
        notes = (rec.get('train', []) + rec.get('other', []))[:NOTE_CAP]   # non-eval 우선(train+other)
        obj[d] = {
            'kg_symptoms': sorted(set(kg_sym.get(d, []))),
            'n_notes_noneval': len(rec.get('train', [])) + len(rec.get('other', [])),
            'hpi_notes': [{'complaint': n['complaint'][:120], 'hpi': n['hpi'][:1500]} for n in notes],
        }
    json.dump(obj, open(f"{P.KG_V3_DIR}/shards/shard_{si:02d}_input.json", 'w'), ensure_ascii=False, indent=1)
    nnote = sum(len(v['hpi_notes']) for v in obj.values())
    ncov = sum(1 for v in obj.values() if v['hpi_notes'])
    print(f"  shard_{si:02d}: {len(ds)} 질병, HPI보유 {ncov}, 노트 {nnote}")
print("완료: data/kg_v3/shards/")
