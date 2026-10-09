"""Build the MIMIC-validated augmented graph (augmented_v3 = G+ used for the final KG-augmentation results).
Usage: python build_augmented_kg_v3.py [output_dir]   (default data/kg/augmented_v3)

augmented KG v3 = prune-only 컴팩트 (실 MIMIC 노트로 검증된 증상만).
- OOG 증상: prune_plan_merged.json 의 keep(검증됨)만, cap 8. no_data 41개는 v2 방식(v1→기존노드, cap).
- cause/risk: paper 노드만 (v2 동일). 기존노드 매핑=dedup. 528 Disease 순서 보존.
출력: data/KG/augmented_v3/{nodes,edges}.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import csv, json, sys
from collections import defaultdict
RV2 = P.RV2
SYM_CAP, CAUSE_CAP, RISK_CAP = 8, 2, 2


def rn(p): return list(csv.DictReader(open(p)))


pn, pe = rn(f"{P.KG_DIR}/paper/nodes.csv"), rn(f"{P.KG_DIR}/paper/edges.csv")
an, ae = rn(f"{P.KG_DIR}/augmented/nodes.csv"), rn(f"{P.KG_DIR}/augmented/edges.csv")
aid = {n['id']: (n['name'].strip(), n['label']) for n in an}
paper_dis = {n['name'].strip().lower() for n in pn if n['label'] == 'Disease'}
plan = json.load(open(f"{P.KG_V3_DIR}/prune_plan_merged.json"))

name2id = {'Symptom': {}, 'Cause': {}, 'Risk_Factor': {}}
for n in pn:
    if n['label'] in name2id:
        name2id[n['label']].setdefault(n['name'].strip().lower(), n['id'])
augonly = {'Symptom': {}, 'Cause': {}, 'Risk_Factor': {}}
for n in an:
    nl = n['name'].strip().lower()
    if n['label'] in name2id and nl not in name2id[n['label']]:
        augonly[n['label']].setdefault(nl, n['id'])

out_nodes = [dict(n) for n in pn]
out_edges = [dict(e) for e in pe]
used = {n['id'] for n in out_nodes}
_ctr = [max(int(i[1:]) for i in list(aid) + [n['id'] for n in pn])]
def new_id():
    _ctr[0] += 1; return f"N{_ctr[0]}"
added = {}
def add_node(nid, name, label):
    if nid not in used: out_nodes.append({'id': nid, 'name': name, 'label': label}); used.add(nid)
def mapf(name, label):
    nl = name.strip().lower()
    if not nl: return None
    if nl in name2id[label]: return name2id[label][nl]
    if nl in augonly[label]:
        nid = augonly[label][nl]; add_node(nid, aid[nid][0], label); return nid
    if nl in added: return added[nl]
    nid = new_id(); add_node(nid, name.strip(), label); added[nl] = nid; return nid

# OOG 질병 노드 (aug 순서)
oog_dis = [n for n in an if n['label'] == 'Disease' and n['name'].strip().lower() not in paper_dis]
for n in oog_dis: add_node(n['id'], n['name'].strip(), 'Disease')
oog_id = {n['name'].strip().lower(): n['id'] for n in oog_dis}

# v1 OOG 후보 (fallback/cause/risk)
oog_ids = set(oog_id.values())
v1 = defaultdict(lambda: {'Symptom': [], 'Cause': [], 'Risk_Factor': []})
for e in ae:
    if e['end'] in oog_ids and e['start'] in aid:
        lb = aid[e['start']][1]
        if lb in v1[e['end']]: v1[e['end']][lb].append(aid[e['start']][0])

n_val = n_nodata = 0
for dl, did in oog_id.items():
    p = plan.get(dl, {})
    if p and not p.get('no_data') and p.get('keep'):   # 검증됨(keep 있음): keep만
        syms = p['keep'][:SYM_CAP]; n_val += 1
    else:                                    # no_data 또는 keep 전멸: v2 fallback (v1→기존노드, cap)
        syms = []; seen = set()
        for s in v1[did]['Symptom']:
            sl = s.strip().lower()
            if sl not in seen: seen.add(sl); syms.append(s)
            if len(syms) >= SYM_CAP: break
        n_nodata += 1
    for s in syms[:SYM_CAP]:
        sid = mapf(s, 'Symptom')
        if sid: out_edges.append({'start': sid, 'end': did, 'type': 'caused_by'})
    nc = nr = 0
    for s in v1[did]['Cause']:
        if s.strip().lower() in name2id['Cause']:
            out_edges.append({'start': name2id['Cause'][s.strip().lower()], 'end': did, 'type': 'can_cause'}); nc += 1
        if nc >= CAUSE_CAP: break
    for s in v1[did]['Risk_Factor']:
        if s.strip().lower() in name2id['Risk_Factor']:
            out_edges.append({'start': name2id['Risk_Factor'][s.strip().lower()], 'end': did, 'type': 'is_a_risk_factor_of'}); nr += 1
        if nr >= RISK_CAP: break

# dedup 엣지
seen = set(); ded = []
for e in out_edges:
    k = (e['start'], e['end'], e['type'])
    if k not in seen: seen.add(k); ded.append(e)
out_edges = ded

import os
od = sys.argv[1] if len(sys.argv) > 1 else f"{P.KG_DIR}/augmented_v3"; os.makedirs(od, exist_ok=True)
with open(f"{od}/nodes.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, ['id', 'name', 'label']); w.writeheader(); w.writerows(out_nodes)
with open(f"{od}/edges.csv", 'w', newline='') as f:
    w = csv.DictWriter(f, ['start', 'end', 'type']); w.writeheader(); w.writerows(out_edges)
from collections import Counter
print(f"v3: {len(out_nodes)} nodes {dict(Counter(n['label'] for n in out_nodes))}, {len(out_edges)} edges")
print(f"  OOG 검증-keep 사용 {n_val}, no_data v2-fallback {n_nodata}, 신규노드 {len(added)}")
