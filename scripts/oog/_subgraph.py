"""paper 정합 subgraph 빌더 (reverse 포맷 + feature 부분선택).

설계 근거 (조사):
- canonical paper 생성(code_1210/data_gen.py)은 reverse 포맷 subgraph 사용 → reverse 로 통일.
- 질병 선택: gold + feature-overlap top-N (사용자 "gold→공유feature→공유질병"; anchor ~28 = paper).
- feature 선택: KG 전체 feature(질병당 ~15-20)를 그대로 넣으면 paper(~7.3 f/d)의 2배 토큰.
  → 'connecting'(선택질병 집합 내 ≥min_share 질병이 공유하는 feature) 위주로 부분선택해 paper 밀도에 맞춤.
  gold 질병은 anchor 이므로 feature 전체 유지.

paper code(main/graph.py)는 수정하지 않고, DiagnosticKnowledgeGraph.graph(networkx) 만 읽어서
reverse 텍스트를 직접 렌더링한다 (get_subgraph_text 의 reverse 분기와 동일 포맷).
"""
import os
import random
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.path.join(_HERE, 'lib')   # graph.py 등 복사본

FEATURE_LABELS = ('Symptom', 'Risk_Factor', 'Cause')
REL_KEY = {'Symptom': 'symptoms', 'Risk_Factor': 'risk factors', 'Cause': 'causes'}


def load_kg(nodes_csv, edges_csv):
    """로컬 lib/graph.py 의 DiagnosticKnowledgeGraph 로드 (embedding 없이 구조만)."""
    import sys
    if _LIB not in sys.path:
        sys.path.insert(0, _LIB)
    from graph import DiagnosticKnowledgeGraph
    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    kg = DiagnosticKnowledgeGraph(nodes, edges, embedding_model=None)
    meta = {
        'name2id': {str(r['name']).strip().lower(): r['id'] for _, r in nodes.iterrows() if r['label'] == 'Disease'},
        'id2name': dict(zip(nodes['id'], nodes['name'])),
        'id2label': dict(zip(nodes['id'], nodes['label'])),
        'disease_ids': [r['id'] for _, r in nodes.iterrows() if r['label'] == 'Disease'],
    }
    return kg, meta


def _features_of(kg, meta, d_id):
    """질병 d_id 의 feature 노드 id 집합 (label 별로)."""
    out = {}
    for p in kg.graph.predecessors(d_id):
        lab = meta['id2label'].get(p)
        if lab in FEATURE_LABELS:
            out.setdefault(lab, set()).add(p)
    return out


def _all_feature_ids(feat_by_label):
    s = set()
    for v in feat_by_label.values():
        s |= v
    return s


def select_diseases(kg, meta, gold_disease, ratio=0.3, seed=42):
    """gold + feature-overlap top-N 질병 선택. (build_rule_subgraph 와 동일 로직)"""
    name2id, id2label = meta['name2id'], meta['id2label']
    gold_ids, gold_feats = set(), set()
    for g in gold_disease:
        gid = name2id.get(str(g).strip().lower())
        if gid is None:
            continue
        gold_ids.add(gid)
        gold_feats |= _all_feature_ids(_features_of(kg, meta, gid))
    if not gold_ids:
        return [], [], 0
    shared = {}
    for d_id in meta['disease_ids']:
        if d_id in gold_ids:
            continue
        df = _all_feature_ids(_features_of(kg, meta, d_id))
        s = len(gold_feats & df)
        if s > 0:
            shared[d_id] = s
    pool_size = len(shared) + len(gold_ids)
    n_others = max(1, int(pool_size * ratio) - len(gold_ids))
    cands = list(shared.items())
    rng = random.Random(seed)
    rng.shuffle(cands)
    cands.sort(key=lambda x: -x[1])
    selected = list(gold_ids) + [d for d, _ in cands[:n_others]]
    return selected, list(gold_ids), pool_size


def _group_by_label(meta, feat_ids):
    """feature id 집합 -> {label: set(id)} (label 별 분류)."""
    out = {}
    for fid in feat_ids:
        lab = meta['id2label'].get(fid)
        if lab in FEATURE_LABELS:
            out.setdefault(lab, set()).add(fid)
    return out


def select_features(kg, meta, selected_ids, gold_ids, mode='cap_own',
                    cap=7, cap_gold=False, seed=42, min_share=2, topk=None):
    """선택 질병별 포함할 feature id 집합 결정.

    paper HG 분석: subgraph feature 의 89% 가 비-gold 질병의 '자기 feature' = 질병-특징-질병-특징(3-hop).
    따라서 paper 충실 방식은 '각 질병의 자기 feature 를 포함하되 개수만 cap' (공유 우선 아님).

    mode:
      - 'cap_own'   : (기본) 질병당 자기 feature 를 cap 개로 seeded-subsample. gold 는 cap_gold 에 따라.
      - 'all'       : 질병의 모든 feature (KG 전체) — 깨끗한 3-hop, paper 대비 ~2배 토큰
      - 'connecting': 선택질병 집합 내 >=min_share 공유 feature (참고용, paper 와 선택방향 다름)
      - 'topk'      : 공유수 상위 topk (참고용)
    반환: {disease_id: {label: set(feature_id)}}
    """
    feats = {d: _features_of(kg, meta, d) for d in selected_ids}
    gold_set = set(gold_ids)

    if mode == 'all':
        return feats

    if mode == 'cap_own':
        out = {}
        for d in selected_ids:
            own = _all_feature_ids(feats[d])
            if d in gold_set and not cap_gold:
                out[d] = feats[d]  # gold 전체 유지
                continue
            if len(own) > cap:
                rng = random.Random(seed ^ (int(d) if str(d).isdigit() else hash(d)))
                own = set(rng.sample(sorted(own, key=lambda x: str(x)), cap))
            out[d] = _group_by_label(meta, own)
        return out

    # --- 참고용 모드 (connecting / topk) ---
    from collections import Counter
    fcount = Counter()
    for d in selected_ids:
        for fid in _all_feature_ids(feats[d]):
            fcount[fid] += 1
    out = {}
    for d in selected_ids:
        if d in gold_set:
            out[d] = feats[d]
            continue
        keep = {}
        if mode == 'connecting':
            for lab, ids in feats[d].items():
                k = {fid for fid in ids if fcount[fid] >= min_share}
                if k:
                    keep[lab] = k
        elif mode == 'topk':
            allids = set(sorted(_all_feature_ids(feats[d]), key=lambda fid: -fcount[fid])[:(topk or 8)])
            for lab, ids in feats[d].items():
                k = ids & allids
                if k:
                    keep[lab] = k
        else:
            keep = feats[d]
        out[d] = keep
    return out


def render_reverse(meta, selected_ids, feat_by_disease):
    """reverse 포맷 줄 생성 (get_subgraph_text 의 reverse 분기와 동일 포맷).
    'Disease 'X' has symptoms: ['a', 'b'].'  (빈 질병은 생략)"""
    id2name = meta['id2name']
    lines = []
    for d in selected_ids:
        dname = id2name[d]
        fb = feat_by_disease.get(d, {})
        for lab in FEATURE_LABELS:
            ids = fb.get(lab)
            if not ids:
                continue
            names = [str(id2name[f]) for f in ids]
            lines.append(f"Disease '{dname}' has {REL_KEY[lab]}: {names}.")
    return lines


REL_PHRASE = {'Symptom': 'is a symptom of', 'Risk_Factor': 'is a risk factor for', 'Cause': 'is a cause of'}

def render_symptom_centric(meta, selected_ids, feat_by_disease):
    """paper 인퍼런스 포맷(증상-중심): "'feature' is a symptom of [diseases]."
    (paper main/graph.py get_subgraph_text 의 subgraph_lines 분기와 동일 포맷)"""
    id2name = meta['id2name']
    inv = {lab: {} for lab in FEATURE_LABELS}
    for d in selected_ids:
        dname = id2name[d]
        fb = feat_by_disease.get(d, {})
        for lab in FEATURE_LABELS:
            for f in fb.get(lab, []):
                inv[lab].setdefault(f, []).append(dname)
    lines = []
    for lab in FEATURE_LABELS:
        for f, dz in inv[lab].items():
            lines.append(f"'{id2name[f]}' {REL_PHRASE[lab]} {dz}.")
    return lines


def build_subgraph_3hop(kg, meta, seeds, total_probs, tau=0.005, hops=3, fmt='symptom'):
    """paper extract_subgraph_3hop 재현: seed 질병의 3-hop 이웃 중 prob>=tau 질병만 포함.
    fmt='symptom'(증상-중심) / 'disease'(질병-중심). 반환 (lines, disease_name_set)."""
    name2id, id2name, id2label = meta['name2id'], meta['id2name'], meta['id2label']
    seed_ids = [name2id[s.strip().lower()] for s in seeds if s.strip().lower() in name2id]
    if not seed_ids:
        return [], set()
    sub = kg.get_subgraph(seed_ids, hops=hops)
    constraint = set(str(d).strip().lower() for d, p in total_probs.items() if p >= tau)
    s2d = {lab: {} for lab in FEATURE_LABELS}
    d2s = {lab: {} for lab in FEATURE_LABELS}
    dz = set()
    for u, v in sub.edges():
        lu, lv = id2label.get(u), id2label.get(v)
        # disease-end / feature-end 판별 (방향 무관)
        if lv == 'Disease' and lu in FEATURE_LABELS:
            d_id, f_id, lab = v, u, lu
        elif lu == 'Disease' and lv in FEATURE_LABELS:
            d_id, f_id, lab = u, v, lv
        else:
            continue
        dname = id2name[d_id]
        if str(dname).strip().lower() not in constraint:
            continue
        fname = id2name[f_id]
        s2d[lab].setdefault(fname, []).append(dname)
        d2s[lab].setdefault(dname, []).append(fname)
        dz.add(str(dname).strip().lower())
    if fmt == 'symptom':
        lines = [f"'{f}' {REL_PHRASE[lab]} {d}." for lab in FEATURE_LABELS for f, d in s2d[lab].items()]
    else:
        lines = [f"Disease '{d}' has {REL_KEY[lab]}: {f}." for lab in FEATURE_LABELS for d, f in d2s[lab].items()]
    return lines, dz


def select_diseases_by_symptoms(kg, meta, symptom_ids, top_n=27, seed=42):
    """OOD 용: 환자 증상 노드 와 증상을 공유하는 in-graph 질병 top-N (gold 없음).
    = doctor_v7_ood_gen 의 '증상 공유 closest in-graph 질병'."""
    symptom_ids = set(symptom_ids)
    if not symptom_ids:
        return []
    share = {}
    for d_id in meta['disease_ids']:
        s = len(symptom_ids & set(kg.graph.predecessors(d_id)))
        if s > 0:
            share[d_id] = s
    if not share:
        return []
    cands = list(share.items())
    rng = random.Random(seed)
    rng.shuffle(cands)               # tie-break
    cands.sort(key=lambda x: -x[1])  # 공유 수 desc
    return [d for d, _ in cands[:top_n]]


def build_ood_subgraph(kg, meta, symptom_ids, seed=42, cap=7, top_n=27):
    """OOD subgraph (증상-anchored, gold 없음). 반환: (lines_reverse, names, stats).
    feature 는 ID 와 동일하게 cap_own 으로 질병당 cap 개 (paper 밀도 정합). gold 없으므로 전부 cap."""
    selected = select_diseases_by_symptoms(kg, meta, symptom_ids, top_n=top_n, seed=seed)
    if not selected:
        return [], [], {'anchor': 0, 'pool_size': 0, 'lines': 0, 'feat_per_disease': 0.0}
    feat_by_disease = select_features(kg, meta, selected, gold_ids=[],
                                      mode='cap_own', cap=cap, cap_gold=True, seed=seed)
    lines = render_reverse(meta, selected, feat_by_disease)
    nfeat = sum(len(_all_feature_ids(fb)) for fb in feat_by_disease.values())
    stats = {'anchor': len(selected), 'pool_size': len(selected), 'lines': len(lines),
             'feat_per_disease': nfeat / max(1, len(selected))}
    return lines, [meta['id2name'][d] for d in selected], stats


OOD_OTHER_LINE = ("The patient's true condition may be a disease NOT present in the knowledge graph "
                  "below; in that case it must be classified as 'Other'.")


def build_ood_hv_subgraph(kg, meta, symptom_ids, seed=42, cap=7, top_k=1):
    """OOD HV subgraph (사용자 설계): 'Other' 1줄 linearize + 증상 공유 top_k in-graph 질병의 reverse subgraph.
    HG(27개 나열)와 달리 최소 형태로, HV(의사모델)가 'Other' 신호를 명확히 학습하게 함.
    반환 (lines, names, stats). 증상 매핑 실패해도 최소 [OTHER_LINE] 반환(빈 subgraph 없음)."""
    selected = select_diseases_by_symptoms(kg, meta, symptom_ids, top_n=top_k, seed=seed)
    if not selected:
        return [OOD_OTHER_LINE], [], {'anchor': 0, 'pool_size': 0, 'lines': 1, 'feat_per_disease': 0.0}
    feat = select_features(kg, meta, selected, gold_ids=[], mode='cap_own', cap=cap, cap_gold=True, seed=seed)
    lines = [OOD_OTHER_LINE] + render_reverse(meta, selected, feat)
    nfeat = sum(len(_all_feature_ids(fb)) for fb in feat.values())
    stats = {'anchor': len(selected), 'pool_size': len(selected), 'lines': len(lines),
             'feat_per_disease': nfeat / max(1, len(selected))}
    return lines, [meta['id2name'][d] for d in selected], stats


def build_subgraph(kg, meta, gold_disease, ratio=0.25, seed=42,
                   feature_mode='cap_own', cap=7, cap_gold=False, min_share=2, topk=None,
                   fmt='disease'):
    """반환: (lines, selected_disease_names, stats). fmt='disease'(질병-중심,기본) / 'symptom'(paper 인퍼런스 증상-중심)"""
    selected, gold_ids, pool_size = select_diseases(kg, meta, gold_disease, ratio=ratio, seed=seed)
    if not selected:
        return [], [], {'anchor': 0, 'pool_size': 0, 'lines': 0, 'feat_per_disease': 0.0}
    feat_by_disease = select_features(kg, meta, selected, gold_ids,
                                      mode=feature_mode, cap=cap, cap_gold=cap_gold,
                                      seed=seed, min_share=min_share, topk=topk)
    lines = render_symptom_centric(meta, selected, feat_by_disease) if fmt == 'symptom' \
        else render_reverse(meta, selected, feat_by_disease)
    nfeat = sum(len(_all_feature_ids(fb)) for fb in feat_by_disease.values())
    stats = {
        'anchor': len(selected),
        'pool_size': pool_size,
        'lines': len(lines),
        'feat_per_disease': nfeat / max(1, len(selected)),
    }
    names = [meta['id2name'][d] for d in selected]
    return lines, names, stats
