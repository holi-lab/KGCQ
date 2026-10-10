"""3-hop subgraph builder used by the out-of-graph experiments (same rule as kgcq.subgraph_extractor)."""

FEATURE_LABELS = ('Symptom', 'Risk_Factor', 'Cause')
REL_KEY = {'Symptom': 'symptoms', 'Risk_Factor': 'risk factors', 'Cause': 'causes'}
REL_PHRASE = {'Symptom': 'is a symptom of', 'Risk_Factor': 'is a risk factor for', 'Cause': 'is a cause of'}


def build_subgraph_3hop(kg, meta, seeds, total_probs, tau=0.005, hops=3, fmt='symptom'):
    """3-hop neighbourhood of the seed diseases, keeping only diseases with HG probability >= tau.
    Returns (lines, set of disease names). fmt: 'symptom' (attribute-centric) or 'disease' (disease-centric)."""
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
