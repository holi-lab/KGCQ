"""HG sweep table: IG / OOG / all Recall@1-4 per experiment and ratio (all = 0.75 IG + 0.25 OOG), from the
test_predictions.json / valid_predictions.json of each adapter.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import json, csv, os
import pandas as pd


def disease_names_from_nodes(nodes_csv):
    df = pd.read_csv(nodes_csv)
    return set(str(r["name"]).strip().lower() for _, r in df.iterrows() if r["label"] == "Disease")


paper = disease_names_from_nodes(P.KG["original"]["nodes"])
W_IG, W_OOG = 0.75, 0.25   # 3:1


def r_at_k(ts, tp, k):
    return len(ts & set(tp[:k])) / len(ts) if ts else None


def is_oog(ts, exp):
    return ('other' in ts) if exp == 'exp5' else any(t not in paper for t in ts)


def compute(exp, R, split):
    fn = 'test_predictions.json' if split == 'test' else 'valid_predictions.json'
    f = f"{P.MODEL_DIR}/{exp}_hg_r{R:02d}_Qwen2.5-7B/{fn}"
    if not os.path.exists(f):
        return None
    d = json.load(open(f)); ig = {k: [] for k in (1, 2, 3, 4)}; oog = {k: [] for k in (1, 2, 3, 4)}
    for r in d:
        ts = set(t.strip().lower() for t in r['true_labels']); tp = [p.strip().lower() for p in r['top4']]
        b = oog if is_oog(ts, exp) else ig
        for k in (1, 2, 3, 4):
            v = r_at_k(ts, tp, k)
            if v is not None:
                b[k].append(v)
    igm = {k: (sum(ig[k]) / len(ig[k]) if ig[k] else None) for k in (1, 2, 3, 4)}
    oom = {k: (sum(oog[k]) / len(oog[k]) if oog[k] else None) for k in (1, 2, 3, 4)}
    return igm, oom


def row_vals(igm, oom):
    """IG@1-4, OOG@1-4, all@1-4(3:1)"""
    out = []
    for k in (1, 2, 3, 4):
        out.append(round(igm[k], 4) if igm[k] is not None else '')
    for k in (1, 2, 3, 4):
        out.append(round(oom[k], 4) if oom[k] is not None else '')
    for k in (1, 2, 3, 4):
        ig = igm[k] if igm[k] is not None else None
        oo = oom[k] if oom[k] is not None else 0.0
        out.append(round(W_IG * ig + W_OOG * oo, 4) if ig is not None else '')
    return out


def main():
    ph = json.load(open(f"{P.RESULTS_DIR}/results_paper_hg_on_exp34.json"))

    def paper_row(split):
        p = ph[split]
        igm = {k: p[f'ID@{k}'] for k in (1, 2, 3, 4)}
        oom = {k: 0.0 for k in (1, 2, 3, 4)}
        return row_vals(igm, oom)

    cols = ['exp', 'ratio', 'split',
            'IG@1', 'IG@2', 'IG@3', 'IG@4', 'OOG@1', 'OOG@2', 'OOG@3', 'OOG@4',
            'all@1', 'all@2', 'all@3', 'all@4']
    out = f"{P.RESULTS_DIR}/hg_sweep_recall_3to1.csv"
    with open(out, 'w', newline='', encoding='utf-8-sig') as fp:
        w = csv.writer(fp); w.writerow(cols)
        for exp in ['exp6', 'exp5']:
            for split in ['test', 'valid']:
                w.writerow(['paper_HG', '-', split] + paper_row(split))
            for R in [0, 5, 10, 15, 20, 25, 30, 35]:
                for split in ['test', 'valid']:
                    c = compute(exp, R, split)
                    w.writerow([exp, R, split] + (row_vals(*c) if c else ['(missing)'] + [''] * 11))
            w.writerow([])
    print("CSV:", out)


if __name__ == '__main__':
    main()
