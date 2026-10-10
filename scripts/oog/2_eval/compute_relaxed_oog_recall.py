"""Relaxed OOG recall for KG augmentation: a case is credited when the top-k contains any of the 190 added
diseases (the analogue of answering "Other"). Computed on the full OOG sets (243 test / 128 valid).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import json, os
import pandas as pd

KG = str(P.KG_DIR)


def norm(x):
    return str(x).strip().lower()


def diseases(path):
    df = pd.read_csv(path)
    lab = 'label' if 'label' in df.columns else df.columns[1]
    return set(norm(n) for n, l in zip(df['name'], df[lab]) if norm(l) == 'disease')


paper = diseases(f'{KG}/original/nodes.csv')          # 338
aug = diseases(f'{KG}/augmented/nodes.csv')      # 528
added = aug - paper                                 # 190 nodes added by expansion


def load_ood(path):
    if not os.path.exists(path):
        _sys.exit(f'missing inference output {path}; run inference_exp34.py first')
    R = json.load(open(path))
    items = R if isinstance(R, list) else list(R.values())
    return [it for it in items if it.get('kind') == 'OOD']


def topk_preds(it, k):
    return [norm(p) for p in it['final_diagnosis'][:k] if norm(p)]


def recall_rows(ood):
    n = len(ood)
    row = {'n_OOG': n}
    for k in (1, 2, 3, 4):
        ex = sum(1.0 for it in ood
                 if any(norm(g) in topk_preds(it, k) for g in it['gt'])) / n
        rx = sum(1.0 for it in ood
                 if any(p in added for p in topk_preds(it, k))) / n
        row[f'exact@{k}'] = round(ex, 4)
        row[f'relaxed@{k}'] = round(rx, 4)
    return row


FILES = {
    'test': f'{P.PIPELINE_RESULTS_DIR}/exp6_hg35_hv30_bal_v3a_results.json',
    'valid': f'{P.PIPELINE_RESULTS_DIR}/exp6_hg35_hv30_validid_v3a_results.json',
}

rows = []
for split, f in FILES.items():
    r = recall_rows(load_ood(f))
    r = {'split': split, **r}
    rows.append(r)

df = pd.DataFrame(rows)
outdir = f'{P.FINAL_TABLES_DIR}'
os.makedirs(outdir, exist_ok=True)
out = f'{outdir}/relaxed_oog_recall.csv'
with open(out, 'w') as fp:
    fp.write('# KG-augmentation OOG recall, exact vs relaxed. relaxed@k = top-k contains any of the %d added diseases '
             '(the analogue of the "Other" flag). Model exp6_hg35_hv30_v3a; original=%d, augmented=%d.\n' % (len(added), len(paper), len(aug)))
    fp.write('# Computed on the full OOG sets (243 test / 128 valid); the paper reports the seed-9 prevalence-matched '
             'subsample (relaxed_oog_recall_seed9.csv).\n')
df.to_csv(out, mode='a', index=False)
print('added-190:', len(added), '| paper:', len(paper), '| aug_v3:', len(aug))
print(df.to_string(index=False))
print('\nsaved ->', out)
