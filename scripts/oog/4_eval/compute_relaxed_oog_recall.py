"""Criterion-matched OOG recall for the KG-augmentation (rescue) strategy.

Motivation: the ``Other''-node strategy is credited whenever the HV flags a case
as out-of-graph (answers ``Other''), i.e. it only has to RECOGNIZE OOG. The
KG-augmentation strategy is scored on EXACT disease naming, a strictly harder
task. To compare on a matched footing we also credit augmentation whenever its
top-k contains ANY node added by the expansion (the 190 diseases in
augmented that are absent from the paper 338-graph) --- the direct analogue
of answering ``Other''.

  exact@k   = frac. of OOG cases whose top-k contains the gold disease
  relaxed@k = frac. of OOG cases whose top-k contains ANY added-190 disease
              (= ``recognized as out-of-original-graph'', matched to ``Other'')

Source: final rescue model exp6_hg35_hv30_v3a (v3a = MIMIC-validated G+, HV
re-trained on inference-matched subgraphs). Uses the saved per-case predictions
on the FULL OOD sets (243 test / 128 valid) --- the REPORTABLE basis.

NOTE on the set: exact@4 here (0.181 test) matches the full-243 result in
ood_full_results.csv (0.179). It does NOT match HV_result.csv's 0.194, which is
a seed=9 prevalence-matched 98-subsample flagged in ood_adopted_seed9_INSPECTION.csv
as a post-hoc cherry-pick ("NOT for reporting; reportable = expectation/full-243").
Because relaxed@k is a mean over cases, its expectation over 98-subsamples equals
the full-243 value, so relaxed@4=0.490 is the reportable figure regardless of seed.
Reference (full-243, reportable): Other-node OOG@4=0.877 (flags OOG), GraphAug
exact OOG@4=0.179 (names exact disease), GraphAug relaxed OOG@4=0.490 (names any OOG disease).
Output: 4_eval/final result/relaxed_oog_recall.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import json, os
import pandas as pd

RV2 = P.RV2
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
# header comment for provenance
with open(out, 'w') as fp:
    fp.write('# KG-augmentation (rescue) OOG recall, exact vs criterion-matched (relaxed).\n')
    fp.write('# relaxed@k = top-k contains ANY of the 190 added (out-of-original-graph) diseases '
             '= analogue of the "Other" flag. Model: exp6_hg35_hv30_v3a. added=%d, paper=%d, aug_v3=%d.\n'
             % (len(added), len(paper), len(aug)))
    fp.write('# REPORTABLE basis = FULL OOD set (243 test / 128 valid), i.e. expectation over '
             'prevalence-matched subsamples; NOT the seed=9 98-subsample (cherry-pick, per ood_adopted_seed9_INSPECTION.csv).\n')
    fp.write('# Reference (full-243): Other-node OOG@4=0.877 (flags OOG); GraphAug exact OOG@4=0.179; GraphAug relaxed OOG@4=0.490.\n')
df.to_csv(out, mode='a', index=False)
print('added-190:', len(added), '| paper:', len(paper), '| aug_v3:', len(aug))
print(df.to_string(index=False))
print('\nsaved ->', out)
