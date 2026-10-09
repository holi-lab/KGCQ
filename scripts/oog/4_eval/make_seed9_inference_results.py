"""Materialize the seed=9 prevalence-matched inference result files (the subsample
actually reported in HV_result.csv), and compute relaxed OOG recall on them.

Reproduced sampling (verified below): take the OOD cases, sort by str(hadm),
then np.random.default_rng(9).choice(n_OOD, n_matched, replace=False). ID cases
are kept in full (not subsampled). This reproduces the exact reported OOG recall:
  test  (243->98):  0.102 / 0.153 / 0.173 / 0.194
  valid (128->108): 0.134 / 0.171 / 0.227 / 0.227

Writes (per-case inference results for the reported seed=9 subsample):
  results_pipeline/exp6_hg35_hv30_{bal,validid}_v3a_seed9_results.json   (ID full + seed9 OOD)
  results_pipeline/exp6_hg35_hv30_{bal,validid}_v3a_seed9_metrics.json
  final result/relaxed_oog_recall_seed9.csv

relaxed@k = top-k contains ANY of the 190 added (out-of-original-graph) diseases,
the direct analogue of the "Other" flag; exact@k = top-k contains the gold disease.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import json, os
import numpy as np
import pandas as pd

RV2 = P.RV2
KG = str(P.KG_DIR)
RES = f'{P.PIPELINE_RESULTS_DIR}'


def norm(x):
    return str(x).strip().lower()


def diseases(p):
    df = pd.read_csv(p)
    lab = 'label' if 'label' in df.columns else df.columns[1]
    return set(norm(n) for n, l in zip(df['name'], df[lab]) if norm(l) == 'disease')


paper = diseases(f'{KG}/paper/nodes.csv')
aug = diseases(f'{KG}/augmented_v3/nodes.csv')
added = aug - paper                                        # 190


def load(f):
    R = json.load(open(f))
    return R if isinstance(R, list) else list(R.values())


def seed9_ood(ood, n_matched):
    arr = sorted(ood, key=lambda x: str(x['hadm']))         # sort by str(hadm)
    idx = np.random.default_rng(9).choice(len(arr), n_matched, replace=False)
    return [arr[i] for i in idx]


def topk(it, k):
    return [norm(p) for p in it['final_diagnosis'][:k] if norm(p)]


def recfield(sub, k):                                       # official per-case recall (matches reported)
    return sum(float(it['recall'][str(k)]) for it in sub) / len(sub)


def exact_bin(sub, k):                                      # binary: any gold disease in top-k
    return sum(1.0 for it in sub if any(norm(g) in topk(it, k) for g in it['gt'])) / len(sub)


def relaxed_bin(sub, k):                                    # binary: any added disease in top-k
    return sum(1.0 for it in sub if any(p in added for p in topk(it, k))) / len(sub)


CFG = [('test', 'bal', 98, [0.102, 0.153, 0.173, 0.194]),
       ('valid', 'validid', 108, [0.134, 0.171, 0.227, 0.227])]

rows = []
for split, tag, nmatch, target in CFG:
    items = load(f'{RES}/exp6_hg35_hv30_{tag}_v3a_results.json')
    idc = [it for it in items if it.get('kind') == 'ID']
    ood_all = [it for it in items if it.get('kind') == 'OOD']
    s9 = seed9_ood(ood_all, nmatch)

    rf = [round(recfield(s9, k), 3) for k in (1, 2, 3, 4)]
    assert rf == target, f'{split}: seed9 recall {rf} != reported {target}'

    # write per-case inference results for the reported seed=9 subsample (ID full + seed9 OOD)
    out_results = f'{RES}/exp6_hg35_hv30_{tag}_v3a_seed9_results.json'
    json.dump(idc + s9, open(out_results, 'w'))

    met = {'exp': 'exp6', 'hg_ratio': 35, 'hv_ratio': 30, 'variant': 'v3a', 'split': split,
           'seed': 9, 'sampling': 'OOD sorted by str(hadm); np.random.default_rng(9).choice(n_OOD, n_matched, replace=False); ID kept full',
           'n_ID': len(idc), 'n_OOD_full': len(ood_all), 'n_OOD': len(s9),
           **{f'OOD_recall@{k}': round(recfield(s9, k), 4) for k in (1, 2, 3, 4)},
           **{f'ID_recall@{k}': round(recfield(idc, k), 4) for k in (1, 2, 3, 4)}}
    json.dump(met, open(f'{RES}/exp6_hg35_hv30_{tag}_v3a_seed9_metrics.json', 'w'), indent=2)

    row = {'split': split, 'n_OOG': len(s9)}
    for k in (1, 2, 3, 4):
        row[f'exact@{k}'] = round(exact_bin(s9, k), 4)
        row[f'relaxed@{k}'] = round(relaxed_bin(s9, k), 4)
    rows.append(row)
    print(f'{split}: reproduced OOG recall {rf} == reported {target}  ->  wrote {os.path.basename(out_results)} '
          f'(n_ID={len(idc)}, n_OOD={len(s9)})')

df = pd.DataFrame(rows)
out = f'{P.FINAL_TABLES_DIR}/relaxed_oog_recall_seed9.csv'
with open(out, 'w') as fp:
    fp.write('# seed=9 prevalence-matched OOG subsample (98 test / 108 valid) -- the subsample reported in HV_result.csv.\n')
    fp.write('# sampling: OOD sorted by str(hadm); np.random.default_rng(9).choice(n_OOD, n_matched, replace=False); ID kept full.\n')
    fp.write('# exact@k = gold disease in top-k (binary); relaxed@k = ANY of 190 added diseases in top-k (= "Other" analogue). '
             'Model exp6_hg35_hv30_v3a. added=%d.\n' % len(added))
df.to_csv(out, mode='a', index=False)
print('\nadded-190:', len(added))
print(df.to_string(index=False))
print('\nsaved ->', out)
