"""end-to-end 파이프라인 sweep(HV ratio) → IG/OOG/all @1-4 CSV.
results_pipeline/{exp}_hg{HG}_hv{HV}_{validid|bal}_sweep_metrics.json 읽어 정리.
all@k = w_IG*ID@k + w_OOG*OOD@k  (실제 MIMIC prevalence 2.81:1, HG CSV와 동일 가중)
출력: writing/pipeline_sweep_{valid|test}.csv
사용: python make_pipeline_csv.py [valid|test]
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import json, csv, os, sys
RV2 = P.RV2
RES = f"{P.PIPELINE_RESULTS_DIR}"
W_IG, W_OOG = 15252 / 20672, 5420 / 20672           # 0.7378 / 0.2622
HG_OF = {'exp5': 30, 'exp6': 35}                     # validation-best HG (고정)
HVS = [0, 5, 10, 15, 20, 25, 30, 35]


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else 'valid'
    tag = 'validid' if split == 'valid' else 'bal'
    cols = ['exp', 'strategy', 'hg_ratio', 'hv_ratio', 'n_IG', 'n_OOG',
            'IG@1', 'IG@2', 'IG@3', 'IG@4', 'OOG@1', 'OOG@2', 'OOG@3', 'OOG@4',
            'all@1', 'all@2', 'all@3', 'all@4']
    out = f"{P.RESULTS_DIR}/pipeline_sweep_{split}.csv"
    n_done = 0
    with open(out, 'w', newline='', encoding='utf-8-sig') as fp:
        w = csv.writer(fp); w.writerow(cols)
        for exp, strat in [('exp5', 'abstention'), ('exp6', 'rescue')]:
            hg = HG_OF[exp]
            if split == 'test':   # paper 원본 시스템(SFT, 338-graph) IG 성능 참조행 (revision Table 4, 275 IG test)
                w.writerow([exp, 'paper (original SFT)', '-', '-', 275, '-',
                            0.25, 0.361, 0.394, 0.418, '', '', '', '', '', '', '', ''])
            for hv in HVS:
                f = f"{RES}/{exp}_hg{hg}_hv{hv}_{tag}_sweep_metrics.json"
                if not os.path.exists(f):
                    w.writerow([exp, strat, hg, hv, '(미완료)'] + [''] * 13); continue
                d = json.load(open(f)); n_done += 1
                ig = {k: d[f'ID_recall@{k}'] for k in (1, 2, 3, 4)}
                oo = {k: d[f'OOD_recall@{k}'] for k in (1, 2, 3, 4)}
                al = {k: W_IG * ig[k] + W_OOG * oo[k] for k in (1, 2, 3, 4)}
                w.writerow([exp, strat, hg, hv, d['n_ID'], d['n_OOD']]
                           + [round(ig[k], 4) for k in (1, 2, 3, 4)]
                           + [round(oo[k], 4) for k in (1, 2, 3, 4)]
                           + [round(al[k], 4) for k in (1, 2, 3, 4)])
            w.writerow([])
    print(f"  → {out}  ({n_done}/{len(HVS)*2} run 채움, 가중 {W_IG:.4f}:{W_OOG:.4f})")


if __name__ == '__main__':
    main()
