"""exp6 OOG를 exp5와 동일선상에서 비교하기 위한 'recognition recall'.
exp5: 의사가 'Other' 라고 하면 OOG 인식 성공.
exp6(여기): 의사가 augmented-only 질병(528∖338, 원래 KG 밖 질병)으로 진단하면 OOG 인식 성공.
  → final_diagnosis[:k] 에 (528∖338) 질병이 하나라도 있으면 hit.
출력: writing/exp6_recognition_recall.csv + exp5 OOG와 비교 표.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import json, csv, glob, re, sys
import pandas as pd
RV2 = P.RV2


def dlist(csvp):
    df = pd.read_csv(csvp)
    return set(str(r['name']).strip().lower() for _, r in df.iterrows() if r['label'] == 'Disease')


paper = dlist(f"{P.KG_DIR}/paper/nodes.csv")          # 338
aug = dlist(f"{P.KG_DIR}/augmented/nodes.csv")        # 528
aug_only = aug - paper                                    # 528∖338 = 원래 그래프 밖(rescued) 질병
print(f"paper={len(paper)}  aug={len(aug)}  aug_only(528∖338)={len(aug_only)}")
assert paper <= aug, "paper ⊄ aug (검증 실패)"


def hit_at_k(final, k, target_set):
    pred = [str(x).strip().lower() for x in final[:k] if str(x).strip()]
    return 1.0 if (set(pred) & target_set) else 0.0


rows = []
for f in sorted(glob.glob(f"{P.PIPELINE_RESULTS_DIR}/exp6_hg35_hv*_validid_sweep_results.json")):
    hv = int(re.search(r'hv(\d+)', f).group(1))
    d = json.load(open(f))
    oog = [c for c in d if c['kind'] == 'OOD']
    n = len(oog)
    # recognition recall@k = final 에 aug-only 질병 포함
    rec = {k: sum(hit_at_k(c['final_diagnosis'], k, aug_only) for c in oog) / n for k in (1, 2, 3, 4)}
    # 참고용 strict(정확 gold) @4
    strict4 = sum(c['recall']['4'] for c in oog) / n
    rows.append((hv, n, rec, strict4))

rows.sort()
out = f"{P.RESULTS_DIR}/exp6_recognition_recall.csv"
with open(out, 'w', newline='', encoding='utf-8-sig') as fp:
    w = csv.writer(fp)
    w.writerow(['exp6_HV', 'n_OOG', 'recogR@1', 'recogR@2', 'recogR@3', 'recogR@4', 'strict_OOG@4'])
    for hv, n, rec, s4 in rows:
        w.writerow([hv, n] + [round(rec[k], 4) for k in (1, 2, 3, 4)] + [round(s4, 4)])
print(f"  → {out}")

# exp5 OOG(=Other recall) 로드해서 비교
print("\n=== exp5(Other-recall) vs exp6(augmented-recognition-recall) — OOG 인식 동일선상 비교 ===")
print(f"  {'HV':>3} | {'exp5 OOG@4':>10} | {'exp6 recog@4':>12} | {'exp6 strict@4':>13}")
for hv, n, rec, s4 in rows:
    m5 = f"{P.PIPELINE_RESULTS_DIR}/exp5_hg30_hv{hv}_validid_sweep_metrics.json"
    try:
        e5 = json.load(open(m5))['OOD_recall@4']
    except Exception:
        e5 = None
    e5s = f"{e5:.3f}" if e5 is not None else "  -  "
    print(f"  {hv:>3} | {e5s:>10} | {rec[4]:>12.3f} | {s4:>13.3f}")
