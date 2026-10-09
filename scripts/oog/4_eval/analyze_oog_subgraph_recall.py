"""OOG 분석: (1) OOG에 추가된 질병들이 실제 희귀질병인지 MIMIC 빈도로 확인,
(2) 최종 진단 직전 '서브그래프'에 정답 OOG 질병이 있는지 = subgraph recall.
   비교: HG top_k recall < subgraph recall < end-to-end final recall 의 어디서 병목인지.
사용: python analyze_oog_subgraph_recall.py
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import json, sys, glob, re
from collections import Counter
RV2 = P.RV2

# ---------- (1) OOG 질병 희귀도 ----------
ood = json.load(open(f"{P.PROFILE_DIR}/ood_valid_128.json"))
oog_dx = []
for h, p in ood.items():
    for d in (p.get('disease_mapped') or []):
        oog_dx.append(str(d).strip().lower())
dx_in_eval = Counter(oog_dx)

# MIMIC 전체 빈도: 5420 OOG 풀 + 전체 disease-labeled 에서 각 질병 prevalence
def disease_freq_from(path_globs):
    cnt = Counter()
    for g in path_globs:
        for f in glob.glob(g):
            try:
                d = json.load(open(f))
            except Exception:
                continue
            items = d.values() if isinstance(d, dict) else d
            for it in items:
                if isinstance(it, dict):
                    for dd in (it.get('disease_mapped') or it.get('disease') or []):
                        if isinstance(dd, str):
                            cnt[dd.strip().lower()] += 1
    return cnt

# OOG 5420 풀 전체에서 빈도
pool_freq = disease_freq_from([
    f"{P.PROFILE_DIR}/ood_*.json",
])

print("=" * 70)
print("(1) OOG 평가셋(valid128)에 등장하는 질병 — 종류/빈도/희귀도")
print(f"    valid128: {len(ood)} 프로필, 고유 OOG 질병 {len(dx_in_eval)} 종")
print("=" * 70)
print(f"  {'질병':40s} {'eval건수':>7s} {'OOG풀빈도':>8s}")
for dx, c in dx_in_eval.most_common():
    print(f"  {dx:40s} {c:>7d} {pool_freq.get(dx,0):>8d}")

# ---------- (2) subgraph recall ----------
def name_in_lines(name, lines):
    n = name.lower().strip()
    # 괄호 약어 등 변형 고려: 핵심 단어 매칭 + 전체 문자열 매칭
    joined = " \n ".join(lines).lower()
    if n in joined:
        return True
    # copd 등 약어 처리: gt에 괄호 약어 있으면 약어도 체크
    m = re.findall(r'\(([^)]+)\)', n)
    return any(a.strip() in joined for a in m if len(a.strip()) > 2)

print("\n" + "=" * 70)
print("(2) subgraph recall — 최종 진단 직전 서브그래프에 정답 OOG 질병 포함 여부")
print("    (HG top_k recall vs subgraph recall vs end-to-end final recall)")
print("=" * 70)
print(f"  {'config':22s} {'top_k@hit':>10s} {'subgraph':>9s} {'final@4':>8s}  (n)")
for f in sorted(glob.glob(f"{P.PIPELINE_RESULTS_DIR}/exp6_hg35_hv*_validid_sweep_results.json")):
    hv = re.search(r'hv(\d+)', f).group(1)
    d = json.load(open(f))
    oogc = [c for c in d if c['kind'] == 'OOD' and 'subgraph_lines' in c]
    if not oogc:
        continue
    n = len(oogc)
    topk_hit = sum(1 for c in oogc
                   if any(str(g).lower() in [str(x).lower() for x in c.get('top_k', [])] for g in c['gt']))
    sg_hit = sum(1 for c in oogc
                 if any(name_in_lines(str(g), c['subgraph_lines']) for g in c['gt']))
    final_hit = sum(1 for c in oogc if c['recall']['4'] > 0)
    print(f"  exp6_hv{hv:<17s} {topk_hit/n:>10.3f} {sg_hit/n:>9.3f} {final_hit/n:>8.3f}  ({n})")

# exp5(abstention)도 같은 방식: gt='other' → subgraph엔 OTHER_NOTE
print()
for f in sorted(glob.glob(f"{P.PIPELINE_RESULTS_DIR}/exp5_hg30_hv*_validid_sweep_results.json")):
    hv = re.search(r'hv(\d+)', f).group(1)
    d = json.load(open(f))
    oogc = [c for c in d if c['kind'] == 'OOD' and 'subgraph_lines' in c]
    if not oogc:
        continue
    n = len(oogc)
    # exp5: 정답='other' → top_k에 'other'? subgraph에 Other note?
    topk_hit = sum(1 for c in oogc if any('other' == str(x).lower() for x in c.get('top_k', [])))
    sg_hit = sum(1 for c in oogc if any('other' in l.lower() for l in c['subgraph_lines']))
    final_hit = sum(1 for c in oogc if c['recall']['4'] > 0)
    print(f"  exp5_hv{hv:<17s} {topk_hit/n:>10.3f} {sg_hit/n:>9.3f} {final_hit/n:>8.3f}  ({n})  [정답=Other]")
