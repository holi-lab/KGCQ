"""전문가 평가용 '현실적인 환자' 대화 선별.
- 현실성 휴리스틱: hedging/불확실성, 1인칭 감정·구어, 환자 발화 적정 길이, 적정 턴수.
- 층화: ID성공 / ID실패 / OOD(abstention exp5) / OOD(rescue exp6), 주호소 다양화.
출력: 후보를 점수순으로 출력 (수동 검토 후 최종 선정).
"""
import json, re, glob
from collections import defaultdict
import os as _os; RV2 = _os.environ.get("KGCQ_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..")))  # repository root

HEDGE = ['um', 'uh', 'maybe', 'i think', 'i guess', 'not sure', "don't know", 'kind of',
         'a little', 'sort of', 'probably', 'i suppose', 'i feel like', 'hard to', 'weird']
EMO = ['worried', 'scared', 'uncomfortable', 'hurts', 'hurt', 'tired', 'really', 'feel',
       'bad', 'awful', 'terrible', 'a lot', 'bothering', "can't"]


def patient_utts(dialogue):
    return [m['content'] for m in dialogue if m['role'] == 'user']


def realism_score(dialogue):
    pus = patient_utts(dialogue)
    if not pus:
        return -1, {}
    text = " ".join(pus).lower()
    words = text.split()
    nw = len(words)
    n_turns = len(pus)
    hedge = sum(text.count(h) for h in HEDGE)
    emo = sum(text.count(e) for e in EMO)
    avg_len = nw / n_turns
    # 한 단어/로봇식 답 비율(짧을수록 비현실)
    short_ratio = sum(1 for u in pus if len(u.split()) <= 3) / n_turns
    # 점수: hedge·emo 밀도(현실) + 적정 길이/턴 - 과도한 단답
    score = (hedge / nw * 100) * 1.0 + (emo / nw * 100) * 0.6
    score += 1.0 if 4 <= n_turns <= 9 else -1.0
    score += 1.0 if 7 <= avg_len <= 22 else -0.5
    score -= short_ratio * 2.0
    return round(score, 2), dict(n_turns=n_turns, avg_len=round(avg_len, 1),
                                 hedge=hedge, emo=emo, short_ratio=round(short_ratio, 2), nwords=nw)


SRC = {
    'exp5_abstention': f"{RV2}/4_eval/results_pipeline/exp5_hg30_hv25_validid_sweep_results.json",
    'exp6_rescue': f"{RV2}/4_eval/results_pipeline/exp6_hg35_hv5_validid_sweep_results.json",
}

buckets = defaultdict(list)   # stratum -> [(score, info, case, src)]
for src, f in SRC.items():
    d = json.load(open(f))
    for c in d:
        if 'dialogue' not in c:
            continue
        s, info = realism_score(c['dialogue'])
        if s < 0:
            continue
        if c['kind'] == 'ID':
            strat = 'ID_success' if c['recall']['4'] >= 0.99 else 'ID_fail'
        else:
            strat = f'OOD_{src.split("_")[1]}'   # abstention / rescue
        buckets[strat].append((s, info, c, src))

print("=== 층화별 현실성 상위 후보 (점수순) ===")
for strat in ['ID_success', 'ID_fail', 'OOD_abstention', 'OOD_rescue']:
    cand = sorted(buckets.get(strat, []), key=lambda x: -x[0])
    print(f"\n##### {strat} (n={len(cand)}) — 상위 5 #####")
    for s, info, c, src in cand[:5]:
        cc = (c['dialogue'][1]['content'][:55]) if len(c['dialogue']) > 1 else ''
        print(f"  score={s:5.2f} | hadm{c['hadm']} t={info['n_turns']} avglen={info['avg_len']} "
              f"hedge={info['hedge']} | gt={c['gt']} | 첫호소: {cc}")
