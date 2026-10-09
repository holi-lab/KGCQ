"""results/*.json 집계 → 척도 평균(A1/B1/C1/D2) + 서술 덤프. rebuttal 보고용 수치 산출."""
import json, glob, os
from collections import defaultdict
HERE = os.path.dirname(os.path.abspath(__file__))
SCALE = ['A1', 'B1', 'C1']
TEXT = ['A2', 'B2', 'C2']

vals = defaultdict(list)        # field -> [int]
texts = defaultdict(list)       # field -> [(evaluator,item,text)]
overall = []
roster = []
files = glob.glob(os.path.join(HERE, "results", "*.json"))
for f in files:
    if f.endswith(".tmp"):
        continue
    d = json.load(open(f, encoding="utf-8"))
    ev = d.get("evaluator", os.path.basename(f))
    done = sum(1 for it in d.get("items", {}).values() if it.get("done"))
    roster.append((ev, d.get("specialty", ""), done))
    for idx, it in d.get("items", {}).items():
        for fld, v in it.get("answers", {}).items():
            if fld in SCALE and str(v).strip():
                try:
                    vals[fld].append(int(v))
                except ValueError:
                    pass
            elif fld in TEXT and str(v).strip():
                texts[fld].append((ev, int(idx) + 1, v.strip()))
    for fld, v in d.get("overall", {}).items():
        if str(v).strip():
            overall.append((ev, fld, v.strip()))

print(f"=== 평가자 {len(files)}명 집계 ===")
for ev, sp, dn in roster:
    print(f"  · {ev} ({sp or '전문과 미기입'}) — 완료 {dn}/10")
print()
print("[척도 평균 1–5]")
LBL = {'A1': '환자 응답 유사성', 'B1': '상호작용 유사성', 'C1': '대화만으로의 충분성'}
for fld in SCALE:
    if vals[fld]:
        v = vals[fld]
        print(f"  {fld} {LBL[fld]:14s}: mean={sum(v)/len(v):.2f}  (n={len(v)})  {sorted(v)}")
    else:
        print(f"  {fld} {LBL[fld]:14s}: (응답 없음)")

print("\n[서술 응답 수]")
for fld in TEXT:
    print(f"  {fld}: {len(texts[fld])}개")
print(f"  Overall(O1/O2): {len(overall)}개")

out = os.path.join(HERE, "aggregate_dump.json")
json.dump({"scale_means": {f: (sum(vals[f]) / len(vals[f]) if vals[f] else None) for f in SCALE},
           "scale_raw": {f: vals[f] for f in SCALE},
           "text": {f: texts[f] for f in TEXT}, "overall": overall},
          open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\n덤프: {out}")
