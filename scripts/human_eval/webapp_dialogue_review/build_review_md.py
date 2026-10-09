"""선별 대화 10개를 EN (KO) 병기 MD로 생성 (전문가 평가용).
- review/item_01.md ... item_10.md : 대화(영+한) + 시스템진단 + 차트정답 + 질문
- review/00_instructions.md : 안내 + 전체 질문
"""
import json, os
import os as _os; RV2 = _os.environ.get("KGCQ_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..")))  # repository root
ED = f"{RV2}/expert_eval"
OUT = f"{ED}/review"
os.makedirs(OUT, exist_ok=True)

en = json.load(open(f"{ED}/_unique_lines.json"))
ko = json.load(open(f"{ED}/_ko_translations.json"))
assert len(en) == len(ko), f"EN {len(en)} != KO {len(ko)}"
KO = {e: k for e, k in zip(en, ko)}
items = json.load(open(f"{ED}/selected_dialogues.json"))

STRAT_KO = {'ID_success': '그래프 내·진단 성공', 'ID_fail': '그래프 내·진단 실패',
            'OOD_abstention': '그래프 밖·abstention(Other)', 'OOD_rescue': '그래프 밖·rescue(증강KG)'}

PER_ITEM_Q = """\
---
### 평가 질문 (이 대화를 모두 읽고 답해주세요 / Answer after reading the whole dialogue)

**A. 환자 현실성 / Patient realism**
- **A1 [1–5]:** 같은 문제를 호소하는 실제 환자와 얼마나 닮았나요? (1=명백히 인공적, 5=실제와 구별 불가)
  How closely does this simulated patient resemble a real patient? (1=clearly artificial, 5=indistinguishable)
  → **답(A1): ___**
- **A2 [서술]:** 무엇이 현실적/인공적이었나요? (모호함·감정·자발적 진술이나 누락·말투)
  What felt realistic vs. artificial? →

**B. 의사 질문의 질 / Clinician-agent question quality**
- **B1 [1–5]:** 의사의 질문이 임상적으로 적절하고 효율적인 순서였나요? → **답(B1): ___**
- **B2 [서술]:** 놓친 중요한 질문(red-flag/can't-miss 증상, 과거력·약물·가족력·사회력)이나 중복·부자연스러운 질문이 있었나요?
  Missing red-flag/history questions, or redundant/unnatural ones? →

**C. 진단 타당성 / Diagnostic plausibility**
- **C1 [1–5]:** 이 대화만으로 볼 때 시스템의 최종 감별진단이 임상적으로 타당한가요? → **답(C1): ___**
- **C2 [서술]:** 추가/삭제/순위변경할 진단과 그 이유는? (아래 차트 정답 대비) →

**D. 실제 격차 / Real-world gap**
- **D1 [서술]:** 실제 환자가 이렇게 왔다면 확진까지 대화 외에 무엇이 더 필요한가요? (집중 진찰·활력징후·검사·영상) 대화만으로는 얼마나 제한되나요? →
- **D2 [1–5]:** 이 대화 진행대로면 실제 환자에게 임상적으로 안전한 결정에 이를 가능성은? → **답(D2): ___**
"""


def line_md(turn):
    who = "**의사 (Doctor)**" if turn['speaker'] == 'doctor' else "**환자 (Patient)**"
    t = turn['text']
    k = KO.get(t, '')
    return f"- {who}: {t}  \n  ({k})"


for o in items:
    h = o['hadm']
    gt = ", ".join(o['true_diagnosis_EHR'])
    sysdx = o['system_diagnosis_utterance']
    sysdx_ko = KO.get(sysdx, '')
    md = []
    md.append(f"# 전문가 평가 — 대화 #{o['item']:02d}  (Expert Review Item {o['item']:02d})\n")
    md.append("> 아래 의사–환자 대화를 모두 읽고 맨 아래 질문에 답해주세요. 각 발화는 **영어 원문 (한국어 번역)** 으로 제시됩니다.\n")
    md.append("| 항목 | 내용 |")
    md.append("|---|---|")
    md.append(f"| 전략 (strategy) | {o['strategy']} |")
    md.append(f"| 유형 (stratum) | {STRAT_KO.get(o['stratum'], o['stratum'])} |")
    md.append(f"| 환자 발화 수 | {o['n_patient_turns']} turns |")
    md.append("\n## 대화 (Dialogue)\n")
    for turn in o['dialogue']:
        md.append(line_md(turn))
    md.append("\n## 시스템 최종 진단 (System's final differential)\n")
    md.append(f"- {sysdx}  \n  ({sysdx_ko})\n")
    md.append("> ⚠️ **차트 정답(실제 EHR 진단)** — 진단 타당성(C) 평가 시 참고: "
              f"**{gt}**\n")
    md.append(PER_ITEM_Q)
    open(f"{OUT}/item_{o['item']:02d}.md", 'w', encoding='utf-8').write("\n".join(md))

# 안내 + 전체 질문
intro = """\
# 전문가 평가 안내 (Expert Evaluation — Instructions)

본 평가는 본 시스템이 생성한 **실제 추론 대화**(의사 에이전트 ↔ 환자 시뮬레이터)를 임상의가 읽고,
**시뮬레이션 대화와 실제 임상 대화의 차이(gap)** 를 평가하기 위한 것입니다.
This evaluation asks practicing clinicians to read the system's **actual inference dialogues**
and assess how the patient simulation and the clinician-agent's questioning differ from real clinical conversations.

## 진행 방법 (How to proceed)
- `item_01.md` ~ `item_10.md` 의 10개 대화를 각각 읽고, 각 파일 하단의 질문에 답해주세요.
- 각 발화는 **영어 원문 (한국어 번역)** 으로 제시됩니다.
- 1–5 척도 문항과 서술형 문항이 함께 있습니다. 척도는 정량 보고용, 서술은 뉘앙스·약점 파악용입니다.

## 평가 축 (리뷰어 요청 기반)
- **A. 환자 현실성** — 합성 환자가 실제 환자처럼 보이는가 (분포 편향·실제 일반화 우려에 대응)
- **B. 의사 질문의 질** — red-flag·과거력 탐색 등 임상적 적절성 (과소탐색 지적에 대응)
- **C. 진단 타당성** — 후보 진단의 임상적 합리성 (타당성 낮은 진단 지적에 대응)
- **D. 실제 격차** — 대화 외에 필요한 것(진찰·검사 등)과 안전성

## 전체 종합 질문 (모든 대화 평가 후 1회만)
- **O1 [서술]:** 10개 대화 전반에서 (a) 환자 시뮬레이션 / (b) 의사 에이전트 각각의 **반복되는 강점과 약점**은?
- **O2 [서술]:** 이 시스템을 **그대로 쓸 수 있는 영역 vs human-in-the-loop가 필요한 영역**은? 실제 배포의 **가장 큰 장벽**은?

> 평가자: ____________   날짜: __________
"""
open(f"{OUT}/00_instructions.md", 'w', encoding='utf-8').write(intro)

print(f"생성 완료 → {OUT}/")
for f in sorted(os.listdir(OUT)):
    print("  ", f)
