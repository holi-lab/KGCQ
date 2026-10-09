#!/usr/bin/env python3
"""전문가 평가 웹툴 (Flask) — 시뮬레이션 대화의 현실성/임상적 격차 평가.

의사가 이름으로 로그인 → 10개 대화(영어+한국어 병기)를 읽고 항목별 질문(A~D, 척도+서술)에
답함. 모든 입력은 즉시 results/<name>.json 에 저장되고, 재로그인 시 이어서 진행. 마지막에
전체 종합 질문(O1/O2).

실행:  python app.py --host 0.0.0.0 --port 8000
"""
import os, re, json, hashlib, argparse, datetime, threading
from flask import Flask, request, session, redirect, url_for, render_template, jsonify, abort

HERE = os.path.dirname(os.path.abspath(__file__))
ED = os.path.join(HERE, "..", "..", "..", "data", "human_eval", "dialogue_review")  # selected_dialogues.json + translations
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
_LOCK = threading.Lock()

# ---------- bilingual data ----------
EN = json.load(open(os.path.join(ED, "_unique_lines.json"), encoding="utf-8"))
KO = json.load(open(os.path.join(ED, "_ko_translations.json"), encoding="utf-8"))
KOMAP = {e: k for e, k in zip(EN, KO)}
SEL = json.load(open(os.path.join(ED, "selected_dialogues.json"), encoding="utf-8"))

STRAT_KO = {'ID_success': '그래프 내·진단 성공 (in-graph, correct)',
            'ID_fail': '그래프 내·진단 실패 (in-graph, wrong)',
            'OOD_abstention': '그래프 밖·abstention/Other (out-of-graph)',
            'OOD_rescue': '그래프 밖·rescue/증강KG (out-of-graph)'}


def bi(s):
    s = (s or "").strip()
    return {"en": s, "ko": KOMAP.get(s, "")}


def item_view(idx):
    o = SEL[idx]
    turns = [{"speaker": t["speaker"], **bi(t["text"])} for t in o["dialogue"]]
    sdx = bi(o["system_diagnosis_utterance"])          # 콤마로 이어진 EN/KO 진단
    en_list = [x.strip() for x in sdx["en"].split(",") if x.strip()]
    ko_list = [x.strip() for x in sdx["ko"].split(",")] if sdx["ko"] else []
    ranked = [{"rank": i + 1, "en": en, "ko": (ko_list[i] if i < len(ko_list) else "")}
              for i, en in enumerate(en_list)]
    en2ko = {d["en"].strip().lower(): d["ko"] for d in ranked}    # AI 진단 EN→KO 재활용
    chart = [{"en": dx, "ko": en2ko.get(dx.lower(), "")}
             for dx in dict.fromkeys(x.strip() for x in o["true_diagnosis_EHR"])]
    return {
        "id": idx, "item_no": o["item"],
        "turns": turns,
        "ranked_dx": ranked,
        "chart_answer": chart,
    }


# 질문 정의: (field, 유형, EN, KO)  유형 scale=1~5 / text=서술
QUESTIONS = [
    ("A1", "scale", "How closely do the patient's responses resemble a real patient? (1=clearly artificial, 5=indistinguishable)",
     "환자의 응답이 실제 환자와 얼마나 유사한가요? (1=명백히 인공적, 5=실제와 구별 불가)"),
    ("A2", "text", "Specifically, which parts led you to that rating?",
     "환자 응답에서 어떤 부분 때문에 그렇게 평가하셨는지 구체적으로 서술해 주세요."),
    ("B1", "scale", "How closely does the doctor-patient interaction resemble a real consultation? (1=very different, 5=very similar)",
     "의사와 환자의 상호작용(대화 흐름)이 실제 진료와 얼마나 유사한가요? (1=매우 다름, 5=매우 유사)"),
    ("B2", "text", "Specifically, which parts led you to that rating?",
     "상호작용에서 어떤 부분 때문에 그렇게 평가하셨는지 구체적으로 서술해 주세요."),
    ("C1", "scale", "When making an initial assessment in a real emergency department using only this conversation, how sufficient is it to reach a diagnosis? (1=not at all, 5=fully sufficient)",
     "실제 응급실에서 이 대화만으로 초진을 한다고 할 때, 진단하기에 얼마나 충분한가요? (1=전혀 불충분, 5=충분)"),
    ("C2", "text", "For that ER initial assessment, what is needed beyond the conversation to reach a diagnosis (focused exam, vitals, labs, imaging)?",
     "응급실 초진 상황에서, 확진까지 대화 외에 무엇이 더 필요한가요? (집중 진찰·활력징후·검사·영상 등) 구체적으로 서술해 주세요."),
]
QFIELDS = [q[0] for q in QUESTIONS]
OVERALL_Q = [
    ("O1", "Across all 10 dialogues, where does the system's dialogue differ most from real clinical conversations (recurring limitations)? Describe for the patient simulation and the doctor agent, focusing especially on weaknesses.",
     "10개 대화 전반에서, 이 시스템의 대화가 실제 임상 대화와 가장 크게 다른 지점(반복되는 한계)은 무엇인가요? 환자 시뮬레이션과 의사 에이전트로 나눠, 특히 약점 위주로 구체적으로 서술해 주세요."),
    ("O2", "For real clinical use, where could it be used as-is vs. where is a physician's intervention essential? What additional safeguards/steps (e.g., test linkage, triage, accountability) are needed for safe deployment?",
     "이 시스템을 실제 임상에 쓴다면, 어느 상황에서 그대로 쓸 수 있고 어디서 사람(의사)의 개입이 반드시 필요한가요? 안전한 실제 배포를 위해 추가로 필요한 안전장치·절차(예: 검사 연계, 응급도 분류, 책임 소재)는 무엇이라 보시나요?"),
]


# ---------- per-evaluator storage (atomic) ----------
def result_path(name):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:40] or "anon"
    tag = hashlib.sha1(name.strip().encode("utf-8")).hexdigest()[:8]
    return os.path.join(RESULTS_DIR, f"{safe}_{tag}.json")


def load_result(name):
    p = result_path(name)
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return {"evaluator": name, "specialty": "", "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "items": {}, "overall": {}}


def save_result(name, data):
    p = result_path(name)
    data["updated"] = datetime.datetime.now().isoformat(timespec="seconds")
    with _LOCK:
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)


def first_incomplete(name):
    res = load_result(name)
    for i in range(len(SEL)):
        if not res["items"].get(str(i), {}).get("done"):
            return i
    return len(SEL)  # all done → overall page


# ---------- app ----------
app = Flask(__name__)
app.secret_key = os.urandom(24)


def current():
    return session.get("evaluator")


@app.route("/")
def index():
    if not current():
        return render_template("login.html")
    nxt = first_incomplete(current())
    return redirect(url_for("overall") if nxt >= len(SEL) else url_for("item", idx=nxt))


@app.route("/login", methods=["POST"])
def login():
    name = (request.form.get("name") or "").strip()
    specialty = (request.form.get("specialty") or "").strip()
    if not name or not specialty:
        return render_template("login.html", error="이름과 전문 과를 입력하세요 / Please enter your name and specialty.")
    session["evaluator"] = name
    session["specialty"] = specialty
    res = load_result(name)
    res["specialty"] = specialty          # 로그인 시 전문 과 기록/갱신
    save_result(name, res)
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    session.pop("evaluator", None)
    return redirect(url_for("index"))


def _grid(res):
    return [{"i": i, "done": bool(res["items"].get(str(i), {}).get("done"))} for i in range(len(SEL))]


@app.route("/item/<int:idx>")
def item(idx):
    if not current():
        return redirect(url_for("index"))
    if idx < 0 or idx >= len(SEL):
        abort(404)
    res = load_result(current())
    saved = res["items"].get(str(idx), {"done": False, "answers": {}})
    done_count = sum(1 for i in range(len(SEL)) if res["items"].get(str(i), {}).get("done"))
    return render_template("evaluate.html", view=item_view(idx), saved=saved, idx=idx,
                           total=len(SEL), done_count=done_count, evaluator=current(),
                           grid=_grid(res), questions=QUESTIONS)


@app.route("/save", methods=["POST"])
def save():
    name = current()
    if not name:
        return jsonify(ok=False, error="not_logged_in"), 401
    p = request.get_json(force=True)
    idx, field, value = str(int(p["item"])), str(p["field"]), p.get("value", "")
    if field not in QFIELDS:
        return jsonify(ok=False, error="bad_field"), 400
    res = load_result(name)
    it = res["items"].setdefault(idx, {"done": False, "answers": {}})
    it["answers"][field] = value
    save_result(name, res)
    return jsonify(ok=True, at=res["updated"])


@app.route("/done/<int:idx>", methods=["POST"])
def done(idx):
    name = current()
    if not name:
        return jsonify(ok=False), 401
    res = load_result(name)
    res["items"].setdefault(str(idx), {"done": False, "answers": {}})["done"] = True
    save_result(name, res)
    nxt = idx + 1
    return jsonify(ok=True, next=nxt, overall=(nxt >= len(SEL)))


@app.route("/overall")
def overall():
    if not current():
        return redirect(url_for("index"))
    res = load_result(current())
    done_count = sum(1 for i in range(len(SEL)) if res["items"].get(str(i), {}).get("done"))
    return render_template("overall.html", saved=res.get("overall", {}), overall_q=OVERALL_Q,
                           total=len(SEL), done_count=done_count, evaluator=current(), grid=_grid(res))


@app.route("/save_overall", methods=["POST"])
def save_overall():
    name = current()
    if not name:
        return jsonify(ok=False), 401
    p = request.get_json(force=True)
    field = str(p["field"])
    if field not in [q[0] for q in OVERALL_Q]:
        return jsonify(ok=False), 400
    res = load_result(name)
    res.setdefault("overall", {})[field] = p.get("value", "")
    save_result(name, res)
    return jsonify(ok=True, at=res["updated"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    print(f"Serving {len(SEL)} dialogues; KO lines: {len(KOMAP)}")
    app.run(host=a.host, port=a.port, debug=a.debug)
