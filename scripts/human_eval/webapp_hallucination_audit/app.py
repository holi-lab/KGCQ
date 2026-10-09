#!/usr/bin/env python3
"""Hallucination evaluation web tool (Flask).

Physicians log in with a name, read each patient profile + doctor-patient dialogue
(English + Korean in parentheses), and for every PATIENT response tick a checkbox
("hallucination occurred") and optionally add a memo. Every action auto-saves
immediately to results/<name>.json; progress resumes on re-login.

Run:  python app.py --host 0.0.0.0 --port 8000
"""
import os, re, json, ast, hashlib, argparse, datetime, threading
from flask import Flask, request, session, redirect, url_for, render_template, jsonify, abort

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "human_eval", "hallucination_eval")  # eval_set.json, data.json, translations
# Use the curated evaluation set (eval_set.json) if present, else the full 72 (data.json).
_EVAL = os.path.join(DATA_DIR, "eval_set.json")
DATA = json.load(open(_EVAL if os.path.exists(_EVAL) else os.path.join(DATA_DIR, "data.json"), encoding="utf-8"))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
_LOCK = threading.Lock()


def load_translations():
    p = os.path.join(DATA_DIR, "translations_ko.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


TRANS = load_translations()


def h(s):
    return hashlib.sha1((s or "").strip().encode("utf-8")).hexdigest()[:12]


def bi(s):
    """English + Korean lookup -> {'en':.., 'ko':..} (ko='' if not translated yet)."""
    s = (s or "").strip()
    return {"en": s, "ko": TRANS.get(h(s), "")} if s else {"en": "", "ko": ""}


# profile field rendering order: (key, EN label, KO label, kind)
PROFILE_FIELDS = [
    ("patient_id", "Patient ID", "환자 ID", "plain"),
    ("age", "Age", "나이", "plain"),
    ("gender", "Gender", "성별", "bi"),
    ("chiefcomplaint", "Chief complaint", "주호소", "bi"),
    ("present_illness_positive", "Present illness (positive)", "현병력(양성)", "list"),
    ("present_illness_negative", "Present illness (denied)", "현병력(음성)", "list"),
    ("medical_history", "Past medical history", "과거력", "list"),
    ("family_medical_history", "Family history", "가족력", "list"),
    ("social_history", "Social history", "사회력", "list"),
    ("disease", "Ground-truth disease", "정답 질병", "bi"),
    ("language_proficiency", "Language proficiency", "언어 능숙도", "bi"),
    ("personality_type", "Personality", "성격 유형", "bi"),
    ("recall_level", "Recall level", "기억 수준", "bi"),
    ("dazed_level", "Confusion level", "혼란 수준", "bi"),
]


def render_profile(profile):
    out = []
    for key, en_lab, ko_lab, kind in PROFILE_FIELDS:
        v = profile.get(key)
        if v is None or (isinstance(v, str) and not v.strip()):
            continue
        if kind == "plain":
            val = [{"en": str(v), "ko": ""}]
        elif kind == "bi":
            val = [bi(str(v))]
        else:  # list
            val = [bi(p) for p in str(v).split(";") if p.strip()]
        out.append({"label_en": en_lab, "label_ko": ko_lab, "vals": val, "is_list": kind == "list"})
    return out


def item_view(idx):
    it = DATA[idx]
    turns = []
    for t in it["turns"]:
        turns.append({"doctor": bi(t["doctor"]), "patient": bi(t["patient"])})
    return {"id": idx, "profile": render_profile(it["profile"]), "turns": turns}


# ---------- per-evaluator result storage (atomic) ----------
def result_path(name):
    # hash suffix guarantees uniqueness even when non-ASCII names sanitize to the same string
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:40] or "anon"
    tag = hashlib.sha1(name.strip().encode("utf-8")).hexdigest()[:8]
    return os.path.join(RESULTS_DIR, f"{safe}_{tag}.json")


def load_result(name):
    p = result_path(name)
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return {"evaluator": name, "created": datetime.datetime.now().isoformat(timespec="seconds"), "items": {}}


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
    for i in range(len(DATA)):
        if not res["items"].get(str(i), {}).get("done"):
            return i
    return 0


# ---------- app ----------
app = Flask(__name__)
app.secret_key = os.urandom(24)


def current():
    return session.get("evaluator")


@app.route("/")
def index():
    if not current():
        return render_template("login.html")
    return redirect(url_for("item", idx=first_incomplete(current())))


@app.route("/login", methods=["POST"])
def login():
    name = (request.form.get("name") or "").strip()
    if not name:
        return render_template("login.html", error="이름을 입력하세요 / Please enter your name.")
    session["evaluator"] = name
    load_result(name)  # ensure file exists
    save_result(name, load_result(name))
    return redirect(url_for("item", idx=first_incomplete(name)))


@app.route("/logout")
def logout():
    session.pop("evaluator", None)
    return redirect(url_for("index"))


@app.route("/item/<int:idx>")
def item(idx):
    if not current():
        return redirect(url_for("index"))
    if idx < 0 or idx >= len(DATA):
        abort(404)
    res = load_result(current())
    saved = res["items"].get(str(idx), {"done": False, "turns": {}})
    done_count = sum(1 for i in range(len(DATA)) if res["items"].get(str(i), {}).get("done"))
    grid = [{"i": i, "done": bool(res["items"].get(str(i), {}).get("done")),
             "flagged": any(tt.get("hallucination") for tt in res["items"].get(str(i), {}).get("turns", {}).values())}
            for i in range(len(DATA))]
    return render_template("evaluate.html", view=item_view(idx), saved=saved, idx=idx,
                           total=len(DATA), done_count=done_count, evaluator=current(), grid=grid)


@app.route("/save", methods=["POST"])
def save():
    name = current()
    if not name:
        return jsonify(ok=False, error="not_logged_in"), 401
    p = request.get_json(force=True)
    idx, turn = str(int(p["item"])), str(int(p["turn"]))
    res = load_result(name)
    item = res["items"].setdefault(idx, {"done": False, "turns": {}})
    tt = item["turns"].setdefault(turn, {"hallucination": False, "memo": ""})
    if "hallucination" in p:
        tt["hallucination"] = bool(p["hallucination"])
    if "memo" in p:
        tt["memo"] = str(p["memo"])
    save_result(name, res)
    return jsonify(ok=True, at=res["updated"])


@app.route("/done/<int:idx>", methods=["POST"])
def done(idx):
    name = current()
    if not name:
        return jsonify(ok=False), 401
    res = load_result(name)
    res["items"].setdefault(str(idx), {"done": False, "turns": {}})["done"] = True
    save_result(name, res)
    nxt = idx + 1 if idx + 1 < len(DATA) else idx
    return jsonify(ok=True, next=nxt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    print(f"Serving {len(DATA)} items; translations loaded: {len(TRANS)}")
    app.run(host=a.host, port=a.port, debug=a.debug)
