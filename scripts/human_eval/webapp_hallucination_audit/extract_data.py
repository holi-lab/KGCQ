
#!/usr/bin/env python3
"""Read profile_hallucination_eval.xlsx (no openpyxl) -> data.json + strings_en.json.

xlsx is a zip: xl/sharedStrings.xml (string table) + xl/worksheets/sheet1.xml (cells).
Cols: A=Patient_profile (python-dict string, only on an item's first row), 
      B=Doctor utterance, C=Patient utterance. New item = non-empty col A.

Outputs (same dir):
  data.json        : [{ "id": i, "profile": {...}, "turns": [{"doctor","patient"}, ...] }]
  strings_en.json  : { sha1[:12] : english_string }  -- deduped, for Korean translation
"""
import os, re, ast, json, zipfile, hashlib
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "..", "..", "data", "human_eval", "hallucination_eval")  # eval_set.json, data.json, translations
XLSX = os.path.join(DATA_DIR, "profile_hallucination_eval.xlsx")
NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

# profile fields whose value is a ';'-delimited list (translate each piece separately)
LIST_FIELDS = {"present_illness_positive", "present_illness_negative",
               "medical_history", "family_medical_history", "social_history"}
# whole-string profile fields to translate
WHOLE_FIELDS = {"chiefcomplaint", "disease", "gender",
                "language_proficiency", "personality_type", "recall_level", "dazed_level"}
NO_TRANSLATE = {"patient_id", "age"}


def h(s):
    return hashlib.sha1(s.strip().encode("utf-8")).hexdigest()[:12]


def _si_text(si):
    # <si> may be plain <t> or rich text runs <r><t>...
    t = si.find("x:t", NS)
    if t is not None:
        return t.text or ""
    return "".join((r.find("x:t", NS).text or "")
                   for r in si.findall("x:r", NS) if r.find("x:t", NS) is not None)


def read_xlsx(path):
    with zipfile.ZipFile(path) as z:
        shared = [_si_text(si) for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("x:si", NS)]
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in sheet.find("x:sheetData", NS).findall("x:row", NS):
        cells = {}
        for c in row.findall("x:c", NS):
            col = re.match(r"[A-Z]+", c.get("r")).group()
            t = c.get("t")
            if t == "s":  # shared string
                v = c.find("x:v", NS)
                cells[col] = shared[int(v.text)] if v is not None else ""
            elif t == "inlineStr":
                it = c.find("x:is/x:t", NS)
                cells[col] = it.text if it is not None else ""
            else:  # str/number/etc
                v = c.find("x:v", NS)
                cells[col] = (v.text if v is not None else "")
        rows.append(cells)
    return rows


def main():
    rows = read_xlsx(XLSX)
    # header row -> column->name (expect A=Patient_profile, B=Doctor, C=Patient)
    header = rows[0]
    print("header:", header)
    body = rows[1:]

    items, cur = [], None
    for r in body:
        prof_raw = (r.get("A") or "").strip()
        if prof_raw:  # new item
            try:
                profile = ast.literal_eval(prof_raw)
            except Exception as e:
                profile = {"_parse_error": str(e), "_raw": prof_raw[:200]}
            cur = {"id": len(items), "profile": profile, "turns": []}
            items.append(cur)
        if cur is None:
            continue
        doctor = (r.get("B") or "").strip()
        patient = (r.get("C") or "").strip()
        if doctor or patient:
            cur["turns"].append({"doctor": doctor, "patient": patient})

    # collect unique english strings for translation
    strings = {}

    def add(s):
        s = (s or "").strip()
        if s:
            strings[h(s)] = s

    for it in items:
        p = it["profile"]
        for k, v in p.items():
            if k in NO_TRANSLATE or not isinstance(v, str):
                continue
            if k in LIST_FIELDS:
                for piece in v.split(";"):
                    add(piece)
            elif k in WHOLE_FIELDS:
                add(v)
        for t in it["turns"]:
            add(t["doctor"])
            add(t["patient"])

    with open(os.path.join(DATA_DIR, "data.json"), "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    with open(os.path.join(DATA_DIR, "strings_en.json"), "w", encoding="utf-8") as f:
        json.dump(dict(sorted(strings.items(), key=lambda kv: kv[1].lower())), f, ensure_ascii=False, indent=1)

    n_turns = sum(len(it["turns"]) for it in items)
    tcounts = [len(it["turns"]) for it in items]
    print(f"items={len(items)}  total_turns={n_turns}  turns/item min/avg/max="
          f"{min(tcounts)}/{n_turns/len(items):.1f}/{max(tcounts)}")
    print(f"unique strings to translate = {len(strings)}")


if __name__ == "__main__":
    main()
