#!/usr/bin/env python3
"""
VALIDATE -> PRUNE -> COMPACT the augmented KG's OOG-disease symptom edges against
real MIMIC-IV HPI notes.

Goal (refined): NOT to expand the KG with every note symptom, but to use real
notes as a REFERENCE to keep the augmented OOG symptom edges COMPACT and close to
real data. Primary result = per-disease KEEP (validated in notes) vs DROP
(not found = likely hallucinated). Then report the compaction effect of rebuilding
OOG symptom edges as "validated KG symptoms, capped ~8".

Consumes the already-computed edge-level validation:
  data/KG/kg_symptom_validation.json   (has kg_symptoms / validated / not_found per OOG disease,
                                         plus n_profiles / n_notes / n_profiles_train_only and the
                                         raw whole-vocab grounded lists)

Produces (pruning-first framing):
  data/KG/kg_prune_plan.json            (per disease: KEEP / DROP edges + capped grounded)
  4_eval/kg_prune_summary.csv           (per-disease compaction table)
and prints the headline compaction / coverage / validation numbers.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import os, re, json, csv, statistics
from collections import Counter

ROOT = P.RV2
VAL_JSON = str(P.KG_V3_DIR / "kg_symptom_validation.json")
OUT_JSON = str(P.KG_V3_DIR / "kg_prune_plan.json")
OUT_CSV  = str(P.RESULTS_DIR / "kg_prune_summary.csv")

CAP = 8  # paper-KG per-disease symptom level (paper median 6, mean 7.8) -> cap ~8

# Canonical buckets to collapse the whole-vocab grounded matcher's collisions
# (e.g. "massive hemoptysis with fever" and "fever due to infection" both really
# mean FEVER). Order matters: first matching bucket wins. Anything not bucketed
# keeps its own (short) name. This is ONLY used for the SHORT secondary
# "note symptoms the KG missed" list, not for the primary KEEP/DROP decision.
CANON = [
    ("dyspnea",            [r"dyspn", r"shortness of breath", r"\bsob\b", r"breathless"]),
    ("chest pain",         [r"chest pain", r"chest discomfort", r"chest pressure", r"chest tightness"]),
    ("fever",              [r"fever", r"febrile", r"pyrexia"]),
    ("abdominal pain",     [r"abdominal pain", r"epigastric pain", r"belly pain", r"stomach pain"]),
    ("edema",              [r"\bedema\b", r"swelling", r"swollen"]),
    ("nausea",             [r"nausea", r"nauseous"]),
    ("vomiting",           [r"vomit", r"emesis"]),
    ("cough",              [r"cough"]),
    ("fatigue",            [r"fatigue", r"malaise", r"lethargy", r"weakness", r"tired"]),
    ("headache",           [r"headache", r"cephalgia"]),
    ("palpitations",       [r"palpitation"]),
    ("dizziness",          [r"dizz", r"lightheaded", r"vertigo"]),
    ("syncope",            [r"syncope", r"syncopal", r"fainting", r"passed out"]),
    ("hemoptysis",         [r"hemoptysis"]),
    ("diarrhea",           [r"diarrhea", r"loose stool"]),
    ("constipation",       [r"constipat"]),
    ("weight loss",        [r"weight loss", r"losing weight"]),
    ("back pain",          [r"back pain", r"\blbp\b"]),
    ("flank pain",         [r"flank pain"]),
    ("hematuria",          [r"hematuria", r"blood in .*urine"]),
    ("dysuria",            [r"dysuria", r"burning.*urin"]),
    ("jaundice",           [r"jaundice", r"icter"]),
    ("confusion",          [r"confus", r"altered mental", r"\bams\b", r"disorient"]),
    ("seizure",            [r"seizure", r"convuls"]),
    ("rash",               [r"rash", r"eruption"]),
    ("anemia",             [r"anemia", r"anemic"]),
    ("chills",             [r"chills", r"rigors"]),
    ("leg pain",           [r"leg pain", r"calf pain"]),
    ("vision loss",        [r"vision.*(loss|blur|change)", r"visual.*(loss|change)"]),
]
CANON = [(name, [re.compile(p, re.I) for p in pats]) for name, pats in CANON]
CANON_NAMES = {name for name, _ in CANON}

# Ambient/non-specific symptoms that appear in almost every HPI note; as a
# "symptom the KG missed" they are noise, so they are excluded from the short
# secondary list unless a disease's KG genuinely centers on them.
AMBIENT = {"fever", "chills", "dyspnea", "cough", "fatigue", "nausea",
           "vomiting", "chest pain", "abdominal pain", "edema"}


def canon_bucket(name):
    for cname, pats in CANON:
        for p in pats:
            if p.search(name):
                return cname
    return name


def main():
    val = json.load(open(VAL_JSON))
    diseases = sorted(val.keys())

    # ---- background bucket prevalence across ALL OOG diseases (for lift) ----
    # For each canonical bucket, fraction of OOG diseases whose patients mention it
    # (using per-disease max grounded freq as "present"). Ambient symptoms have
    # high background prevalence; disease-characteristic ones are rare -> high lift.
    bg_present = Counter()
    n_dis_with_notes = 0
    for d in diseases:
        r = val[d]
        if r["n_notes"] == 0:
            continue
        n_dis_with_notes += 1
        buckets = set()
        for g in r["real_grounded_symptoms"]:
            b = canon_bucket(g["name"])
            if b in CANON_NAMES and g["freq"] >= max(3, int(0.10 * r["n_notes"])):
                buckets.add(b)
        for b in buckets:
            bg_present[b] += 1
    bg_rate = {b: bg_present[b] / n_dis_with_notes for b in CANON_NAMES}

    # ---------------- primary: edge-level KEEP / DROP ----------------
    tot_edges = tot_keep = tot_drop = 0
    with_note = with_train = 0
    plan = {}
    rows = []

    for d in diseases:
        r = val[d]
        kg = r["kg_symptoms"]
        keep = r["validated"]            # supported by real HPI -> KEEP
        drop = r["not_found"]            # not found -> likely hallucinated -> DROP
        n_notes = r["n_notes"]
        n_train = r["n_profiles_train_only"]
        tot_edges += len(kg); tot_keep += len(keep); tot_drop += len(drop)
        if n_notes > 0: with_note += 1
        if n_train > 0: with_train += 1

        # ---- capped grounded set (SECONDARY, short) ----
        # Prefer symptoms that are BOTH in KG AND validated (the KEEP set), ranked
        # by patient frequency from the whole-vocab grounded list; then top up with
        # canonical note buckets the KG missed. Cap to CAP.
        gfreq = {g["name"]: g["freq"] for g in r["real_grounded_symptoms"]}
        # KEEP set ordered by its own note frequency (fall back 1 if unknown)
        keep_ranked = sorted(keep, key=lambda s: (-gfreq.get(s, 1), s))
        grounded_kg = keep_ranked[:CAP]

        # note buckets not represented by any KEEP symptom -> short "KG missed" list
        keep_buckets = {canon_bucket(s) for s in keep}
        bucket_freq = Counter()
        for g in r["real_grounded_symptoms"]:
            b = canon_bucket(g["name"])
            bucket_freq[b] = max(bucket_freq[b], g["freq"])
        # Only surface CANONICALIZED buckets (drop raw long descriptive matcher
        # artifacts) that the KG missed. Rank by LIFT = this disease's mention rate
        # / background rate across OOG diseases, so disease-CHARACTERISTIC misses
        # rise above ambient noise (diarrhea/dysuria/headache appear everywhere ->
        # low lift). Require >=10% notes (or >=3 pts) AND lift >= 1.3. Short (<=5).
        thr = max(3, int(0.10 * n_notes)) if n_notes else 3
        cand_missed = []
        for b, f in bucket_freq.most_common():
            if b not in CANON_NAMES or b in keep_buckets or f < thr:
                continue
            drate = f / n_notes if n_notes else 0.0
            lift = drate / bg_rate[b] if bg_rate.get(b) else 0.0
            if lift >= 1.5:
                cand_missed.append((b, f, round(lift, 2)))
        cand_missed.sort(key=lambda x: -x[2])
        missed = cand_missed[:5]

        # ---- compaction: rebuilt degree = validated KG symptoms capped at CAP ----
        new_deg = min(len(keep), CAP)
        old_deg = len(kg)

        plan[d] = {
            "n_profiles": r["n_profiles"],
            "n_notes": n_notes,
            "n_profiles_train_only": n_train,
            "kg_symptoms": kg,
            "KEEP_validated": keep,
            "DROP_not_found": drop,
            "validation_rate": round(len(keep) / len(kg), 4) if kg else 0.0,
            "grounded_capped": grounded_kg,          # <= CAP, KEEP-preferred
            # short secondary list: (bucket, lift) disease-characteristic misses
            "note_symptoms_kg_missed": [{"symptom": b, "lift": l} for b, f, l in missed],
            "old_degree": old_deg,
            "new_degree_compact": new_deg,
        }
        rows.append({
            "disease": d,
            "n_profiles": r["n_profiles"],
            "n_notes": n_notes,
            "n_train_only": n_train,
            "old_deg": old_deg,
            "keep": len(keep),
            "drop": len(drop),
            "validation_rate": round(len(keep) / len(kg), 4) if kg else 0.0,
            "new_deg_compact": new_deg,
            "n_note_missed": len(missed),
        })

    # ---- compaction totals ----
    new_total_edges = sum(min(len(val[d]["validated"]), CAP) for d in diseases)
    # nodes: symptoms only reachable via dropped OOG edges become orphan candidates
    kept_syms = set()
    for d in diseases:
        for s in val[d]["validated"][:CAP]:
            kept_syms.add(s)
    dropped_only_candidates = set()
    for d in diseases:
        for s in val[d]["not_found"]:
            dropped_only_candidates.add(s)
    orphan_syms = dropped_only_candidates - kept_syms  # symptom names no OOG KEEP edge uses

    # ---------------- write outputs ----------------
    with open(OUT_JSON, "w") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in sorted(rows, key=lambda x: -x["n_profiles"]):
            w.writerow(row)

    # ---------------- headline print ----------------
    old_degs = [len(val[d]["kg_symptoms"]) for d in diseases]
    new_degs = [min(len(val[d]["validated"]), CAP) for d in diseases]
    n_zero_keep = sum(1 for d in diseases if val[d]["kg_symptoms"] and not val[d]["validated"])
    n_zero_note = sum(1 for d in diseases if val[d]["n_notes"] == 0)

    P = print
    P("=" * 72)
    P("VALIDATE -> PRUNE -> COMPACT  (augmented KG OOG symptom edges vs MIMIC HPI)")
    P("=" * 72)
    P(f"OOG diseases                                : {len(diseases)}")
    P(f"  with >=1 real MIMIC note (all-profile)    : {with_note}  ({100*with_note/len(diseases):.0f}%)")
    P(f"  with >=1 TRAIN-ONLY note                  : {with_train}  ({100*with_train/len(diseases):.0f}%)")
    P(f"  with NO note at all (unvalidatable)       : {n_zero_note}")
    P("")
    P("PRUNE (per-edge decision on the CURRENT augmented OOG symptom edges):")
    P(f"  current OOG symptom edges (caused_by)     : {tot_edges}")
    P(f"  KEEP  (validated in real HPI)             : {tot_keep}  ({100*tot_keep/tot_edges:.1f}%)")
    P(f"  DROP  (not found = likely hallucinated)   : {tot_drop}  ({100*tot_drop/tot_edges:.1f}%)")
    P(f"  OOG diseases left with 0 KEEP edges       : {n_zero_keep}")
    P("")
    P(f"COMPACT (rebuild OOG edges = validated KG symptoms, capped {CAP}):")
    P(f"  current total OOG symptom edges           : {tot_edges}")
    P(f"  compact  total OOG symptom edges          : {new_total_edges}")
    P(f"  edge reduction                            : {tot_edges - new_total_edges}  "
      f"(-{100*(tot_edges-new_total_edges)/tot_edges:.1f}%)")
    P(f"  per-disease degree  old  mean {statistics.mean(old_degs):.2f} median {statistics.median(old_degs)}  "
      f"max {max(old_degs)}")
    P(f"  per-disease degree  new  mean {statistics.mean(new_degs):.2f} median {statistics.median(new_degs)}  "
      f"max {max(new_degs)}")
    P(f"  symptom names used only by DROPPED edges  : {len(orphan_syms)}  (orphan-node candidates)")
    P("")
    P(f"[out] {OUT_JSON}")
    P(f"[out] {OUT_CSV}")

    # worst-validated (candidates for heaviest pruning), with notes
    P("\nHeaviest-pruned diseases (>=4 edges, >=5 notes, lowest KEEP rate):")
    cand = [r for r in rows if r["old_deg"] >= 4 and r["n_notes"] >= 5]
    cand.sort(key=lambda x: (x["validation_rate"], -x["old_deg"]))
    for r in cand[:15]:
        P(f"  keep {r['keep']:>2}/{r['old_deg']:<2} ({r['validation_rate']:.2f})  "
          f"notes={r['n_notes']:<4} {r['disease']}")


if __name__ == "__main__":
    main()
