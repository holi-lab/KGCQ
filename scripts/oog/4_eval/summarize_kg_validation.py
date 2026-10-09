#!/usr/bin/env python3
"""Aggregate/report helper over kg_symptom_validation.json + summary CSV.
Run AFTER validate_kg_symptoms_mimic.py."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import json, csv, statistics as st
from collections import Counter
ROOT = P.RV2
J = json.load(open(f"{P.KG_DIR}/kg_symptom_validation.json"))

# overall edge validation rate
tot = sum(len(v["kg_symptoms"]) for v in J.values())
val = sum(len(v["validated"]) for v in J.values())
# only diseases with >=1 note
with_notes = {k: v for k, v in J.items() if v["n_notes"] > 0}
tot_wn = sum(len(v["kg_symptoms"]) for v in with_notes.values())
val_wn = sum(len(v["validated"]) for v in with_notes.values())

val_naive = sum(len(v.get("validated_naive_substring", [])) for v in J.values())
val_naive_wn = sum(len(v.get("validated_naive_substring", [])) for v in with_notes.values())
print(f"diseases in json: {len(J)}   with >=1 note: {len(with_notes)}")
print(f"[ALL 190 diseases] KG symptom edges={tot}")
print(f"  neg-aware validated={val} ({100*val/tot:.1f}%)   naive validated={val_naive} ({100*val_naive/tot:.1f}%)")
print(f"[159 with-note diseases] edges={tot_wn}")
print(f"  neg-aware validated={val_wn} ({100*val_wn/tot_wn:.1f}%)   naive validated={val_naive_wn} ({100*val_naive_wn/tot_wn:.1f}%)")
print(f"  => negation removes {val_naive_wn-val_wn} edges ({100*(val_naive_wn-val_wn)/max(val_naive_wn,1):.1f}% of naive) as pertinent-negatives only")

# per-disease validation-rate distribution (>=1 note)
rates = [len(v["validated"]) / len(v["kg_symptoms"]) for v in with_notes.values() if v["kg_symptoms"]]
print(f"per-disease val-rate: median={st.median(rates):.2f} mean={st.mean(rates):.2f} "
      f"| =0:{sum(r==0 for r in rates)} <0.5:{sum(r<0.5 for r in rates)} =1:{sum(r==1.0 for r in rates)}")

# NEW real symptoms the KG missed (aggregate) — count how many diseases gained >=1 new
new_counts = [len(v["new_real_symptoms_not_in_kg"]) for v in with_notes.values()]
print(f"NEW real symptoms not in KG: mean/disease={st.mean(new_counts):.1f} "
      f"median={st.median(new_counts):.0f} total(sum)={sum(new_counts)}")

# most frequent NEW real symptoms across diseases (weighted by patient freq)
newsym = Counter()
for v in with_notes.values():
    kgset = set(v["kg_symptoms"])
    for g in v["real_grounded_symptoms"]:
        if g["name"] not in kgset:
            newsym[g["name"]] += g["freq"]
print("\nTop NEW real symptoms KG missed (by total patient mentions):")
for name, c in newsym.most_common(25):
    print(f"  {c:5d}  {name}")

# ---- disease-SPECIFIC new symptoms (enriched vs global background) ----
# global background prevalence of each symptom across all with-note diseases
tot_patients = sum(v["n_notes"] for v in with_notes.values())
bg = Counter()
for v in with_notes.values():
    for g in v["real_grounded_symptoms"]:
        bg[g["name"]] += g["freq"]
bg_rate = {s: c / tot_patients for s, c in bg.items()}

print("\nExample disease-SPECIFIC new symptoms KG missed (enriched >=3x background, in-disease prevalence>=0.15):")
examples = 0
for name in sorted(with_notes):
    v = with_notes[name]
    if v["n_notes"] < 10:
        continue
    kgset = set(v["kg_symptoms"])
    picks = []
    for g in v["real_grounded_symptoms"]:
        if g["name"] in kgset:
            continue
        prev = g["freq"] / v["n_notes"]
        enr = prev / max(bg_rate.get(g["name"], 1e-9), 1e-9)
        if prev >= 0.15 and enr >= 3.0:
            picks.append((g["name"], round(prev, 2), round(enr, 1)))
    if picks:
        print(f"  {name} (n={v['n_notes']}): {picks[:6]}")
        examples += 1
    if examples >= 12:
        break

# train-only coverage
n_train = sum(1 for v in J.values() if v["n_profiles_train_only"] > 0)
print(f"\nTrain-only coverage: {n_train} diseases have >=1 non-eval train note")
