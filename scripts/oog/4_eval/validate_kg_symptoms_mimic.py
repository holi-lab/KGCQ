#!/usr/bin/env python3
"""
Validate the AUGMENTED diagnostic KG's OOG-disease symptom edges against REAL
MIMIC-IV discharge notes (History of Present Illness sections).

For each of the 190 OOG diseases (augmented Disease names NOT in the paper KG),
whose Symptom/Cause/Risk_Factor attributes were INVENTED by an LLM (Claude), we
check which augmented-KG symptom edges are actually SUPPORTED by that disease's
real patients' HPI notes, vs NOT found (likely hallucinated). We also extract a
real-grounded symptom set per disease from the full KG symptom vocabulary.

Data sources (all absolute paths, MIMIC read-only):
  1. Profiles (5517):  /data2/.../rebuttal/data/kg_mapping/profile_classification/profiles_paperKG_OOD_50289.jsonl
  2. Catalog map:      data/oog/profile_split_inputs/catalog_with_relationship.csv  (raw disease -> kg_disease, relationship)
  3. MIMIC HPI:        /data2/.../data/mimic/note_section.csv  (has 'History of Present Illness' col, keyed by hadm_id)
  4. KG:               data/KG/{augmented,paper}/{nodes,edges}.csv
  5. Leakage splits:   data/ood_split/{hg,hv}_train_ood_ordered.json ; data/profile/{ood_valid_128,ood_test_clean243,ood_test_275}.json

Outputs:
  data/KG/kg_symptom_validation.json
  4_eval/kg_symptom_validation_summary.csv
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..'))
import _paths as P
_sys.path.insert(0, str(P.ROOT))
import os, re, json, csv
from collections import defaultdict, Counter
import pandas as pd

ROOT = P.RV2
PROF = P.OOG_PROFILE_POOL_JSONL
NS   = P.MIMIC_NOTE_SECTION_CSV
AUG_NODES = P.KG["augmented"]["nodes"]
AUG_EDGES = P.KG["augmented"]["edges"]
PAPER_NODES = P.KG["paper"]["nodes"]
CATALOG = str(P.SPLIT_INPUT_DIR / "catalog_with_relationship.csv")
OUT_JSON = str(P.KG_V3_DIR / "kg_symptom_validation.json")
OUT_CSV  = str(P.RESULTS_DIR / "kg_symptom_validation_summary.csv")

# ---------------------------------------------------------------------------
# Synonym map: KG symptom name (lower) -> list of extra regex alternatives to
# look for in real HPI text (in addition to the literal KG phrase itself).
# Keys are matched by EXACT equality against the KG symptom name (NOT substring)
# so that broad heads like "edema"/"fever"/"dyspnea" do not leak into compound
# clinical terms (e.g. "laryngeal edema"). Values are raw regex fragments.
# ---------------------------------------------------------------------------
SYNONYMS = {
    "dyspnea":            [r"shortness of breath", r"\bsob\b", r"short of breath", r"breathless", r"dyspne", r"difficulty breathing", r"trouble breathing"],
    "dyspnea on exertion":[r"dyspnea on exertion", r"\bdoe\b", r"exertional dyspnea", r"short of breath.*exert"],
    "paroxysmal nocturnal dyspnea": [r"\bpnd\b", r"paroxysmal nocturnal dyspnea"],
    "hematuria":          [r"blood in (the )?urine", r"bloody urine", r"hematuria"],
    "emesis":             [r"vomit", r"emesis"],
    "pyrexia":            [r"fever", r"febrile", r"pyrexia"],
    "fever":              [r"fever", r"febrile", r"pyrexia", r"\bt(emp)?\s?1[0-9][0-9]\b"],
    "chills":             [r"chills", r"rigors"],
    "abdominal pain":     [r"abdominal pain", r"abd(ominal)? (pain|tenderness)", r"\babd\b.{0,6}pain", r"belly pain", r"stomach pain", r"epigastric pain"],
    "abdominal distension":[r"abdominal disten", r"abd(ominal)? disten", r"abdominal distention", r"distended abdomen", r"abdominal swelling", r"abdominal bloating"],
    "abdominal tenderness":[r"abdominal tender", r"abd(ominal)? tender", r"tender.{0,8}abdomen"],
    "nausea":             [r"nausea", r"nauseous"],
    "chest pain":         [r"chest pain", r"chest discomfort", r"chest pressure", r"chest tightness", r"\bcp\b"],
    "pleuritic chest pain":[r"pleuritic", r"pleuritic chest pain"],
    "ischemic chest pain (squeezing pain)": [r"chest pain", r"squeezing", r"chest pressure", r"substernal"],
    "palpitations":       [r"palpitation", r"heart racing", r"racing heart"],
    "tachycardia":        [r"tachycard", r"\bhr\b.{0,4}1[0-9][0-9]", r"heart rate.{0,6}1[0-9][0-9]"],
    "bradycardia":        [r"bradycard", r"slow heart rate"],
    "headaches":          [r"headache", r"\bha\b", r"cephalgia"],
    "severe headache":    [r"severe headache", r"worst headache", r"thunderclap"],
    "fatigue":            [r"fatigue", r"tired", r"lethargy", r"lethargic", r"malaise", r"weak(ness)?", r"low energy", r"exhaust"],
    "chronic fatigue":    [r"fatigue", r"chronic fatigue", r"tired", r"malaise"],
    "weight loss":        [r"weight loss", r"lost .{0,6}(lbs|pounds|kg|weight)", r"losing weight", r"unintentional weight"],
    "jaundice":           [r"jaundice", r"jaundiced", r"icteric", r"icterus", r"yellow(ing)? (of )?(skin|eyes|sclera)", r"scleral icterus"],
    "hepatomegaly":       [r"hepatomegaly", r"enlarged liver", r"liver.{0,10}enlarg"],
    "splenomegaly":       [r"splenomegaly", r"enlarged spleen"],
    "hepatosplenomegaly": [r"hepatosplenomegaly"],
    "leg swelling":       [r"leg swelling", r"leg(s)? .{0,6}swollen", r"swollen leg", r"swelling .{0,12}\bleg\b", r"lower extremity (edema|swelling)", r"\blle\b", r"\brle\b", r"\bble\b", r"\bble\b.{0,6}edema", r"pedal edema"],
    "leg pain":           [r"leg pain", r"\blle\b.{0,8}pain", r"pain.{0,6}(leg|calf|thigh)", r"calf pain"],
    "leg cramps":         [r"leg cramp", r"cramp"],
    "edema":              [r"edema", r"swelling", r"swollen"],
    "peripheral coldness":[r"cold.{0,6}(extremit|feet|foot|leg|hand|toe)", r"peripheral cold", r"cool extremit"],
    "flank pain":         [r"flank pain", r"flank tender", r"\bcva\b tender", r"costovertebral"],
    "back pain":          [r"back pain", r"backache", r"\blbp\b"],
    "seizure":            [r"seizure", r"convuls", r"\bsz\b", r"tonic.?clonic", r"epilep"],
    "syncope":            [r"syncope", r"syncopal", r"fainting", r"pass(ed|ing) out", r"loss of consciousness", r"\bloc\b", r"blackout"],
    "dizziness":          [r"dizz", r"lightheaded", r"light-headed", r"vertigo"],
    "confusion":          [r"confus", r"disorient", r"altered mental", r"\bams\b"],
    "mental status changes":[r"altered mental", r"mental status (change|decline)", r"\bams\b", r"confus", r"disorient", r"encephalopath"],
    "impaired consciousness":[r"unresponsive", r"altered mental", r"decreased consciousness", r"obtunded", r"lethargic", r"loss of consciousness", r"\bloc\b"],
    "disorder of consciousness":[r"unresponsive", r"altered mental", r"obtunded", r"loss of consciousness"],
    "cognitive decline":  [r"cognitive decline", r"memory loss", r"memory impair", r"forgetful", r"confus", r"dementia"],
    "cognitive impairment":[r"cognitive impair", r"memory loss", r"confus", r"dementia"],
    "hemoptysis":         [r"hemoptysis", r"coughing (up )?blood", r"blood.{0,6}sputum"],
    "hematochezia":       [r"hematochezia", r"blood.{0,6}(stool|rectum|per rectum)", r"\bbrbpr\b", r"bright red blood"],
    "bloody stool":       [r"bloody stool", r"blood.{0,6}stool", r"melena", r"\bbrbpr\b", r"hematochezia", r"tarry stool"],
    "bloody diarrhea":    [r"bloody diarrhea", r"blood.{0,6}diarrhea"],
    "melena":             [r"melena", r"black.{0,6}(stool|tarry)", r"tarry stool"],
    "hematemesis":        [r"hematemesis", r"vomit.{0,6}blood", r"coffee.?ground"],
    "diarrhea":           [r"diarrhea", r"loose stool", r"watery stool"],
    "constipation":       [r"constipat", r"no bowel movement", r"unable to.{0,10}(stool|bm)"],
    "dysuria":            [r"dysuria", r"burning.{0,6}urin", r"pain.{0,6}urinat", r"painful urination"],
    "frequent urination": [r"frequen.{0,6}urin", r"urinary frequency", r"polyuria"],
    "urinary retention":  [r"urinary retention", r"unable to.{0,6}(urinat|void)", r"retention of urine"],
    "oliguria":           [r"oliguria", r"decreased urine", r"low urine output", r"decreased urinary"],
    "anuria":             [r"anuria", r"no urine", r"not.{0,6}urinat", r"absent urine"],
    "decreased urinary output": [r"decreased urin", r"low urine output", r"oliguria", r"anuria", r"making less urine"],
    "cough":              [r"cough"],
    "dry cough":          [r"dry cough", r"nonproductive cough", r"non-productive cough"],
    "frequent cough":     [r"cough"],
    "productive cough":   [r"productive cough", r"cough.{0,6}sputum", r"cough.{0,6}phlegm"],
    "sputum":             [r"sputum", r"phlegm"],
    "increased sputum volume":[r"sputum", r"phlegm", r"increased secretion"],
    "sore throat":        [r"sore throat", r"throat pain", r"pharyngitis", r"odynophagia"],
    "severe sore throat": [r"sore throat", r"severe.{0,6}throat", r"odynophagia"],
    "dysphagia":          [r"dysphagia", r"difficulty swallow", r"trouble swallow", r"unable to swallow"],
    "odynophagia":        [r"odynophagia", r"painful swallow", r"pain.{0,6}swallow"],
    "hoarseness":         [r"hoarse", r"voice change"],
    "dysphonia":          [r"dysphonia", r"hoarse", r"voice change"],
    "wheezing":           [r"wheez"],
    "stridor":            [r"stridor"],
    "hypoxemia":          [r"hypox", r"desatur", r"\bo2 sat", r"oxygen saturation", r"sat.{0,6}[0-8][0-9]%"],
    "tachypnea":          [r"tachypn", r"\brr\b.{0,4}[2-4][0-9]", r"rapid breathing"],
    "cyanosis":           [r"cyanos", r"blue.{0,6}(lip|skin|fing, toe)"],
    "hypotension":        [r"hypotens", r"low blood pressure", r"\bsbp\b.{0,4}[5-9][0-9]\b", r"\bbp\b.{0,6}[5-9][0-9]/"],
    "resistant hypertension":[r"resistant hypertension", r"uncontrolled.{0,6}(bp|blood pressure|hypertension)", r"refractory hypertension"],
    "jugular venous distension":[r"\bjvd\b", r"jugular venous", r"elevated jvp", r"neck vein"],
    "heart murmur":       [r"murmur"],
    "holosystolic heart murmur":[r"holosystolic", r"pansystolic", r"murmur"],
    "skin rash":          [r"rash", r"eruption", r"skin lesion"],
    "rash":               [r"rash", r"eruption"],
    "pruritus":           [r"pruritus", r"itch", r"itchy"],
    "urticaria":          [r"urticaria", r"hives", r"wheal"],
    "wheals":             [r"wheal", r"hives", r"urticaria"],
    "itchy wheals":       [r"wheal", r"hives", r"itch"],
    "petechial rash":     [r"petechia", r"petechial"],
    "palpable purpura":   [r"purpura", r"palpable purpura"],
    "skin ulceration":    [r"ulcer", r"skin ulcer"],
    "skin ulcer":         [r"ulcer"],
    "skin necrosis":      [r"necros", r"necrotic", r"gangrene", r"eschar"],
    "tissue necrosis":    [r"necros", r"necrotic", r"gangren"],
    "skin blisters":      [r"blister", r"bulla", r"bullae", r"vesicle", r"vesicul"],
    "vesicular lesions":  [r"vesic", r"vesicular"],
    "erythema":           [r"erythem", r"redness"],
    "cutaneous erythema": [r"erythem", r"redness", r"red skin", r"skin redness"],
    "facial flushing":    [r"flush", r"facial flush"],
    "facial swelling":    [r"facial swelling", r"face.{0,6}swollen", r"swelling.{0,6}face"],
    "facial droop":       [r"facial droop", r"face.{0,6}droop", r"facial asymmetry"],
    "facial paralysis":   [r"facial paralysis", r"facial palsy", r"facial droop", r"facial weakness"],
    "facial numbness":    [r"facial numb", r"numb.{0,6}face"],
    "slurred speech":     [r"slurred speech", r"dysarthr", r"slurr"],
    "speech difficulty":  [r"speech difficult", r"aphasia", r"dysarthr", r"slurr", r"difficulty speaking"],
    "language impairment":[r"aphasia", r"language", r"difficulty speaking", r"word.finding"],
    "dysphasia":          [r"dysphasia", r"aphasia"],
    "limb weakness":      [r"weakness", r"weak.{0,6}(arm|leg|extremit|limb)", r"hemiparesis", r"paresis"],
    "proximal muscle weakness":[r"proximal.{0,6}weakness", r"muscle weakness", r"weak"],
    "difficulty walking long distances":[r"difficulty walking", r"unable to walk", r"trouble walking", r"gait", r"ambulat"],
    "gait dysfunction":   [r"gait", r"unsteady", r"difficulty walking", r"ambulat", r"balance"],
    "ataxia":             [r"ataxia", r"ataxic", r"unsteady", r"imbalance"],
    "tremor":             [r"tremor", r"shaking", r"shaky"],
    "intermittent claudication":[r"claudicat", r"leg pain.{0,10}walk", r"cramp.{0,10}walk"],
    "arthralgia":         [r"arthralgia", r"joint pain"],
    "joint pain":         [r"joint pain", r"arthralgia", r"joint ache"],
    "myalgia":            [r"myalgia", r"muscle ache", r"muscle pain"],
    "generalized myalgia":[r"myalgia", r"muscle ache", r"body ache"],
    "bone pain":          [r"bone pain", r"bony pain"],
    "hearing loss":       [r"hearing loss", r"decreased hearing", r"deaf"],
    "tinnitus":           [r"tinnitus", r"ringing.{0,6}ear"],
    "vision impairment":  [r"vision.{0,6}(loss|impair|change|blur)", r"blurr.{0,6}vision", r"visual.{0,6}(loss|change)", r"decreased vision", r"blind"],
    "visual disturbance": [r"visual.{0,6}(disturb|change)", r"vision.{0,6}(change|blur|loss)", r"blurr.{0,6}vision", r"diplopia", r"double vision"],
    "visual field defect":[r"visual field", r"hemianopia", r"field.{0,6}defect", r"field cut"],
    "eye pain":           [r"eye pain", r"ocular pain", r"painful eye"],
    "eye redness":        [r"eye.{0,6}(red|redness)", r"red eye", r"conjunctival inject"],
    "conjunctival injection":[r"conjunctival inject", r"red eye", r"injected conjunct"],
    "epistaxis":          [r"epistaxis", r"nosebleed", r"nose bleed", r"bleeding.{0,6}nose"],
    "easy bruising":      [r"easy bruis", r"bruis", r"ecchymos"],
    "prolonged bleeding": [r"prolonged bleed", r"bleeding.{0,10}(stop|control)", r"excessive bleed"],
    "gingival bleeding":  [r"gingival bleed", r"gum bleed", r"bleeding gum"],
    "hypercalcemia":      [r"hypercalcemia", r"elevated calcium", r"high calcium"],
    "hyperkalemia":       [r"hyperkalemia", r"elevated potassium", r"high potassium", r"\bk\b.{0,4}[6-9]\.\d"],
    "hypokalemia":        [r"hypokalemia", r"low potassium"],
    "hyponatremia":       [r"hyponatremia", r"low sodium", r"\bna\b.{0,4}1[0-2][0-9]\b"],
    "leukocytosis":       [r"leukocytos", r"elevated (wbc|white)", r"high white", r"\bwbc\b.{0,6}1[5-9]"],
    "eosinophilia":       [r"eosinophil"],
    "elevated liver enzymes":[r"elevated.{0,10}(lft|transaminase|liver enzyme)", r"transaminitis", r"elevated (ast|alt)", r"\balt\b.{0,6}\d{3}", r"\bast\b.{0,6}\d{3}"],
    "elevated serum lactate":[r"lactate", r"elevated lactate", r"lactic acid"],
    "coagulopathy":       [r"coagulopath", r"\binr\b.{0,6}[2-9]", r"elevated inr"],
    "proteinuria":        [r"proteinuria", r"protein in urine", r"foamy urine"],
    "foamy urine":        [r"foamy urine", r"proteinuria"],
    "pancytopenia":       [r"pancytopenia"],
    "thrombocytopenia":   [r"thrombocytopenia", r"low platelet"],
    "anemia":             [r"anemia", r"anemic", r"low.{0,6}(hgb|hemoglobin|hct)"],
    "conjunctival pallor":[r"pallor", r"pale", r"conjunctival pallor"],
    "night sweats":       [r"night sweat", r"diaphoresis at night"],
    "cold sweat":         [r"cold sweat", r"diaphore", r"clammy", r"sweating"],
    "jaw pain":           [r"jaw pain", r"jaw claudicat"],
    "jaw claudication":   [r"jaw claudicat", r"jaw pain.{0,10}chew"],
    "right upper quadrant pain":[r"right upper quadrant", r"\bruq\b", r"rt upper quadrant"],
    "left upper quadrant pain":[r"left upper quadrant", r"\bluq\b"],
    "right lower quadrant pain":[r"right lower quadrant", r"\brlq\b"],
    "upper abdominal pain":[r"upper abdominal", r"epigastr"],
    "postprandial epigastric pain":[r"epigastr", r"postprandial", r"after eating", r"after meal"],
    "regurgitation/heartburn worsens when lying down":[r"heartburn", r"reflux", r"regurgitat", r"\bgerd\b", r"acid reflux"],
    "the sensation of acid reflux":[r"acid reflux", r"reflux", r"heartburn", r"\bgerd\b"],
    "steatorrhea":        [r"steatorrhea", r"greasy stool", r"fatty stool", r"oily stool"],
    "clay-colored stools":[r"clay.color", r"pale stool", r"acholic"],
    "scrotal swelling":   [r"scrotal swelling", r"scrotum.{0,8}swollen", r"swollen scrotum"],
    "testicular pain":    [r"testicular pain", r"testicle pain", r"scrotal pain"],
    "groin pain":         [r"groin pain"],
    "groin bulge":        [r"groin.{0,6}bulge", r"inguinal.{0,6}bulge", r"groin mass"],
    "perianal pain and swelling":[r"perianal", r"anal pain", r"rectal pain", r"anal swelling"],
    "anal pain":          [r"anal pain", r"rectal pain", r"perianal pain"],
    "rectal urgency":     [r"rectal urgency", r"tenesmus", r"urgency"],
    "delusions":          [r"delusion", r"paranoi"],
    "hallucinations":     [r"hallucinat", r"seeing things", r"hearing voices"],
    "agitation":          [r"agitat", r"combative", r"restless"],
    "depressed mood":     [r"depress", r"low mood", r"sad"],
    "anhedonia":          [r"anhedonia", r"loss of interest", r"no interest"],
    "anxiety":            [r"anxiety", r"anxious", r"nervous"],
    "sleep disturbance":  [r"sleep disturb", r"insomnia", r"difficulty sleep", r"trouble sleep", r"can't sleep"],
    "trouble falling asleep":[r"insomnia", r"trouble.{0,6}sleep", r"difficulty falling asleep", r"can't sleep"],
    "withdrawal symptoms":[r"withdrawal", r"\bdt\b", r"delirium tremens"],
    "alcohol craving":    [r"alcohol craving", r"crave.{0,6}alcohol", r"drinking"],
    "asterixis":          [r"asterixis", r"flapping"],
    "fetor hepaticus":    [r"fetor hepaticus"],
    "encephalopathy":     [r"encephalopath", r"altered mental", r"confus"],
    "positive blood culture":[r"blood culture.{0,10}(positive|grew|growing)", r"bacteremia", r"positive blood cx"],
    "positive costovertebral angle tenderness":[r"\bcva\b tender", r"costovertebral", r"flank tender"],
    "recurrent infections":[r"recurrent infection", r"frequent infection"],
    "swollen lymph nodes":[r"lymphadenopath", r"swollen (lymph|node)", r"enlarged (lymph|node)"],
    "localized lymphadenopathy":[r"lymphadenopath", r"swollen.{0,6}node", r"enlarged.{0,6}node"],
    "regional lymphadenopathy":[r"lymphadenopath", r"swollen.{0,6}node"],
    "pulsatile abdominal mass":[r"pulsatile.{0,6}(mass|abdom)", r"abdominal mass"],
    "pulsatile mass":     [r"pulsatile mass"],
    "abdominal or pelvic mass":[r"abdominal mass", r"pelvic mass", r"palpable mass"],
    "impaired executive functions":[r"executive function", r"cognitive"],
    "toothache":          [r"tooth.?ache", r"tooth pain", r"dental pain"],
    "tooth pain":         [r"tooth pain", r"tooth.?ache", r"dental pain"],
    "dental caries":      [r"dental car", r"cavit", r"tooth decay"],
    "erythema migrans rash":[r"erythema migrans", r"target lesion", r"bull.?s.?eye"],
    "target skin lesions":[r"target lesion", r"targetoid"],
    # --- additional exact-key entries for common literal-only OOG symptoms ---
    "orthopnea":          [r"orthopnea", r"orthopnoea", r"sleep.{0,6}(upright|propped)", r"pillow"],
    "lymphadenopathy":    [r"lymphadenopath", r"swollen.{0,6}node", r"enlarged.{0,6}node", r"swollen gland"],
    "muscle atrophy":     [r"muscle atrophy", r"muscle wasting", r"atrophy", r"wasting"],
    "muscle pain":        [r"muscle pain", r"myalgia", r"muscle ache"],
    "limb pain":          [r"limb pain", r"extremity pain", r"arm pain", r"leg pain"],
    "limb swelling":      [r"limb swelling", r"extremity (swelling|edema)", r"swollen (arm|leg|limb|extremit)"],
    "arm swelling":       [r"arm swelling", r"arm.{0,6}swollen", r"swollen arm", r"\bue\b.{0,6}(edema|swelling)"],
    "neck pain":          [r"neck pain", r"neck.{0,6}pain", r"cervical pain"],
    "neck swelling":      [r"neck swelling", r"neck.{0,6}(swollen|mass)", r"swelling.{0,6}neck"],
    "hip pain":           [r"hip pain", r"hip.{0,6}pain"],
    "knee pain":          [r"knee pain", r"knee.{0,6}pain"],
    "shoulder pain":      [r"shoulder pain"],
    "lower back pain":    [r"lower back pain", r"low back pain", r"\blbp\b", r"lumbar pain"],
    "lower abdominal pain":[r"lower abdominal pain", r"lower abdomen.{0,6}pain", r"suprapubic pain", r"pelvic pain"],
    "loss of consciousness":[r"loss of consciousness", r"\bloc\b", r"unconscious", r"passed out", r"unresponsive", r"syncop"],
    "mental confusion":   [r"confus", r"disorient", r"altered mental", r"\bams\b"],
    "diminished breath sounds":[r"diminished breath sound", r"decreased breath sound", r"reduced breath sound"],
    "hepatic steatosis":  [r"hepatic steatosis", r"fatty liver", r"steatosis"],
    "kidney dysfunction": [r"kidney dysfunction", r"renal.{0,10}(failure|insufficiency|dysfunction)", r"\baki\b", r"acute kidney", r"elevated creatinine", r"rising creatinine", r"\bckd\b"],
    "fluid overload":     [r"fluid overload", r"volume overload", r"hypervolem", r"fluid retention"],
    "non-healing wound":  [r"non.?healing wound", r"non.?healing ulcer", r"chronic wound", r"wound.{0,10}not heal"],
    "oral ulcers":        [r"oral ulcer", r"mouth ulcer", r"aphthous"],
    "morning stiffness (>1 hour)":[r"morning stiffness", r"stiffness.{0,10}morning"],
    "heavy menstruation": [r"heavy menstru", r"menorrhagia", r"heavy period", r"heavy vaginal bleeding"],
    "fecal incontinence": [r"fecal incontinence", r"incontinen.{0,6}(stool|bowel)", r"bowel incontinence"],
    "urinary incontinence":[r"urinary incontinence", r"incontinen.{0,6}urine", r"leak.{0,6}urine"],
    "ear pain":           [r"ear pain", r"otalgia", r"earache"],
    "ear discharge":      [r"ear discharge", r"otorrhea", r"drainage.{0,6}ear"],
    "focal neurological abnormality":[r"focal neuro", r"focal deficit", r"neurologic.{0,6}deficit", r"hemiparesis", r"facial droop", r"focal weakness"],
    "cranial nerve palsies":[r"cranial nerve", r"\bcn\b.{0,4}(palsy|deficit)"],
    "pathologic fracture":[r"pathologic.{0,6}fracture", r"pathological fracture"],
    "fragility fracture": [r"fragility fracture", r"low.?trauma fracture", r"compression fracture"],
    "chronic diarrhea":   [r"chronic diarrhea", r"diarrhea", r"loose stool"],
}


def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower())


# A small set of "head-word" synonym expansions that are safe to apply even when
# the KG symptom name is a COMPOUND term (e.g. "leg pain"->has "pain" but we do NOT
# want broad synonyms from generic heads). We ONLY expand a compound name via a
# head-word key when the KG name *ends with* that key AND the modifier is preserved
# in the alternative. To stay strictly precise we DISABLE compound inheritance and
# apply synonyms by EXACT key match only. This prevents broad synonyms (swelling,
# edema, fever, dyspnea, pain) from leaking into long descriptive KG symptom names.

def build_matcher_terms(kg_sym_lower):
    """Return list of compiled regexes for a KG symptom name.

    Matching = literal phrase (word-boundary, flexible whitespace)
             + synonym alternatives ONLY when the KG name EXACTLY equals a
               synonym-map key. Substring/compound inheritance is intentionally
               disabled so that generic heads like 'edema'/'fever'/'dyspnea' do
               not validate unrelated compound symptoms.
    """
    pats = []
    literal = re.escape(kg_sym_lower).replace(r"\ ", r"\s+")
    # word-boundary anchor the literal where the phrase starts/ends with a word char
    lit = literal
    if kg_sym_lower[:1].isalnum():
        lit = r"\b" + lit
    if kg_sym_lower[-1:].isalnum():
        lit = lit + r"\b"
    pats.append(lit)
    if kg_sym_lower in SYNONYMS:            # EXACT key match only
        pats.extend(SYNONYMS[kg_sym_lower])
    seen, out = set(), []
    for p in pats:
        if p in seen:
            continue
        seen.add(p)
        try:
            out.append(re.compile(p, re.IGNORECASE))
        except re.error:
            out.append(re.compile(re.escape(p), re.IGNORECASE))
    return out


# --- NegEx-style negation handling ---------------------------------------
# HPI notes are dense with pertinent-negatives ("denies chest pain", "no fever",
# "without dyspnea"). A naive substring/regex match counts these as POSITIVE,
# which was measured to inflate common-symptom matches by ~40%. We therefore
# affirm a match only if NO negation trigger appears in a short preceding window
# (bounded by clause punctuation), and NO "pseudo-negation" that is actually
# affirming ("not ruled out", "cannot exclude") reverses it.
NEG_TRIGGERS = re.compile(
    r"\b(no|not|non|without|w/o|denies|denied|deny|negative for|neg\.? for|"
    r"absent|absence of|free of|ruled out|r/o|rule out|no evidence of|"
    r"no sign(s)? of|no complaints? of|not have|didn'?t have|does(n'?t| not) have|"
    r"unremarkable for|no h/o|no history of)\b", re.IGNORECASE)
# clause boundaries that stop a negation's scope
CLAUSE_STOP = re.compile(r"[.;:!?]|\bbut\b|\bhowever\b|\bpositive for\b|\bendorses?\b|\bwith\b(?!out)")


def _is_negated(text, start):
    """True if the match at `text[start:]` is within a preceding negation scope
    (window of up to ~45 chars back, not crossing a clause boundary)."""
    win_start = max(0, start - 48)
    window = text[win_start:start]
    # cut window at the last clause boundary so negation from a prior clause
    # does not spill over ("has fever. no chest pain" -> fever not negated)
    stops = list(CLAUSE_STOP.finditer(window))
    if stops:
        window = window[stops[-1].end():]
    return bool(NEG_TRIGGERS.search(window))


def phrase_in_text(regexes, text, negation_aware=True):
    """Return True if any regex has at least one AFFIRMED (non-negated) match."""
    for rgx in regexes:
        for m in rgx.finditer(text):
            if not negation_aware or not _is_negated(text, m.start()):
                return True
    return False


def main():
    # -------- KG --------
    an = pd.read_csv(AUG_NODES)
    pn = pd.read_csv(PAPER_NODES)
    an["name_l"] = an["name"].astype(str).map(norm)
    id2name = dict(zip(an["id"], an["name_l"]))
    id2label = dict(zip(an["id"], an["label"]))
    aug_dis = set(an[an["label"] == "Disease"]["name_l"])
    paper_dis = set(pn[pn["label"] == "Disease"]["name"].astype(str).map(norm))
    oog = aug_dis - paper_dis
    print(f"[KG] augmented diseases={len(aug_dis)}  paper diseases={len(paper_dis)}  OOG={len(oog)}")

    ae = pd.read_csv(AUG_EDGES)
    # OOG disease -> set of KG symptom names (caused_by edges, start=Symptom)
    disease_kg_sym = defaultdict(set)
    for _, r in ae.iterrows():
        if r["type"] != "caused_by":
            continue
        s, e = r["start"], r["end"]
        if id2label.get(s) == "Symptom":
            d = id2name.get(e)
            if d in oog:
                disease_kg_sym[d].add(id2name.get(s))
    print(f"[KG] OOG diseases with >=1 symptom edge = {len(disease_kg_sym)}")

    # full augmented Symptom vocabulary (for grounded extraction)
    all_symptoms = sorted(set(an[an["label"] == "Symptom"]["name_l"]))
    print(f"[KG] total augmented Symptom vocabulary = {len(all_symptoms)}")

    # precompile matchers for every symptom in vocabulary (reused)
    sym_matcher = {s: build_matcher_terms(s) for s in all_symptoms}

    # -------- catalog: raw disease -> OOG kg_disease (same/child) --------
    cat = pd.read_csv(CATALOG)
    cat["disease_l"] = cat["disease"].astype(str).map(norm)
    cat["kg_disease_l"] = cat["kg_disease"].astype(str).map(norm)
    cc = cat[cat["disease_relationship"].isin(["same", "child"])]
    raw2kg = defaultdict(set)
    for _, r in cc.iterrows():
        if r["kg_disease_l"] in oog:
            raw2kg[r["disease_l"]].add(r["kg_disease_l"])
    oog_reachable = set().union(*raw2kg.values()) if raw2kg else set()
    print(f"[catalog] raw diseases mapping to OOG (same/child) = {len(raw2kg)}; "
          f"distinct OOG reachable = {len(oog_reachable)} / {len(oog)}")

    # -------- profiles: OOG disease -> hadm_ids, + chiefcomplaint --------
    disease2hadm = defaultdict(set)
    hadm2cc = {}
    n_prof = 0
    with open(PROF) as f:
        for line in f:
            o = json.loads(line)
            n_prof += 1
            h = str(o["hadm_id"])
            hadm2cc[h] = norm(o.get("chiefcomplaint", ""))
            for rw in [norm(d) for d in o.get("disease", [])]:
                for kg in raw2kg.get(rw, ()):
                    disease2hadm[kg].add(h)
    print(f"[profiles] total={n_prof}  OOG diseases with >=1 hadm = {len(disease2hadm)}")

    all_needed_hadm = set().union(*disease2hadm.values()) if disease2hadm else set()
    print(f"[profiles] unique hadm needed = {len(all_needed_hadm)}")

    # -------- load HPI for needed hadm from note_section.csv --------
    hadm2hpi = {}
    got_rows = 0
    for chunk in pd.read_csv(NS, usecols=["hadm_id", "History of Present Illness"],
                             dtype={"hadm_id": str}, chunksize=200000):
        m = chunk[chunk["hadm_id"].isin(all_needed_hadm)]
        for _, r in m.iterrows():
            txt = r["History of Present Illness"]
            if isinstance(txt, str) and txt.strip():
                hadm2hpi[r["hadm_id"]] = norm(txt)
                got_rows += 1
    print(f"[notes] hadm with non-empty HPI = {got_rows} / {len(all_needed_hadm)}")

    # -------- leakage: train-only hadm set & eval-exclude set --------
    train_ood = set()
    for fn in ["data/ood_split/hg_train_ood_ordered.json", "data/ood_split/hv_train_ood_ordered.json"]:
        d = json.load(open(os.path.join(ROOT, fn)))
        train_ood.update(str(x) for x in d.get("order", []))
    eval_exclude = set()
    for fn in ["data/profile/ood_valid_128.json", "data/profile/ood_test_clean243.json", "data/profile/ood_test_275.json"]:
        d = json.load(open(os.path.join(ROOT, fn)))
        eval_exclude.update(str(k) for k in d.keys())
    print(f"[leakage] train_ood(order union)={len(train_ood)}  eval_exclude(keys union)={len(eval_exclude)}")
    print(f"[leakage] train_ood ∩ eval_exclude = {len(train_ood & eval_exclude)}")

    # -------- per-disease validation & grounded extraction --------
    result = {}
    summary_rows = []
    total_edges = 0
    total_validated = 0          # negation-aware (primary)
    total_validated_naive = 0    # naive substring (comparison)

    for d in sorted(disease_kg_sym.keys()):
        kg_syms = sorted(disease_kg_sym[d])
        hadms_all = sorted(disease2hadm.get(d, set()))
        hadms_with_note = [h for h in hadms_all if h in hadm2hpi]
        # concatenate HPI (all-profile)
        notes = [hadm2hpi[h] for h in hadms_with_note]
        # weak signal: chiefcomplaint too
        cc_texts = [hadm2cc.get(h, "") for h in hadms_with_note]
        combined_texts = [n + " || " + c for n, c in zip(notes, cc_texts)]

        # train-only subset
        hadms_train = [h for h in hadms_with_note if (h in train_ood) and (h not in eval_exclude)]
        train_texts = [hadm2hpi[h] + " || " + hadm2cc.get(h, "") for h in hadms_train]

        # ---- validate KG symptom edges (negation-aware primary + naive) ----
        validated, not_found, validated_naive = [], [], []
        sym_support = {}   # kg symptom -> #patients with an AFFIRMED mention
        for s in kg_syms:
            rgx = sym_matcher.get(s) or build_matcher_terms(s)
            n_aff = sum(1 for t in combined_texts if phrase_in_text(rgx, t, negation_aware=True))
            hit_naive = any(phrase_in_text(rgx, t, negation_aware=False) for t in combined_texts)
            sym_support[s] = n_aff
            (validated if n_aff > 0 else not_found).append(s)
            if hit_naive:
                validated_naive.append(s)
        total_edges += len(kg_syms)
        total_validated += len(validated)
        total_validated_naive += len(validated_naive)

        # ---- real-grounded extraction (whole vocab, freq = #patients affirmed) ----
        grounded = []
        grounded_train = []
        for s in all_symptoms:
            rgx = sym_matcher[s]
            freq = sum(1 for t in combined_texts if phrase_in_text(rgx, t, negation_aware=True))
            if freq > 0:
                grounded.append({"name": s, "freq": freq})
            freq_tr = sum(1 for t in train_texts if phrase_in_text(rgx, t, negation_aware=True))
            if freq_tr > 0:
                grounded_train.append({"name": s, "freq": freq_tr})
        grounded.sort(key=lambda x: (-x["freq"], x["name"]))
        grounded_train.sort(key=lambda x: (-x["freq"], x["name"]))

        # which grounded symptoms are NEW (not in KG for this disease)
        kg_set = set(kg_syms)
        new_grounded = [g for g in grounded if g["name"] not in kg_set]

        result[d] = {
            "kg_symptoms": kg_syms,
            "validated": validated,                 # negation-aware
            "not_found": not_found,
            "validated_naive_substring": validated_naive,
            "kg_symptom_patient_support": sym_support,
            "real_grounded_symptoms": grounded,
            "real_grounded_symptoms_train_only": grounded_train,
            "new_real_symptoms_not_in_kg": [g["name"] for g in new_grounded],
            "n_profiles": len(hadms_all),
            "n_notes": len(hadms_with_note),
            "n_profiles_train_only": len(hadms_train),
        }
        summary_rows.append({
            "disease": d,
            "n_profiles": len(hadms_all),
            "n_notes": len(hadms_with_note),
            "n_profiles_train_only": len(hadms_train),
            "n_kg_sym": len(kg_syms),
            "n_validated": len(validated),
            "n_validated_naive": len(validated_naive),
            "validation_rate": round(len(validated) / len(kg_syms), 4) if kg_syms else 0.0,
            "validation_rate_naive": round(len(validated_naive) / len(kg_syms), 4) if kg_syms else 0.0,
            "n_real_grounded": len(grounded),
            "n_real_grounded_train_only": len(grounded_train),
            "n_new_real_not_in_kg": len(new_grounded),
        })

    # OOG diseases with NO real profile at all (edges exist but no MIMIC note)
    no_profile = sorted(set(disease_kg_sym.keys()) - set(disease2hadm.keys()))

    # -------- write outputs --------
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    fieldnames = ["disease", "n_profiles", "n_notes", "n_profiles_train_only",
                  "n_kg_sym", "n_validated", "validation_rate",
                  "n_validated_naive", "validation_rate_naive",
                  "n_real_grounded", "n_real_grounded_train_only", "n_new_real_not_in_kg"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in sorted(summary_rows, key=lambda x: -x["n_profiles"]):
            w.writerow(row)

    # -------- console summary --------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"OOG diseases (total)                       : {len(oog)}")
    print(f"OOG diseases with symptom edges            : {len(disease_kg_sym)}")
    print(f"OOG diseases with >=1 real MIMIC note      : {len([d for d in disease_kg_sym if result[d]['n_notes']>0])}")
    print(f"OOG diseases with NO real profile          : {len(no_profile)}")
    print(f"  (missing: {no_profile})")
    edges_with_note = sum(len(result[d]['kg_symptoms']) for d in result if result[d]['n_notes'] > 0)
    print(f"Overall KG symptom edges (all 190)         : {total_edges}")
    print(f"  ...of which on diseases WITH >=1 note     : {edges_with_note}")
    print(f"Validated NEG-AWARE (primary):")
    print(f"    vs all {total_edges} edges              : {total_validated}  ({round(100*total_validated/total_edges,1)}%)")
    print(f"    vs {edges_with_note} validatable edges  : {total_validated}  ({round(100*total_validated/edges_with_note,1)}%)")
    print(f"Validated NAIVE substring (inflated, incl. negated mentions):")
    print(f"    vs all {total_edges} edges              : {total_validated_naive}  ({round(100*total_validated_naive/total_edges,1)}%)")
    print(f"    vs {edges_with_note} validatable edges  : {total_validated_naive}  ({round(100*total_validated_naive/edges_with_note,1)}%)")

    # train-only coverage
    n_train_cov = len([d for d in result if result[d]['n_profiles_train_only'] > 0])
    print(f"OOG diseases with >=1 TRAIN-ONLY note      : {n_train_cov}")

    print("\nWorst-validated diseases (>=3 kg syms, lowest rate):")
    worst = sorted([r for r in summary_rows if r["n_kg_sym"] >= 3 and r["n_notes"] > 0],
                   key=lambda x: (x["validation_rate"], -x["n_kg_sym"]))[:15]
    for r in worst:
        print(f"  {r['validation_rate']:.2f}  {r['n_validated']:>2}/{r['n_kg_sym']:<2} "
              f"n_notes={r['n_notes']:<4} {r['disease']}")

    print(f"\n[out] {OUT_JSON}")
    print(f"[out] {OUT_CSV}")

    # -------- spot-check a few diseases (auditable) --------
    print("\n" + "=" * 70)
    print("SPOT-CHECKS (HPI excerpt + KG symptom decisions)")
    print("=" * 70)
    spot = ["deep vein thrombosis", "acute kidney injury", "cellulitis",
            "amyotrophic lateral sclerosis", "tenosynovitis"]
    spot = [s for s in spot if s in result] or list(result.keys())[:3]
    for d in spot:
        r = result[d]
        sup = r["kg_symptom_patient_support"]
        neg_only = [s for s in r["validated_naive_substring"] if s not in set(r["validated"])]
        print(f"\n### {d}  (n_profiles={r['n_profiles']}, n_notes={r['n_notes']})")
        print(f"  KG symptoms ({len(r['kg_symptoms'])}): {r['kg_symptoms']}")
        print(f"  VALIDATED neg-aware ({len(r['validated'])}): "
              f"{[(s, sup[s]) for s in r['validated']]}")
        print(f"  NOT FOUND ({len(r['not_found'])}): {r['not_found']}")
        print(f"  NAIVE-ONLY (matched but ONLY as negated/pertinent-negative) ({len(neg_only)}): {neg_only}")
        print(f"  Top real-grounded (affirmed patient freq): "
              f"{[(g['name'], g['freq']) for g in r['real_grounded_symptoms'][:12]]}")
        # print one HPI excerpt
        hs = sorted(disease2hadm.get(d, set()))
        hs = [h for h in hs if h in hadm2hpi][:1]
        if hs:
            print(f"  HPI excerpt [{hs[0]}]: {hadm2hpi[hs[0]][:500]}")

    return result


if __name__ == "__main__":
    main()
