#!/usr/bin/env python
"""Build the pairwise patient-simulator evaluation set (paper Sec. 6.3).

For each dialogue produced by KGCQ with the baseline PatientSim (dialog_anal.json of a run whose patient prompt was
prompts/patientsim/patient_original_patientsim.txt), cut the history at a random doctor turn and regenerate the
next patient response with (a) the original PatientSim and (b) our specificity-augmented simulator under the same
context. Pairs whose two responses are near-identical (cosine similarity >= 0.55 with all-MiniLM-L6-v2) are
dropped; the rest are randomised into "Response A/B" for blinded physician rating.

  python scripts/human_eval/build_simulator_eval_set.py --run runs/kgcq_original_patientsim --output data/human_eval/simulator_eval/to_eval.json
"""
import argparse
import os
import random
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402
from tqdm import tqdm  # noqa: E402

from kgcq import paths  # noqa: E402
from kgcq.utils import load_json, save_to_json, file_to_string  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run dir with dialog_anal.json (summarize_run.py)")
    ap.add_argument("--profiles", default=str(paths.PROFILE_DIR / "test_balanced_275.json"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--sim_threshold", type=float, default=0.55)
    ap.add_argument("--translate_ko", action="store_true", help="append Korean translations for the raters")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    from kgcq.models import get_openai_response, EmbeddingModel
    from kgcq.simulator import generate_patient_response_w_persona, generate_patient_response_w_persona_org

    profiles = load_json(args.profiles)
    dialogues = load_json(os.path.join(args.run, "dialog_anal.json"))
    prompt_org = file_to_string(paths.PATIENT_PROMPT_ORIGINAL)
    prompt_new = file_to_string(paths.PATIENT_PROMPT)
    emb = EmbeddingModel(paths.EMBEDDING_MODEL)

    def gen(messages, temperature=0.0):
        return get_openai_response(args.model, messages, temperature=0.001)

    def ko(text, role):
        if not args.translate_ko:
            return ""
        return "(" + gen([{"role": "user", "content": f"Translate the following {role}'s utterance in Korean. Output only the translation:\n{text}"}]) + ")"

    items = []
    for d in tqdm(dialogues):
        ctx = d['dialogue']
        if len(ctx) - 1 < 3:
            continue
        end = random.choice(list(range(3, len(ctx) - 1, 2)))
        messages = ctx[:end]
        profile = profiles[d['patient_id']]
        resp_org = generate_patient_response_w_persona_org(prompt_org, messages, gen, dict(profile))
        resp_new = generate_patient_response_w_persona(prompt_new, messages, gen, dict(profile))
        sim = cosine_similarity(emb.encode([resp_org]), emb.encode([resp_new])).flatten()[0]
        if sim >= args.sim_threshold:
            continue
        context = [("Patient: " if t['role'] == 'user' else "Doctor: ") + t['content'] + ko(t['content'], 'patient' if t['role'] == 'user' else 'doctor')
                   for t in messages]
        items.append({"patient_id": d['patient_id'], "persona": {k: profile[k] for k in ['cefr_type', 'personality_type', 'recall_level_option', 'dazed_level_option']},
                      "context": context, "response_original_patientsim": resp_org + ko(resp_org, 'patient'),
                      "response_ours": resp_new + ko(resp_new, 'patient')})
        save_to_json(items, args.output)

    rows = []
    for it in items:
        ours_first = random.random() < 0.5
        rows.append({"Dialogue History": "\n".join(it['context'][1:]),
                     "Response A": it['response_ours'] if ours_first else it['response_original_patientsim'],
                     "Response B": it['response_original_patientsim'] if ours_first else it['response_ours'],
                     "ours": "A" if ours_first else "B"})
    pd.DataFrame(rows).to_csv(os.path.splitext(args.output)[0] + "_blinded.csv", index=False)
    print(f"{len(items)} pairs -> {args.output}")


if __name__ == "__main__":
    main()
