"""Conversational diagnosis loops.

All managers share the same turn structure (paper Fig. 2):
    patient utterance -> [HG: disease probabilities] -> [subgraph extraction] -> HV: <question> or <diagnosis>
and the same bookkeeping: ``pure_messages`` (the plain dialogue) and ``messages`` (full trace with HG output,
subgraph lines and the HV's raw <think>... response).

ConversationalDiagnosis   KGCQ (kg_extract_type="3hop"), "+KG only" ablation ("2hop"), "1hop" variant.
NoKGDiagnosis             parametric-knowledge-only baseline (no HG, no graph).
GenerativeHGDiagnosis     generative Hypothesis Generator baseline (Appendix C.1).
SyntheticDialogueGenerator gold-conditioned clinician + simulator for training-data generation (Sec. 3.5.1).
"""
import random

from .metrics import fill_recall
from .simulator import (check_duplicate, generate_doctor_response, generate_doctor_response_multiturn,
                        generate_patient_response_w_persona)
from .subgraph_extractor import (extract_subgraph_1hop, extract_subgraph_2hop, extract_subgraph_3hop,
                                 extract_subgraph_3hop_clf_lm)
from .symptom_extractor import extract_symptom_state

GREETING = "Hello, how can I help you today?"
DIAGNOSIS_MATCH_THRESHOLD = 0.9   # free-text diagnosis -> KG disease node (cosine similarity)


class _BaseDiagnosis:
    """Shared state and diagnosis scoring."""

    def __init__(self, kg, doctor_prompt, patient_prompt, doctor_generate_func, patient_generate_func, max_turn=50):
        self.kg = kg
        self.doctor_prompt = doctor_prompt
        self.patient_prompt = patient_prompt
        self.generate_q = doctor_generate_func
        self.generate_p = patient_generate_func
        self.max_turn = max_turn
        self.reset_all()

    def reset_all(self):
        self.patient_profile = None
        self.pure_messages = [{"role": "assistant", "content": GREETING}]
        self.messages = []
        self.curr_turn = 0
        self.final_diagnosis = []
        self.recall = {"1": 0, "2": 0, "3": 0, "4": 0}
        self.current_subgraph = None
        self.current_subgraph_lines = []

    def set_profile(self, profile: dict):
        """``profile['disease']`` must hold the gold disease list (KG node names)."""
        self.patient_profile = profile

    def generate_patient_response(self):
        response = generate_patient_response_w_persona(
            prompt=self.patient_prompt, messages=self.pure_messages,
            generate_func=self.generate_p, patient_profile=self.patient_profile)
        self.pure_messages.append({"role": "user", "content": response})
        self.messages.append({"role": "user", "content": response})
        return response

    def score_diagnosis(self, utterance):
        """Map each comma-separated predicted disease to a KG node (sim > 0.9, else '') and fill Recall@k."""
        for u in utterance.lower().split(','):
            name, sim = self.kg.find_matching_disease(u)
            self.final_diagnosis.append(name if sim > DIAGNOSIS_MATCH_THRESHOLD else "")
        self.recall = fill_recall(self.patient_profile['disease'], self.final_diagnosis)

    def snapshot(self):
        return {
            "diagnosis": self.final_diagnosis,
            "recall": self.recall,
            "dialogue_length": self.curr_turn,
            "dialogue": self.pure_messages,
            "full_process": self.messages,
        }

    # --- hooks implemented by subclasses ---
    def start(self):
        """Called after the first patient utterance (HG + subgraph for KG-based variants)."""

    def after_patient_turn(self):
        """Called after each subsequent patient utterance."""

    def generate_doctor_response(self):
        raise NotImplementedError

    def process_action(self, action_json):
        if action_json['action'] == 'diagnosis':
            self.score_diagnosis(action_json['utterance'])
        else:
            self.generate_patient_response()
            self.after_patient_turn()
        self.curr_turn += 1

    def run(self, profile, max_turns=None, on_step=None):
        """Run one dialogue for ``profile`` (already containing 'disease'); returns snapshot()."""
        max_turns = self.max_turn if max_turns is None else max_turns
        self.reset_all()
        self.set_profile(profile)
        self.generate_patient_response()
        self.start()
        if on_step:
            on_step(self)
        while self.curr_turn <= max_turns:
            action_json = self.generate_doctor_response()
            self.process_action(action_json)
            if on_step:
                on_step(self)
            if self.final_diagnosis:
                break
        return self.snapshot()


class ConversationalDiagnosis(_BaseDiagnosis):
    """KGCQ: HG -> hypothesis-driven subgraph -> HV.

    kg_extract_type: "3hop" (KGCQ; top-n anchors + tau), "2hop" (no HG; symptom-anchored), "1hop".
    """

    def __init__(self, kg, doctor_prompt, patient_prompt, doctor_generate_func, patient_generate_func,
                 kg_extract_type="3hop", top_k=2, threshold_tau=0.005, disease_detector=None,
                 symptom_extract_func=None, max_turn=50, subgraph_fmt="disease", symptom_match_threshold=0.5):
        self.kg_extract_type = kg_extract_type
        self.top_k = top_k
        self.threshold_tau = threshold_tau
        self.disease_detector = disease_detector
        self.generate_s = symptom_extract_func or patient_generate_func
        self.subgraph_fmt = subgraph_fmt
        self.symptom_match_threshold = symptom_match_threshold
        if kg_extract_type in ("3hop", "1hop"):
            assert disease_detector is not None, f"{kg_extract_type} requires a DiseaseDetector (HG)"
        super().__init__(kg, doctor_prompt, patient_prompt, doctor_generate_func, patient_generate_func, max_turn)

    def reset_all(self):
        super().reset_all()
        self.current_gold_disease_rank = float('inf')
        self.current_diseases = []       # top-k (name, prob)
        self.current_all_diseases = []   # all (name, prob)
        self.curr_symptom_state = set()
        self.curr_symptom_nodes = set()
        self.curr_symptom_nodes_text = set()

    # ---- Step 1: hypothesis generation
    def detect_diseases(self):
        result, all_results = self.disease_detector.rank_diseases(
            messages=self.pure_messages[1:], gold_disease_list=self.patient_profile['disease'], top_k=self.top_k)
        self.messages.append({"role": "disease_prob", "content": {
            "gold_prob": result['gold_prob'], "gold_rank": result['gold_rank'],
            "top_k_diseases": result['ranked_list'], "entropy": result['entropy']}})
        self.current_all_diseases = all_results
        self.current_diseases = result['ranked_list']
        self.current_gold_disease_rank = result['gold_rank']

    # ---- Step 2: subgraph extraction
    def extract_subgraph(self):
        if self.kg_extract_type == "3hop":
            names = [d for d, _ in self.current_diseases]
            probs = {d: p for d, p in self.current_all_diseases}
            self.current_subgraph, self.current_subgraph_lines = extract_subgraph_3hop(
                self.kg, names, probs, self.threshold_tau, fmt=self.subgraph_fmt)
        elif self.kg_extract_type == "1hop":
            names = [d for d, _ in self.current_diseases]
            self.current_subgraph, self.current_subgraph_lines = extract_subgraph_1hop(self.kg, names, fmt=self.subgraph_fmt)
        elif self.kg_extract_type == "2hop":
            self._extract_subgraph_2hop()
        else:
            raise ValueError(self.kg_extract_type)
        self.messages.append({"role": "subgraph_extractor", "content": self.current_subgraph_lines})

    def _extract_subgraph_2hop(self):
        response = extract_symptom_state(self.generate_s, self.pure_messages[-2:])
        if response:
            self.curr_symptom_state.update([s.strip() for s in response])
        self.messages.append({"role": "symptom_extractor", "content": list(self.curr_symptom_state)})
        for symptom in self.curr_symptom_state:
            for e in self.kg.find_similar_nodes(symptom, top_k=self.top_k, threshold=self.symptom_match_threshold,
                                                node_subset=self.kg.symptom_ids):
                self.curr_symptom_nodes.add(e[0])
                self.curr_symptom_nodes_text.add(e[1])
        self.messages.append({"role": "symptom_node_extractor", "content": list(self.curr_symptom_nodes_text)})
        self.current_subgraph, self.current_subgraph_lines = extract_subgraph_2hop(
            self.kg, self.curr_symptom_nodes, fmt=self.subgraph_fmt)

    def _lines(self):
        return self.current_subgraph_lines if isinstance(self.current_subgraph_lines, list) else [self.current_subgraph_lines]

    # ---- Step 3: hypothesis verification
    def generate_doctor_response(self):
        inputs = {"max_turn": self.max_turn, "curr_turn": self.curr_turn + 1, "subgraph_text": '\n'.join(self._lines())}
        action, full_response, utterance = generate_doctor_response(
            prompt=self.doctor_prompt, messages=self.pure_messages, generate_func=self.generate_q, inputs=inputs)
        self.pure_messages.append({"role": "assistant", "content": utterance})
        self.messages.append({"role": "assistant", "action": action, "full_response": full_response})
        return {"action": action, "utterance": utterance, "full_response": full_response}

    def start(self):
        if self.kg_extract_type != "2hop":
            self.detect_diseases()
        self.extract_subgraph()

    after_patient_turn = start


class NoKGDiagnosis(_BaseDiagnosis):
    """Parametric-knowledge-only baseline: chat-style HV with no graph (Table 2, row 1)."""

    def generate_doctor_response(self):
        inputs = {"max_turn": self.max_turn, "curr_turn": self.curr_turn + 1}
        action, full_response, utterance = generate_doctor_response_multiturn(
            prompt=self.doctor_prompt, messages=self.messages, generate_func=self.generate_q, inputs=inputs)
        self.pure_messages.append({"role": "assistant", "content": utterance})
        self.messages.append({"role": "assistant", "action": action, "full_response": full_response})
        return {"action": action, "utterance": utterance, "full_response": full_response}


class GenerativeHGDiagnosis(_BaseDiagnosis):
    """Generative HG baseline: an LLM lists top-4 disease names; subgraph = anchors + diseases sharing >= 30%
    of an anchor's attributes (Appendix C.1)."""

    def __init__(self, kg, doctor_prompt, patient_prompt, doctor_generate_func, patient_generate_func,
                 disease_detector_func, disease_detector_prompt, n=4, max_turn=50, subgraph_fmt="disease"):
        self.disease_detector_func = disease_detector_func
        self.disease_detector_prompt = disease_detector_prompt
        self.n = n
        self.subgraph_fmt = subgraph_fmt
        super().__init__(kg, doctor_prompt, patient_prompt, doctor_generate_func, patient_generate_func, max_turn)

    def reset_all(self):
        super().reset_all()
        self.current_diseases = []

    def detect_diseases(self):
        lines = []
        for turn in self.pure_messages[1:]:
            if turn['role'] == 'user':
                lines.append(f"Patient: {turn['content']}")
            elif turn['role'] == 'assistant':
                lines.append(f"Doctor: {turn['content']}")
        inputs = {"dialogue_history": "\n".join(lines), "disease_names": "\n".join(self.kg.disease_names),
                  "disease_names_len": len(self.kg.disease_names), "n": self.n}
        response = self.disease_detector_func([{"role": "user", "content": self.disease_detector_prompt.format(**inputs)}],
                                              temperature=0.0001)
        pred_list = [self.kg.find_matching_disease(d.strip())[0] for d in response.split(',')]
        gt_list = self.patient_profile['disease']
        recall4 = len(set(gt_list) & set(pred_list)) / (len(gt_list) if gt_list else 1)
        self.messages.append({"role": "disease_detector", "content": {"top4_diseases": pred_list, "recall_at_4": recall4}})
        self.current_diseases = pred_list

    def extract_subgraph(self):
        self.current_subgraph, self.current_subgraph_lines = extract_subgraph_3hop_clf_lm(
            self.kg, self.current_diseases, fmt=self.subgraph_fmt)
        contain = int(any(d in l for l in self.current_subgraph_lines for d in self.patient_profile['disease']))
        self.messages.append({"role": "subgraph_extractor", "contain": contain, "content": self.current_subgraph_lines})

    def generate_doctor_response(self):
        inputs = {"max_turn": self.max_turn, "curr_turn": self.curr_turn + 1,
                  "subgraph_text": '\n'.join(self.current_subgraph_lines)}
        action, full_response, utterance = generate_doctor_response(
            prompt=self.doctor_prompt, messages=self.pure_messages, generate_func=self.generate_q, inputs=inputs)
        self.pure_messages.append({"role": "assistant", "content": utterance})
        self.messages.append({"role": "assistant", "action": action, "full_response": full_response})
        return {"action": action, "utterance": utterance, "full_response": full_response}

    def start(self):
        self.detect_diseases()
        self.extract_subgraph()

    after_patient_turn = start


class SyntheticDialogueGenerator(_BaseDiagnosis):
    """Gold-conditioned clinician (prompts/synth_doctor_gold.txt) + simulator -> training dialogues.

    The clinician sees the oracle subgraph (3-hop from the gold disease), the gold disease, and a hidden
    KG grounding ratio gamma = |found_symptoms| / |gold_symptoms| that steers how strongly it relies on the
    graph (Appendix B). A soft turn budget max_turn ~ U[10-LR, 13-LR] is drawn from the profile's
    likelihood rating LR; duplicate questions are re-sampled (up to 5 times) with temperature 1.0.
    """

    def __init__(self, kg, doctor_prompt, patient_prompt, doctor_generate_func, patient_generate_func,
                 duplicate_check_func=None, max_turn=50):
        self.duplicate_check_func = duplicate_check_func or patient_generate_func
        super().__init__(kg, doctor_prompt, patient_prompt, doctor_generate_func, patient_generate_func, max_turn)

    def set_profile(self, profile: dict):
        self.patient_profile = profile
        gold_symptoms = profile.get('gold_symptoms') or []
        self.relevance_score = len(profile.get('found_symptoms') or []) / len(gold_symptoms) if gold_symptoms else 0.0
        start = 10 - profile['likelihood_rating']
        self.soft_max_turn = random.randint(start, start + 3)

    def extract_subgraph(self):
        self.current_subgraph_lines = self.patient_profile['subgraph']

    def generate_doctor_response(self):
        inputs = {
            "gold_disease": ', '.join(self.patient_profile['disease']),
            "subgraph_text": '\n'.join(self.current_subgraph_lines),
            "relevance_score": self.relevance_score,
            "dialogue_length": self.curr_turn,
            "max_turn": self.soft_max_turn,
        }
        action, full_response, utterance = generate_doctor_response(
            prompt=self.doctor_prompt, messages=self.pure_messages, generate_func=self.generate_q, inputs=inputs)
        # Re-sample a duplicate question at temperature 1.0 (up to 5 times). Note: in the original data-generation
        # code the temperature argument was not propagated, so the released dialogues were re-sampled at 0.0001.
        for _ in range(5):
            if action == 'question' and check_duplicate(utterance, self.pure_messages, self.duplicate_check_func):
                action, full_response, utterance = generate_doctor_response(
                    prompt=self.doctor_prompt, messages=self.pure_messages, generate_func=self.generate_q,
                    inputs=inputs, temperature=1.0)
            else:
                break
        self.pure_messages.append({"role": "assistant", "content": utterance})
        self.messages.append({"role": "assistant", "action": action, "full_response": full_response})
        return {"action": action, "utterance": utterance, "full_response": full_response}

    def start(self):
        self.extract_subgraph()
