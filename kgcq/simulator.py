"""Patient simulator (PatientSim + low-specificity symptom reporting) and HV response generation/parsing."""
import json
import random
import re
from functools import lru_cache

from .paths import PERSONA_JSON


@lru_cache(maxsize=4)
def load_persona(persona_file=None):
    with open(str(persona_file or PERSONA_JSON), 'r') as f:
        return json.load(f)


def generate_patient_response(prompt, messages, generate_func, patient_profile):
    """Persona-free simulator (full history as chat). Kept for completeness; not used in the paper."""
    assert messages[-1]['role'] == 'assistant', "The last message should be from the Doctor"
    patient_profile['disease_combined'] = ', '.join(patient_profile['disease_mapped'])
    dialogue_history = [{"role": "system", "content": prompt.format(**patient_profile)}]
    for turn in messages:
        if turn["role"] == "assistant":
            dialogue_history.append({"role": "user", "content": turn['content']})
        elif turn["role"] == "user":
            dialogue_history.append({"role": "assistant", "content": turn['content']})
    return generate_func(dialogue_history, temperature=0.001)


def _fill_persona_fields(patient_profile, bias_prompt_dict):
    """Shared PatientSim persona construction (language proficiency, personality, recall, dazedness)."""
    cefr_type = patient_profile['cefr_type']
    personality_type = patient_profile['personality_type']
    recall_level_type = patient_profile['recall_level_option']
    dazed_level_type = patient_profile['dazed_level_option']

    num_word_sample = 3
    cefr_levels = ["A", "B", "C"]
    current_index = cefr_levels.index(cefr_type)
    higher_level = cefr_levels[current_index + 1] if cefr_type != "C" else None
    patient_profile["understand_med_words"] = ", ".join(random.sample(bias_prompt_dict["cefr_level_word"][cefr_type], num_word_sample))
    patient_profile["misunderstand_med_words"] = ", ".join(random.sample(bias_prompt_dict["cefr_level_word"][higher_level], num_word_sample)) if higher_level else ""

    parts = bias_prompt_dict["cefr_level"][cefr_type].split("\n\t")
    patient_profile["cefr"] = "\n\t\t" + "\n\t\t\t".join(parts[1:]).format(**patient_profile)

    parts = bias_prompt_dict["personality"][personality_type].split("\n\t")
    patient_profile["personality"] = "\n\t\t" + "\n\t\t".join(parts[1:])
    if personality_type != "plain":
        patient_profile["personality"] += "\n\t\tIMPORTANT: Ensure that your personality is clearly represented throughout the conversation, while allowing your emotional tone and style to vary naturally across turns."

    parts = bias_prompt_dict["recall_level"][recall_level_type].split("\n\t")
    patient_profile["memory_recall_level"] = f"{recall_level_type.capitalize()}\n\t\t" + "\n\t\t".join(parts[1:])

    dazed_levels = ["high", "moderate", "normal"]
    dazed_states = ["initial", "intermediate", "later"]
    if dazed_level_type not in dazed_levels:
        dazed_level_type = 'normal'
    dazed_index = dazed_levels.index(dazed_level_type)
    if dazed_level_type != "normal":
        dazed_description = (
            f"\n\tThe patient's initial dazed level is {dazed_level_type}. "
            "The dazedness should gradually fade throughout the conversation as the doctor continues to reassure them. "
            "Transitions should feel smooth and natural, rather than abrupt. "
            "While the change should be subtle and progressive, the overall dazed level is expected to decrease noticeably every 4-5 turns, following the instructions for each level below."
        )
        for i in range(dazed_index, len(dazed_levels)):
            level, state = dazed_levels[i], dazed_states[i]
            parts = bias_prompt_dict["dazed_level"][level].split("\n\t")
            dazed_description += f"\n\t{level.capitalize()} Dazedness ({state.capitalize()} Phase)\n\t\t" + "\n\t\t".join(parts[1:])
        dazed_description += "\n\tNote: Dazedness reflects the patient's state of confusion and inability in following the conversation, independent of their language proficiency."
    else:
        parts = bias_prompt_dict["dazed_level"][dazed_level_type].split("\n\t")
        dazed_description = f"{dazed_level_type.capitalize()}\n\t\t" + "\n\t\t".join(parts[1:])
    patient_profile["dazed_level"] = dazed_description

    cefr_short = bias_prompt_dict["cefr_level"][cefr_type].split("\n\t")[0]
    pers_short = bias_prompt_dict["personality"][personality_type].split("\n\t")[0]
    recall_short = bias_prompt_dict["recall_level"][recall_level_type].split("\n\t")[0].lower()
    dazed_short = bias_prompt_dict["dazed_level"][dazed_level_type].split("\n\t")[0]
    reminder = "You should act like " + cefr_short + " You are " + pers_short + ". Also, you " + recall_short + " " + dazed_short
    patient_profile["sent_limit"] = bias_prompt_dict["sentence_limit"].get(personality_type, "3")
    patient_profile['disease_combined'] = ', '.join(patient_profile['disease_mapped'])
    return reminder


def generate_patient_response_w_persona_org(prompt, messages, generate_func, patient_profile, persona_file=None):
    """Original PatientSim (4 persona dimensions, no specificity trait). Baseline simulator."""
    assert messages[-1]['role'] == 'assistant', "The last message should be from the Doctor"
    bias_prompt_dict = load_persona(persona_file)
    patient_profile["reminder"] = _fill_persona_fields(patient_profile, bias_prompt_dict)
    dialogue_history = [
        {"role": "system", "content": prompt.format(**patient_profile)},
        {"role": "user", "content": messages[-1]['content']},
    ]
    return generate_func(dialogue_history, temperature=0.8)


def generate_patient_response_w_persona(prompt, messages, generate_func, patient_profile, persona_file=None):
    """Our simulator: PatientSim persona + low-specificity symptom description (paper Sec. 3.4.2).

    Specificity prompt = base low-specificity instruction + persona-compatibility add-ons:
      high recall -> recall applies to past history only; proficiency C -> self-diagnoser;
      verbose personality -> masks missing details with irrelevant text.
    """
    assert messages[-1]['role'] == 'assistant', "The last message should be from the Doctor"
    bias_prompt_dict = load_persona(persona_file)
    reminder = _fill_persona_fields(patient_profile, bias_prompt_dict)

    specificity_desc = bias_prompt_dict['specificity_base']['low']
    if patient_profile['recall_level_option'] == "high":
        specificity_desc += "\n\t\t" + bias_prompt_dict["specificity_addon_recall"]["high"]
    if patient_profile['cefr_type'] == "C":
        specificity_desc += "\n\t\t" + bias_prompt_dict["specificity_addon_proficiency"]["C"]
    if patient_profile['personality_type'] == "verbose":
        specificity_desc += "\n\t\t" + bias_prompt_dict["specificity_addon_personality"]["verbose"]
    patient_profile["symptom_specificity"] = specificity_desc
    patient_profile["reminder"] = reminder + " Lastly, you struggle to provide specific details when describing your present symptoms."

    dialogue_history = [
        {"role": "system", "content": prompt.format(**patient_profile)},
        {"role": "user", "content": messages[-1]['content']},
    ]
    return generate_func(dialogue_history, temperature=0.8)


def check_duplicate(question, pure_messages, generate_func):
    """LLM check used during synthetic dialogue generation to avoid repeated questions."""
    prompt = """The following are questions previously asked by the doctor to the patient.
Previous questions: 
{previous_questions}

Please determine whether the current question is semantically overlapping with any of the previous questions.
Current question: {current_question}

- Output "Yes" if it is the same question; otherwise, output "No" only.
"""
    inputs = {
        'previous_questions': '\n'.join([msg['content'] for msg in pure_messages if msg['role'] == 'assistant']),
        'current_question': question,
    }
    response = generate_func([{"role": "user", "content": prompt.format(**inputs)}], temperature=0.0001)
    return 'Yes' in response


def strip_quotes(s):
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1].strip()
    return s


def parse_response(s):
    """Return (action, content): action in {question, diagnosis, error}."""
    question_match = re.search(r'<question>(.*?)</question>', s, re.DOTALL)
    diagnosis_match = re.search(r'<diagnosis>(.*?)</diagnosis>', s, re.DOTALL)
    if question_match:
        return "question", strip_quotes(question_match.group(1))
    if diagnosis_match:
        return "diagnosis", strip_quotes(diagnosis_match.group(1))
    return "error", s


def dialogue_to_text(messages):
    lines = []
    for turn in messages:
        if turn['role'] == 'user':
            lines.append(f"Patient: {turn['content']}")
        elif turn['role'] == 'assistant':
            lines.append(f"Doctor: {turn['content']}")
    return "\n".join(lines)


def generate_doctor_response(prompt, messages, generate_func, inputs, temperature=0.0001):
    """HV call: prompt.format(dialogue_text=..., **inputs) as a single user message."""
    assert messages[-1]['role'] == 'user', "The last message should be from the Patient"
    inputs = dict(inputs)
    inputs['dialogue_text'] = dialogue_to_text(messages)
    response = generate_func([{"role": "user", "content": prompt.format(**inputs)}], temperature=temperature)
    action, content = parse_response(response)
    return action, response, content


def generate_doctor_response_multiturn(prompt, messages, generate_func, inputs):
    """Chat-style HV call (system prompt + alternating turns with full_response); no-KG baseline."""
    assert messages[-1]['role'] == 'user', "The last message should be from the Patient"
    messages_trimmed = messages[1:] if messages[0]['role'] == 'assistant' else messages[:]
    new_messages = []
    for turn in messages_trimmed:
        if turn['role'] == 'user':
            new_messages.append({"role": "user", "content": turn['content']})
        elif turn['role'] == 'assistant':
            new_messages.append({"role": "assistant", "content": turn['full_response']})
    dialogue_history = [{"role": "system", "content": prompt.format(**inputs)}] + new_messages
    response = generate_func(dialogue_history, temperature=0)
    action, content = parse_response(response)
    return action, response, content
