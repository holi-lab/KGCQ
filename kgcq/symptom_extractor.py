"""Dialogue-state symptom extraction (one doctor question + one patient answer -> confirmed symptoms).

Used by the HG-free "+KG" ablation (2-hop expansion from attribute nodes similar to the patient utterance).
"""

SYMPTOM_EXTRACTOR_PROMPT = """You are a medical dialogue state extractor.

Your task is to read one doctor–patient turn (one question from the doctor and one answer from the patient) and output the symptoms that the patient explicitly confirms as present.

### Output Rules
- Output only the symptoms the patient confirms as present.
- If the patient denies a symptom or says they don’t know / not sure, output: none
- If the patient confirms multiple symptoms, output them as EHR-style entities separated by semicolons (;).
- Normalize symptoms to clinical terms (e.g., “throat feels scratchy” → “throat irritation”).

### Output Format
A single line containing either:
- none, or
- one or more symptom entities separated by ;

### Examples
#### Input:
Doctor: How long has your abdominal pain lasted?
Patient: About three days.
#### Output:
abdominal pain

#### Input:
Doctor: Do you have a cough or sore throat?
Patient: I don’t have a cough, but my throat feels scratchy.
#### Output:
throat irritation

#### Input:
Doctor: Do you have fever?
Patient: I don’t know.
#### Output:
none

#### Input:
Doctor: Hello, how can I help you today?
Patient: I've been feeling really off lately.
#### Output:
fatigue;malaise

### Your Task
#### Input:
{dialogue_text}
#### Output:
"""


def create_prompt(messages):
    assert messages[0]['role'] == 'assistant', "first message must be the doctor turn"
    assert messages[1]['role'] == 'user', "second message must be the patient turn"
    return f"Doctor: {messages[0]['content']}\nPatient: {messages[1]['content']}"


def extract_symptom_state(generate_func, messages):
    dialogue_text = create_prompt(messages)
    response = generate_func([{"role": "user", "content": SYMPTOM_EXTRACTOR_PROMPT.format(dialogue_text=dialogue_text)}],
                             temperature=0.01)
    if 'none' in response.lower():
        return None
    return response.split(';')
