"""KGCQ: Knowledge-Graph-grounded Clarifying Question generation for conversational diagnosis.

Library modules (ported from the original MedQA/main/ code, logic preserved):
  paths              repository paths (ROOT, DATA_DIR, PROMPT_DIR, ...)
  utils              json/csv helpers, seeding
  graph              DiagnosticKnowledgeGraph (nodes/edges CSV -> networkx), subgraph text linearization
  subgraph_extractor hypothesis-driven subgraph extraction (3-hop + tau, 2-hop symptom-anchored, oracle)
  models             LLM wrappers (OpenAI / OpenRouter / local HF), EmbeddingModel, DiseaseDetector (HG)
  simulator          PatientSim-based patient simulator (+ low-specificity augmentation), HV response parsing
  symptom_extractor  dialogue-state symptom extraction (used by the 2-hop "+KG only" ablation)
  pipeline           ConversationalDiagnosis loops (KGCQ, KG-only, no-KG, generative-HG, synthetic generation)
  metrics            Recall@k and run-level aggregation
"""
__version__ = "1.0.0"
