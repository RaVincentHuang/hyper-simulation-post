"""SENTLI sequence-to-sequence labels for query-document entailment."""

import sentencepiece
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from hyper_simulation.query_instance import QueryInstance
from typing import List
class SENTLIScorer:
    """Lazily initialized process-wide SENTLI model scorer."""

    _instance = None
    def __new__(cls, model_path):
        """Return the process-wide scorer instance."""

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    def __init__(self, model_path):
        """Load the tokenizer and model once on the best available device."""

        # Later construction calls intentionally reuse the first loaded model.
        if self._initialized: return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self._initialized = True
    def predict(self, hypothesis: str, premise: str) -> str:
        """Generate one entailment label for a hypothesis-premise pair."""

        input_text = f"entailment: {hypothesis} [SEP] {premise}"
        inputs = self.tokenizer(input_text, max_length=512, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=3, num_beams=1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
def query_fixup(query_instance: QueryInstance, model_path: str = "/home/vincent/hyper-simulation/models/SENTLI") -> QueryInstance:
    """Return a copied query instance with each document prefixed by its label."""

    scorer = SENTLIScorer(model_path)
    fixed_data = []
    hypothesis = query_instance.query
    for doc in query_instance.data:
        # Keep the historical character-level cap before tokenizer truncation.
        doc_truncated = doc[:512] if len(doc) > 512 else doc
        label = scorer.predict(hypothesis, doc_truncated)
        fixed_doc = f"[SENTLI: {label}] {doc}"
        fixed_data.append(fixed_doc)
    from copy import deepcopy
    # Baseline annotations must not mutate the caller's original retrieval rows.
    fixed_instance = deepcopy(query_instance)
    fixed_instance.fixed_data = fixed_data
    return fixed_instance
