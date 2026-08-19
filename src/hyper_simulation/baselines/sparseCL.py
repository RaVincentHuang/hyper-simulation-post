"""SparseCL similarity annotations for query-document retrieval baselines."""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from hyper_simulation.query_instance import QueryInstance
class SparseCLScorer:
    """Process-wide SparseCL encoder with cosine and Hoyer scoring."""

    _instance = None
    def __new__(cls, model_path: str, device: str = "cuda"):
        """Return the singleton scorer used to avoid repeated model loading."""

        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    def __init__(self, model_path: str, device: str = "cuda"):
        """Initialize the first requested local model and device."""

        # Model path and device are frozen by the first construction in a process.
        if self._initialized:
            return
        self.device = device
        self.model_path = model_path
        self._load_model()
        self._initialized = True
    def _load_model(self):
        """Load tokenizer and encoder strictly from local model files."""

        print(f"[SparseCL] {self.model_path} {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True, local_files_only=True).to(self.device)
        self.model.eval()
    def _mean_pooling(self, model_output, attention_mask):
        """Mean-pool token embeddings while excluding padded positions."""

        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    def get_embedding(self, text: str):
        """Encode and L2-normalize one text string."""

        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = self._mean_pooling(outputs, inputs['attention_mask'])
        return F.normalize(embedding, p=2, dim=1)
    def _calc_hoyer_sparsity(self, v1: torch.Tensor, v2: torch.Tensor):
        """Measure Hoyer sparsity of the difference between two embeddings."""

        diff = v1 - v2
        d = diff.shape[1]
        sqrt_d = torch.sqrt(torch.tensor(d, device=self.device))
        l1_norm = torch.norm(diff, p=1, dim=1)
        l2_norm = torch.norm(diff, p=2, dim=1)
        l2_norm = torch.clamp(l2_norm, min=1e-9)
        hoyer = (sqrt_d - (l1_norm / l2_norm)) / (sqrt_d - 1)
        return hoyer
    def compute_score(self, text_a: str, text_b: str, alpha: float = 1.0) -> float:
        """Combine cosine similarity with weighted difference sparsity."""

        emb_a = self.get_embedding(text_a)
        emb_b = self.get_embedding(text_b)
        cosine_score = torch.sum(emb_a * emb_b, dim=1)
        hoyer_score = self._calc_hoyer_sparsity(emb_a, emb_b)
        final_score = cosine_score + alpha * hoyer_score
        return final_score.item()
def query_fixup(
    query_instance: QueryInstance, 
    model_path: str = "/home/vincent/hyper-simulation/models/GTE-SparseCL-msmarco", 
    alpha: float = 1.5
) -> QueryInstance:
    """Return a copied query instance with SparseCL scores on each document."""

    import logging
    logger = logging.getLogger(__name__)
    scorer = SparseCLScorer(model_path)
    query = query_instance.query
    documents = query_instance.data if query_instance.data else []
    if not documents:
        # Preserve object identity for the no-retrieval fast path.
        return query_instance
    fixed_data = []
    for idx, doc in enumerate(documents):
        try:
            score = scorer.compute_score(query, doc, alpha=alpha)
            fixed_doc = f"[SparseCL: {score:.2f}] {doc}"
            fixed_data.append(fixed_doc)
        except Exception as e:
            # A per-document model failure retains the original document.
            logger.warning(f"[SparseCL]  {idx} {e}")
            fixed_data.append(doc)
    from copy import deepcopy
    # Keep annotations isolated from the source query instance.
    fixed_instance = deepcopy(query_instance)
    fixed_instance.fixed_data = fixed_data
    logger.info(f"[SparseCL] {len(documents)}")
    return fixed_instance
