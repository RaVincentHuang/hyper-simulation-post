import os
import random
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import torch
from torch import Tensor
import torch.nn.functional as F
import numpy as np
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
_model_cache = {}
max_length = 8192
_DEFAULT_SEED = int(os.environ.get("SC_SEED", "42"))
random.seed(_DEFAULT_SEED)
np.random.seed(_DEFAULT_SEED)
torch.manual_seed(_DEFAULT_SEED)
torch.cuda.manual_seed_all(_DEFAULT_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True, warn_only=True)
except Exception:
    pass
def init_embedding_model():
    if "Qwen/Qwen3-Embedding-0.6B" not in _model_cache:
        model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')
        model.eval()
        _model_cache["Qwen/Qwen3-Embedding-0.6B"] = model
def _get_sentence_transformer() -> SentenceTransformer:
    if "Qwen/Qwen3-Embedding-0.6B" not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/c54f2e6e80b2d7b7de06f51cec4959f6b3e03418"
        model = SentenceTransformer(local_model_path)
        model.eval()
        _model_cache["Qwen/Qwen3-Embedding-0.6B"] = model
    return _model_cache["Qwen/Qwen3-Embedding-0.6B"]
def get_embedding_batch(texts: list[str], batch_size: int=256, cache: None | dict[str, np.ndarray]=None) -> list[np.ndarray]:
    model = _get_sentence_transformer()
    if cache is None:
        cache = {}
    unique_texts = list(set(texts))
    missing_texts = [t for t in unique_texts if t not in cache]
    if missing_texts:
        new_embeddings = model.encode(
            missing_texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        cache.update(zip(missing_texts, new_embeddings))
    return [cache[t] for t in texts]
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return float(np.dot(a_norm, b_norm))
def get_similarity_batch(query: list[str], data: list[str], N: int=8) -> list[float]:
    query_embeddings = get_embedding_batch(query, N)
    data_embeddings = get_embedding_batch(data, N)
    pairs = [(q_emb, d_emb) for q_emb in query_embeddings for d_emb in data_embeddings]
    similarities = get_cosine_similarity_batch(pairs, is_normalized=True)
    return similarities
def get_similarity(text1: str, text2: str) -> float:
    emb1 = get_embedding_batch([text1])[0]
    emb2 = get_embedding_batch([text2])[0]
    return cosine_similarity(emb1, emb2)
def get_cosine_similarity_batch(pairs: list[tuple[np.ndarray, np.ndarray]], is_normalized: bool=False) -> list[float]:
    if not pairs:
        return []
    A_list, B_list = zip(*pairs)
    A = np.stack(A_list)
    B = np.stack(B_list)
    if is_normalized:
        similarities = np.einsum('ij,ij->i', A, B)
        return similarities.tolist()
    A_norms = np.linalg.norm(A, axis=1, keepdims=True)
    B_norms = np.linalg.norm(B, axis=1, keepdims=True)
    A_norms[A_norms == 0] = 1e-10
    B_norms[B_norms == 0] = 1e-10
    A_normalized = A / A_norms
    B_normalized = B / B_norms
    similarities = np.einsum('ij,ij->i', A_normalized, B_normalized)
    return similarities.tolist()