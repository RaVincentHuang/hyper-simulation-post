"""Cached cross-encoder helpers for NLI decisions and scores."""

import os
import torch
from sentence_transformers import CrossEncoder
_model_cache = {}
BATCH_SIZE = 1024
def init_nli_model():
    """Eagerly initialize the configured NLI cross-encoder."""

    _model_cache['nli-deberta-v3-base'] = CrossEncoder('cross-encoder/nli-deberta-v3-base')
def get_nli_labels_batch(pairs: list[tuple[str, str]]) -> list[str]:
    """Predict contradiction, entailment, or neutral for each ordered text pair."""

    if 'nli-deberta-v3-base' not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
    model = _model_cache['nli-deberta-v3-base']
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    return labels
def get_nli_labels_with_score_batch(pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Return each predicted label with its raw entailment score."""

    if 'nli-deberta-v3-base' not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
    model = _model_cache['nli-deberta-v3-base']
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    entailment_index = 1
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels_with_score = [(label_mapping[score_max], scores[i][entailment_index]) for i, score_max in enumerate(scores.argmax(axis=1))]
    return labels_with_score
def get_nli_label(text1: str, text2: str) -> str:
    """Predict one ordered text pair's NLI label."""

    labels = get_nli_labels_batch([(text1, text2)])
    return labels[0]
def get_nli_entailment_score_batch(pairs: list[tuple[str, str]]) -> list[float]:
    """Return raw entailment scores for ordered text pairs."""

    if 'nli-deberta-v3-base' not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
    model = _model_cache['nli-deberta-v3-base']
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    entailment_scores = [score[1] for score in scores]
    return entailment_scores
def get_nli_contradiction_score_batch(pairs: list[tuple[str, str]]) -> list[float]:
    """Return raw contradiction scores for ordered text pairs."""

    if 'nli-deberta-v3-base' not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
    model = _model_cache['nli-deberta-v3-base']
    scores = model.predict(pairs, batch_size=BATCH_SIZE)
    contradiction_scores = [score[0] for score in scores]
    return contradiction_scores
def get_nli_remix_score_batch(pairs: list[tuple[str, str]], to_refine: bool = False) -> list[float]:
    """Return entailment probabilities, optionally zeroing predicted contradictions."""

    if 'nli-deberta-v3-base' not in _model_cache:
        local_model_path = "/home/vincent/.cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base/snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
        _model_cache['nli-deberta-v3-base'] = CrossEncoder(local_model_path)
    if not pairs:
        return []
    model = _model_cache['nli-deberta-v3-base']
    logits = model.predict(pairs, convert_to_tensor=True, batch_size=BATCH_SIZE)
    probs = torch.softmax(logits, dim=1)
    labels_idx = torch.argmax(probs, dim=1)
    entailment_probs = probs[:, 1]
    if to_refine:
        pair_scores = torch.where(labels_idx == 0, torch.zeros_like(entailment_probs), entailment_probs)
    else:
        pair_scores = entailment_probs
    return pair_scores.detach().cpu().tolist()
