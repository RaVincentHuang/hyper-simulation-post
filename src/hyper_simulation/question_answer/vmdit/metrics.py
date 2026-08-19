"""Normalization and answer-quality metrics for legacy QA evaluation."""

import numpy as np
import string
import re
from collections import Counter
import re
def exact_match_score(prediction, ground_truth):
    """Return whether normalized prediction and ground truth are identical."""

    return (normalize_answer(prediction) == normalize_answer(ground_truth))
def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Take the best metric score across acceptable ground-truth answers."""

    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)
def accuracy(preds, labels):
    """Compute percentage accuracy against the first label per example."""

    match_count = 0
    for pred, label in zip(preds, labels):
        target = label[0]
        if pred == target:
            match_count += 1
    return 100 * (match_count / len(preds))
def f1(decoded_preds, decoded_labels):
    """Compute mean best-reference token F1 as a percentage."""

    f1_all = []
    for prediction, answers in zip(decoded_preds, decoded_labels):
        if type(answers) == list:
            if len(answers) == 0:
                return 0
            f1_all.append(np.max([qa_f1_score(prediction, gt)
                          for gt in answers]))
        else:
            f1_all.append(qa_f1_score(prediction, answers))
    return 100 * np.mean(f1_all)
def qa_f1_score(prediction, ground_truth):
    """Compute token-overlap F1 after answer normalization."""

    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
def normalize_answer(s):
    """Lowercase text and remove punctuation, articles, and excess spacing."""

    if isinstance(s, list):
        s = s[0] if len(s) > 0 else ""
    if not isinstance(s, str):
        s = str(s)
    def remove_articles(text):
        """Remove English articles unless the answer consists only of one."""

        if text.strip() in ['a', 'an', 'the']:
            return text
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        """Collapse consecutive whitespace."""

        return ' '.join(text.split())
    def remove_punc(text):
        """Remove ASCII punctuation characters."""

        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        """Lowercase text."""

        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))
def find_entity_tags(sentence):
    """Parse ``entity <tag>`` segments into an entity-to-tag mapping."""

    entity_regex = r'(.+?)(?=\s<|$)'
    tag_regex = r'<(.+?)>'
    entity_names = re.findall(entity_regex, sentence)
    tags = re.findall(tag_regex, sentence)
    results = {}
    for entity, tag in zip(entity_names, tags):
        if "<" in entity:
            results[entity.split("> ")[1]] = tag
        else:
            results[entity] = tag
    return results
def match(prediction, ground_truth):
    """Return one when any ground truth occurs in the prediction."""

    if not isinstance(prediction, str):
        prediction = str(prediction)
    for gt in ground_truth:
        if isinstance(gt, list):
            gt = gt[0] if gt else ""
        if not isinstance(gt, str):
            gt = str(gt)
        if not gt:
            continue
        pred_lower = prediction.lower()
        gt_lower = gt.lower()
        if len(gt) <= 2:
            if re.search(rf'\b{re.escape(gt_lower)}\b', pred_lower):
                return 1
        else:
            if gt_lower in pred_lower:
                return 1
    return 0
