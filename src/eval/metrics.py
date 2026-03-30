import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    if not ground_truths:
        ground_truths = [""]
    return max(metric_fn(prediction, truth) for truth in ground_truths)


def recall_at_k(retrieved_contexts: list[str], ground_truths: list[str], k: int) -> float:
    if not ground_truths:
        return 1.0
    top_contexts = retrieved_contexts[:k]
    normalized_contexts = [normalize_answer(c) for c in top_contexts]
    for truth in ground_truths:
        t = normalize_answer(truth)
        if t and any(t in context for context in normalized_contexts):
            return 1.0
    return 0.0


def mrr(retrieved_contexts: list[str], ground_truths: list[str]) -> float:
    if not ground_truths:
        return 1.0
    normalized_contexts = [normalize_answer(c) for c in retrieved_contexts]
    normalized_truths = [normalize_answer(t) for t in ground_truths if normalize_answer(t)]
    for index, context in enumerate(normalized_contexts, start=1):
        if any(t in context for t in normalized_truths):
            return 1.0 / index
    return 0.0
