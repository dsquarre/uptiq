import re
import time
from collections import Counter

from src.common.schemas import Prediction, QASample


_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _split_chunks(context: str) -> list[str]:
    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+", context) if part.strip()]
    return chunks if chunks else [context]


def _score_overlap(question: str, chunk: str) -> int:
    q_tokens = Counter(_tokenize(question))
    c_tokens = Counter(_tokenize(chunk))
    return sum((q_tokens & c_tokens).values())


def _extract_answer(question: str, contexts: list[str]) -> str:
    if not contexts:
        return ""
    scored = sorted(contexts, key=lambda c: _score_overlap(question, c), reverse=True)
    best = scored[0]
    return best[:240].strip()


class SimpleRAGAgent:
    name = "simple_rag"

    def __init__(self, top_k: int = 3):
        self.top_k = top_k

    def predict(self, sample: QASample) -> Prediction:
        start = time.perf_counter()
        chunks = _split_chunks(sample.context)
        ranked = sorted(chunks, key=lambda c: _score_overlap(sample.question, c), reverse=True)
        retrieved = ranked[: self.top_k]
        answer = _extract_answer(sample.question, retrieved)
        latency_ms = (time.perf_counter() - start) * 1000
        return Prediction(
            id=sample.id,
            architecture=self.name,
            question=sample.question,
            predicted_answer=answer,
            gold_answers=sample.answers,
            retrieved_contexts=retrieved,
            latency_ms=latency_ms,
            trace={"steps": ["retrieve", "answer"], "top_k": self.top_k},
        )
