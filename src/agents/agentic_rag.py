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


def _score_overlap(query: str, chunk: str) -> int:
    q_tokens = Counter(_tokenize(query))
    c_tokens = Counter(_tokenize(chunk))
    return sum((q_tokens & c_tokens).values())


def _draft_answer(query: str, contexts: list[str]) -> str:
    if not contexts:
        return ""
    ranked = sorted(contexts, key=lambda c: _score_overlap(query, c), reverse=True)
    return ranked[0][:260].strip()


class AgenticRAGAgent:
    name = "agentic_rag"

    def __init__(self, top_k: int = 3, max_steps: int = 3):
        self.top_k = top_k
        self.max_steps = max_steps

    def predict(self, sample: QASample) -> Prediction:
        start = time.perf_counter()

        chunks = _split_chunks(sample.context)
        working_query = sample.question
        retrieved_accum: list[str] = []
        step_log: list[dict] = []

        for step in range(self.max_steps):
            ranked = sorted(chunks, key=lambda c: _score_overlap(working_query, c), reverse=True)
            top = ranked[: self.top_k]
            for item in top:
                if item not in retrieved_accum:
                    retrieved_accum.append(item)
            answer_draft = _draft_answer(working_query, top)
            working_query = f"{sample.question} {answer_draft[:80]}".strip()
            step_log.append({
                "step": step + 1,
                "retrieved": len(top),
                "draft_len": len(answer_draft),
            })

        answer = _draft_answer(sample.question, retrieved_accum)
        latency_ms = (time.perf_counter() - start) * 1000

        return Prediction(
            id=sample.id,
            architecture=self.name,
            question=sample.question,
            predicted_answer=answer,
            gold_answers=sample.answers,
            retrieved_contexts=retrieved_accum[: self.top_k],
            latency_ms=latency_ms,
            trace={"steps": ["plan", "retrieve", "refine", "answer"], "iterations": step_log},
        )
