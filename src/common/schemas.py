from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class QASample:
    id: str
    question: str
    context: str
    answers: list[str]
    is_impossible: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Prediction:
    id: str
    architecture: str
    question: str
    predicted_answer: str
    gold_answers: list[str]
    retrieved_contexts: list[str]
    latency_ms: float
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalRecord:
    id: str
    architecture: str
    em: int
    f1: float
    recall_at_k: float
    mrr: float
    judge_correctness: int
    judge_completeness: int
    judge_reasoning: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
