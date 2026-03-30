import os
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from src.eval.metrics import f1_score, normalize_answer


@dataclass
class JudgeScore:
    correctness: int
    completeness: int
    reasoning: int

    def to_dict(self) -> dict[str, int]:
        return {
            "judge_correctness": self.correctness,
            "judge_completeness": self.completeness,
            "judge_reasoning": self.reasoning,
        }


class Judge:
    def __init__(self, mode: str = "local"):
        self.mode = mode
        self.client = None
        if mode in {"api", "hybrid"} and OpenAI is not None and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI()

    def score(self, question: str, predicted: str, gold_answers: list[str]) -> JudgeScore:
        local = self._local_score(predicted, gold_answers)
        if self.mode == "local":
            return local
        if self.mode == "api":
            return self._api_score(question, predicted, gold_answers) or local

        low_confidence = local.correctness <= 2 or local.completeness <= 2
        if low_confidence:
            api_score = self._api_score(question, predicted, gold_answers)
            if api_score is not None:
                return api_score
        return local

    def _local_score(self, predicted: str, gold_answers: list[str]) -> JudgeScore:
        truth = gold_answers[0] if gold_answers else ""
        f1 = f1_score(predicted, truth)
        correct = 1 if normalize_answer(predicted) == normalize_answer(truth) else 0

        if correct:
            return JudgeScore(correctness=5, completeness=5, reasoning=4)
        if f1 >= 0.8:
            return JudgeScore(correctness=4, completeness=4, reasoning=4)
        if f1 >= 0.5:
            return JudgeScore(correctness=3, completeness=3, reasoning=3)
        if f1 > 0.0:
            return JudgeScore(correctness=2, completeness=2, reasoning=2)
        return JudgeScore(correctness=1, completeness=1, reasoning=1)

    def _api_score(self, question: str, predicted: str, gold_answers: list[str]) -> JudgeScore | None:
        if self.client is None:
            return None

        prompt = (
            "Score answer quality from 1-5 for correctness, completeness, reasoning. "
            "Return strict JSON with keys correctness, completeness, reasoning.\n"
            f"Question: {question}\n"
            f"Predicted: {predicted}\n"
            f"Gold: {gold_answers}\n"
        )
        try:
            response = self.client.responses.create(
                model="gpt-4.1-mini",
                input=prompt,
                temperature=0,
            )
            text = response.output_text.strip()
            parts = text.replace("{", "").replace("}", "").replace('"', "").split(",")
            values: dict[str, int] = {}
            for part in parts:
                if ":" not in part:
                    continue
                key, value = part.split(":", 1)
                key = key.strip()
                value_int = int("".join(ch for ch in value if ch.isdigit()) or "0")
                values[key] = max(1, min(5, value_int))
            if {"correctness", "completeness", "reasoning"}.issubset(values):
                return JudgeScore(
                    correctness=values["correctness"],
                    completeness=values["completeness"],
                    reasoning=values["reasoning"],
                )
            return None
        except Exception:
            return None
