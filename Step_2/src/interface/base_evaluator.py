from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    question: str
    response: str
    expected_answer: str
    is_correct: bool
    reasoning: str


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(
        self, query: str, response: str, expected_answer: str
    ) -> EvaluationResult:
        pass
