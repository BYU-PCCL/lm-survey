import functools
import typing

import numpy as np
from lm_survey.survey.variable import Variable


class Completion:
    def __init__(
        self,
        possible_completions: typing.List[str],
        correct_completion: str,
        completion_log_probs: typing.Optional[typing.Dict[str, float]] = None,
        **kwargs,
    ):
        self.possible_completions = possible_completions
        self.correct_completion = correct_completion

        if completion_log_probs is not None:
            self.set_completion_log_probs(completion_log_probs)
        else:
            self._completion_log_probs = None

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        self_dict = self.__dict__.copy()

        if self.are_completion_log_probs_set():
            self_dict["top_completion"] = self.top_completion
            self_dict["is_completion_correct"] = self.is_completion_correct
        else:
            self_dict["top_completion"] = None
            self_dict["is_completion_correct"] = None

        return self_dict

    def set_completion_log_probs(
        self, completion_log_probs: typing.Dict[str, float]
    ) -> None:
        self._completion_log_probs = dict(
            sorted(
                completion_log_probs.items(),
                key=lambda completion: completion[1],
                reverse=True,
            )
        )

    def are_completion_log_probs_set(self) -> bool:
        return self._completion_log_probs is not None

    def _extract_letter(self, completion: str) -> str:
        return completion.strip().upper()[:1]

    @property
    def top_completion(self) -> str:
        if not self.are_completion_log_probs_set():
            raise ValueError("No completion log probs have been set.")

        return max(
            self.possible_completions,
            key=lambda completion: self._completion_log_probs[completion],  # type: ignore
        )

    def get_correct_completion_ranking(self) -> int:
        if not self.are_completion_log_probs_set():
            raise ValueError("No completion log probs have been set.")

        ranked_completions_dict = {
            self._extract_letter(completion): rank
            for rank, completion in enumerate(self._completion_log_probs.keys())  # type: ignore
        }

        return ranked_completions_dict[self._extract_letter(self.correct_completion)]

    @property
    def is_completion_correct(self) -> bool:
        return self._extract_letter(self.top_completion) == self._extract_letter(
            self.correct_completion
        )


class DependentVariableSample:
    def __init__(
        self,
        index: int,
        independent_variables: typing.Dict[str, str],
        variable_name: str,
        question: str,
        prompt: str,
        completion: typing.Union[Completion, typing.Dict[str, typing.Any]],
        **kwargs,
    ) -> None:
        self.index = index
        self.independent_variables = independent_variables
        self.variable_name = variable_name
        self.question = question
        self.prompt = prompt

        if isinstance(completion, Completion):
            self.completion = completion
        else:
            self.completion = Completion(**completion)

    def __str__(self) -> str:
        sep = "\n\n"
        prompt = (
            self.prompt + self.completion.top_completion
            if self.completion.are_completion_log_probs_set()
            else self.prompt
        )
        return sep.join(
            [
                f"Variable Name: {self.variable_name}",
                prompt,
                f"Correct Answer: {self.completion.correct_completion}",
            ]
        )

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict:
        self_dict = self.__dict__.copy()

        self_dict["completion"] = self.completion.to_dict()

        return self_dict
