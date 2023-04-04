import typing


class DependentVariableSample:
    def __init__(
        self,
        question: str,
        independent_variables: typing.Dict[str, str],
        df_index: int,
        key: str,
        prompt: str,
        correct_letter: str,
        completion: typing.Optional[str] = None,
        **kwargs,
    ):
        self.question = question
        self.independent_variables = independent_variables
        self.df_index = df_index
        self.key = key
        self.prompt = prompt
        self.correct_letter = correct_letter
        self.completion = completion

    def _extract_letter(self, completion: str) -> str:
        return completion.strip().upper()[:1]

    @property
    def is_completion_correct(self) -> bool:
        if self.completion is None:
            raise ValueError("Completion is None.")
        else:
            return self.correct_letter == self._extract_letter(self.completion)

    def __str__(self) -> str:
        sep = "\n\n"
        prompt = (
            self.prompt + self.completion
            if self.completion is not None
            else self.prompt
        )
        return sep.join(
            [
                f"Question Key: {self.key}",
                f"Row Index: {self.df_index}",
                prompt,
                f"Correct Answer: {self.correct_letter}",
            ]
        )

    def __repr__(self) -> str:
        return self.__str__()

    def to_dict(self) -> dict:
        self_dict = self.__dict__
        self_dict.update({"is_completion_correct": self.is_completion_correct})

        return self_dict
