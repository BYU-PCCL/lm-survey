import typing

import pandas as pd
from lm_survey.survey.question import Question


class Variable:
    def __init__(self, name: str, questions: typing.Dict[str, dict]) -> None:
        self.name = name
        self.questions = {
            key: Question(key=key, **value) for key, value in questions.items()
        }

    def _get_key(self, row: pd.Series) -> str:
        for key in self.questions.keys():
            if row[key] not in self.questions[key].invalid_options:
                return key

        raise ValueError(
            f"This row has no key containing a valid value for the variable: {self.name})."
        )

    def to_prompt(self, row: pd.Series) -> str:
        key = self._get_key(row)
        return self.questions[key].to_prompt()

    def _to_value(self, value_key: str, row: pd.Series) -> str:
        key = self._get_key(row)

        value = getattr(self.questions[key].valid_options[row[key]], value_key)

        if value is None:
            raise ValueError(
                f"This row's response has not been converted to {value_key} for variable: {self.name}, question: {key}, response: {row[key]}."
            )

        return value

    def to_raw(self, row: pd.Series) -> str:
        return self._to_value(value_key="raw", row=row)

    def to_text(self, row: pd.Series) -> str:
        return self._to_value(value_key="text", row=row)

    def to_phrase(self, row: pd.Series) -> str:
        return self._to_value(value_key="phrase", row=row)

    def get_correct_letter(self, row: pd.Series) -> str:
        key = self._get_key(row)

        return self.questions[key].get_correct_letter(row)
