import typing
import pandas as pd
from lm_survey.survey.question import Question


class Variable:
    def __init__(self, name: str, questions: typing.Dict[str, dict]) -> None:
        self.name = name
        self.questions = {
            key: Question(key=key, **value) for key, value in questions.items()
        }

    def is_valid(self, row: pd.Series) -> bool:
        return any(question.is_valid(row) for question in self.questions.values())

    def _get_key(self, row: pd.Series) -> str:
        for key, question in self.questions.items():
            if question.is_valid(row):
                return key

        raise ValueError(
            f"This row has no key containing a valid value for the variable: {self.name}."
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

    def to_question(self, row: pd.Series) -> str:
        key = self._get_key(row)
        return self.questions[key].text

    def get_correct_letter(self, row: pd.Series) -> str:
        key = self._get_key(row)

        return self.questions[key].get_correct_letter(row)


if __name__ == "__main__":
    # TODO probably make this a unit test later.
    variable = Variable(
        name="color",
        questions={
            "q1": {
                "text": "What is your favorite color?",
                "valid_options": {
                    "0": {"text": "red", "phrase": "I like the color red."},
                    "1": {"text": "blue", "phrase": "I like the color blue."},
                },
                "invalid_options": ["2", "3", "4"],
            },
            "q2": {
                "text": "What is your favorite color?",
                "valid_options": {
                    "2": {"text": "green", "phrase": "I like the color green."},
                },
                "invalid_options": ["3", "4"],
            },
        },
    )

    valid_rows = [pd.Series({"q1": "0", "q2": "0"}), pd.Series({"q1": "2", "q2": "2"})]
    invalid_row = pd.Series({"q1": "3", "q2": "3"})

    print(variable.to_prompt(valid_rows[0]))
    print(variable.to_prompt(valid_rows[1]))

    print(variable.to_raw(valid_rows[0]))
    print(variable.to_raw(valid_rows[1]))

    print(variable.to_text(valid_rows[0]))
    print(variable.to_text(valid_rows[1]))

    print(variable.to_phrase(valid_rows[0]))
    print(variable.to_phrase(valid_rows[1]))

    print(variable.get_correct_letter(valid_rows[0]))
    print(variable.get_correct_letter(valid_rows[1]))

    try:
        variable.to_prompt(invalid_row)
    except ValueError as error:
        print(error)

    try:
        variable.to_raw(invalid_row)
    except ValueError as error:
        print(error)

    try:
        variable.get_correct_letter(invalid_row)
    except ValueError as error:
        print(error)
