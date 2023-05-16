import typing
import pandas as pd
from lm_survey.survey.question import Question


class Variable:
    def __init__(self, name: str, questions: typing.List[dict] = []) -> None:
        self.name = name
        self.questions = {
            question["key"]: Question(**question) for question in questions
        }

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            "name": self.name,
            "questions": [question.to_dict() for question in self.questions.values()],
        }

    def upsert_question(self, question: Question) -> None:
        self.questions[question.key] = question

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
        try:
            key = self._get_key(row)

            value = getattr(self.questions[key].valid_options[row[key].lower()], value_key)
        except:
            raise

        if value is None:
            raise ValueError(
                f"This row's response has not been converted to {value_key} for variable: {self.name}, question: {key}, response: {row[key]}."
            )

        return value

    def to_raw(self, row: pd.Series) -> str:
        return self._to_value(value_key="raw", row=row)

    def to_text(self, row: pd.Series) -> str:
        return self._to_value(value_key="text", row=row)

    def to_natural_language(self, row: pd.Series) -> str:
        return self._to_value(value_key="natural_language", row=row)

    def to_question(self, row: pd.Series) -> str:
        key = self._get_key(row)
        return self.questions[key].text

    def get_correct_letter(self, row: pd.Series) -> str:
        key = self._get_key(row)
        return self.questions[key].get_correct_letter(row)

    def get_possible_letters(self, row: pd.Series) -> typing.List[str]:
        key = self._get_key(row)
        return self.questions[key].get_possible_letters()


if __name__ == "__main__":
    # TODO probably make this a unit test later.
    variable = Variable(
        name="color",
        questions=[
            {
                "key": "q1",
                "text": "What is your favorite color?",
                "valid_options": {
                    "0": {"text": "red", "natural_language": "I like the color red."},
                    "1": {"text": "blue", "natural_language": "I like the color blue."},
                },
                "invalid_options": ["2", "3", "4"],
            },
            {
                "key": "q2",
                "text": "What is your favorite color?",
                "valid_options": {
                    "2": {
                        "text": "green",
                        "natural_language": "I like the color green.",
                    },
                },
                "invalid_options": ["3", "4"],
            },
        ],
    )

    valid_rows = [pd.Series({"q1": "0", "q2": "0"}), pd.Series({"q1": "2", "q2": "2"})]
    invalid_row = pd.Series({"q1": "3", "q2": "3"})

    print(variable.to_prompt(valid_rows[0]))
    print(variable.to_prompt(valid_rows[1]))

    print(variable.to_raw(valid_rows[0]))
    print(variable.to_raw(valid_rows[1]))

    print(variable.to_text(valid_rows[0]))
    print(variable.to_text(valid_rows[1]))

    print(variable.to_natural_language(valid_rows[0]))
    print(variable.to_natural_language(valid_rows[1]))

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
