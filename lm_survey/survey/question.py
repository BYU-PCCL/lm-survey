import typing
import pandas as pd
from lm_survey.constants import MULTIPLE_CHOICE_LIST
from lm_survey.prompt_templates import (
    DEPENDENT_VARIABLE_TEMPLATE,
    format_multiple_choice_options,
)

# Make it immutable so people don't accidentally change it during inference.
class ValidOption(typing.NamedTuple):
    raw: str
    text: typing.Optional[str] = None
    phrase: typing.Optional[str] = None


class Question:
    def __init__(
        self,
        key: str,
        text: str,
        valid_options: typing.Dict[str, typing.Dict[str, str]],
        invalid_options: typing.List[str],
    ) -> None:
        self.key = key
        self.text = text
        self.invalid_options = set(invalid_options)

        self.valid_options = {
            key: ValidOption(raw=key, **value) for key, value in valid_options.items()
        }

        self.valid_options_index_map = {
            option: i for i, option in enumerate(valid_options.keys())
        }

    def to_prompt(self) -> str:
        return DEPENDENT_VARIABLE_TEMPLATE.format(
            question=self.text,
            choices=format_multiple_choice_options(
                [value.text for value in self.valid_options.values()]
            ),
        )

    def is_valid(self, row: pd.Series) -> bool:
        return row[self.key] not in self.invalid_options

    def get_correct_letter(self, row: pd.Series) -> str:
        try:
            return MULTIPLE_CHOICE_LIST[self.valid_options_index_map[row[self.key]]]
        except KeyError:
            raise ValueError(
                f"This row's response is not a valid option: {row[self.key]}"
            )

    def __str__(self):
        return self.text


if __name__ == "__main__":
    # TODO make this a unit test later.
    question = Question(
        key="q1",
        text="What is your favorite color?",
        valid_options={
            "0": {"text": "red", "phrase": "I like the color red."},
            "1": {"text": "blue", "phrase": "I like the color blue."},
            "2": {"text": "green", "phrase": "I like the color green."},
        },
        invalid_options=["3", "4"],
    )

    print(question.to_prompt())

    print(question.is_valid(pd.Series({"q1": "0"})))
    print(question.is_valid(pd.Series({"q1": "3"})))

    print(question.get_correct_letter(pd.Series({"q1": "0"})))
    print(question.get_correct_letter(pd.Series({"q1": "1"})))
    print(question.get_correct_letter(pd.Series({"q1": "2"})))

    try:
        question.get_correct_letter(pd.Series({"q1": "3"}))
    except ValueError as error:
        print(error)
