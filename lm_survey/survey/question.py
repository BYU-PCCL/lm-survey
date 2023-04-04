import typing
import pandas as pd
from lm_survey.constants import MULTIPLE_CHOICE_LIST
from lm_survey.prompt_templates import (
    DEPENDENT_VARIABLE_TEMPLATE,
    format_multiple_choice_options,
)


class ValidOption:
    def __init__(
        self,
        raw: str,
        text: typing.Optional[str] = None,
        natural_language: typing.Optional[str] = None,
    ) -> None:
        self.raw = raw
        self.text = text
        self.natural_language = natural_language

    def to_dict(self) -> typing.Dict[str, typing.Optional[str]]:
        return self.__dict__

    def __str__(self) -> str:
        return "\n".join(
            [
                f"\tRaw: {self.raw}",
                f"\tText: {self.text}",
                f"\tNatural Language: {self.natural_language}",
            ]
        )

    def __repr__(self) -> str:
        return self.__str__()


class Question:
    def __init__(
        self,
        key: str,
        text: str,
        valid_options: typing.List[typing.Dict[str, typing.Any]],
        invalid_options: typing.List[str],
    ) -> None:
        self.key = key
        self.text = text
        self.invalid_options = set(invalid_options)

        self.valid_options = {
            option["raw"]: ValidOption(**option) for option in valid_options
        }

        self.valid_options_index_map = {
            option: i for i, option in enumerate(self.valid_options.keys())
        }

    def to_dict(self) -> typing.Dict[str, typing.Any]:
        return {
            "key": self.key,
            "text": self.text,
            "valid_options": [
                option.to_dict() for option in self.valid_options.values()
            ],
            "invalid_options": list(self.invalid_options),
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
        valid_options=[
            {"raw": "0", "text": "red", "natural_language": "I like the color red."},
            {"raw": "1", "text": "blue", "natural_language": "I like the color blue."},
            {
                "raw": "2",
                "text": "green",
                "natural_language": "I like the color green.",
            },
        ],
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
