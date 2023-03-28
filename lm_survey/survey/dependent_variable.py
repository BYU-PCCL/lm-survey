import typing
import pandas as pd
from lm_survey.constants import MULTIPLE_CHOICE_LIST


class DependentVariable:
    def __init__(
        self,
        key: str,
        question: str,
        valid_options: typing.List[str],
        invalid_options: typing.List[str],
    ):
        self.key = key
        self.question = question
        self.valid_options = valid_options
        self.valid_options_index_map = {
            option: i for i, option in enumerate(valid_options)
        }
        self.invalid_options = set(invalid_options)

    def templatize(self) -> str:
        sep = "\n\n"
        return sep.join(
            [
                f"Question: {self.question}",
                "\n".join(
                    [
                        f"{MULTIPLE_CHOICE_LIST[i]}) {choice}"
                        for i, choice in enumerate(self.valid_options)
                    ]
                ),
                f"Please respond with the letter of the answer choice that best fits the self-identified person's response.",
                f"Answer:",
            ]
        )

    def is_valid(self, row: pd.Series) -> bool:
        return row[self.key].strip() not in self.invalid_options

    def get_correct_letter(self, row: pd.Series) -> str:
        return MULTIPLE_CHOICE_LIST[self.valid_options_index_map[row[self.key].strip()]]

    def __str__(self):
        return self.question


if __name__ == "__main__":
    dependent_variable = DependentVariable(
        key="summer",
        question="What is your favorite month of the year?",
        valid_options=["January", "March", "June", "December"],
        invalid_options=["N/A"],
    )

    print(dependent_variable.templatize())
