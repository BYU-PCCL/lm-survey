import typing
import pandas as pd
import json
import functools
from lm_survey.constants import MULTIPLE_CHOICE_LIST


class Question:
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


class Demographic:
    def __init__(
        self,
        name: str,
        keys: typing.List[str],
        invalid_options: typing.Dict[str, typing.List[str]],
        valid_options: typing.Dict[str, typing.Dict[str, str]],
    ):
        self.name = name
        self.keys = keys
        self.invalid_options = {
            key: set(options) for key, options in invalid_options.items()
        }
        self.valid_options = valid_options

    def _templatize(self, row: pd.Series) -> str:
        key = self._get_key(row)

        return self.valid_options[key][row[key]]

    def _get_key(self, row: pd.Series) -> str:
        for key in self.keys:
            if row[key].strip() not in self.invalid_options[key]:
                return key

        raise ValueError(
            f"This row has no key containing a valid value for the demographic: {self.name})."
        )

    def get_sentence(self, row: pd.Series) -> str:
        return self._templatize(row)

    def get_option(self, row: pd.Series) -> str:
        key = self._get_key(row)
        return row[key]


class QuestionSample:
    def __init__(
        self,
        question: str,
        demographics: typing.Dict[str, str],
        df_index: int,
        key: str,
        prompt: str,
        correct_letter: str,
        completion: typing.Optional[str] = None,
    ):
        self.question = question
        self.demographics = demographics
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

    def to_dict(self) -> dict:
        self_dict = self.__dict__
        self_dict.update({"is_completion_correct": self.is_completion_correct})

        return self_dict


class Survey:
    def __init__(
        self,
        name: str,
        data_filename: str,
        demographic_filename: str,
        question_filename: str,
    ):
        self.name = name
        self.df = pd.read_csv(data_filename)

        with open(demographic_filename, "r") as file:
            self.demographics = [
                Demographic(**demographic) for demographic in json.load(file)
            ]

        with open(question_filename, "r") as file:
            self.questions = {
                question["key"]: Question(**question) for question in json.load(file)
            }

    def _handle_missing_demographic(func: typing.Callable) -> typing.Callable:  # type: ignore
        @functools.wraps(func)
        def wrapper(self, row: pd.Series) -> str:
            try:
                return func(self, row)
            except ValueError as error:
                raise ValueError(
                    f"Row does not contain all fields for backstory. {error}"
                )

        return wrapper

    @_handle_missing_demographic
    def _create_backstory(self, row: pd.Series) -> str:
        return " ".join(
            [demographic.get_sentence(row) for demographic in self.demographics]
        )

    @_handle_missing_demographic
    def _get_demographic_dict(self, row: pd.Series) -> dict:
        return {
            demographic.name: demographic.get_option(row)
            for demographic in self.demographics
        }

    def _templatize(self, backstory: str, question: Question) -> str:
        return "\n\n".join(
            [
                f"Self-Identification: {backstory}",
                question.templatize(),
            ]
        )

    def iterate(
        self, n_samples_per_question: typing.Optional[int] = None
    ) -> typing.Iterator[QuestionSample]:
        if n_samples_per_question is None:
            n_samples_per_question = len(self.df)

        n_sampled_per_question = {key: 0 for key in self.questions.keys()}

        # The index from iterrows gives type errors when using it as a key in iloc.
        for i, (_, row) in enumerate(self.df.iterrows()):
            try:
                backstory = self._create_backstory(row)
            except ValueError:
                continue

            for key, question in self.questions.items():
                if n_sampled_per_question[
                    key
                ] >= n_samples_per_question or not question.is_valid(row):
                    continue

                prompt = self._templatize(backstory, question)
                correct_letter = question.get_correct_letter(row)
                demographics = self._get_demographic_dict(row)

                yield QuestionSample(
                    question=question.question,
                    demographics=demographics,
                    df_index=i,
                    key=key,
                    prompt=prompt,
                    correct_letter=correct_letter,
                )

                n_sampled_per_question[key] += 1

    def __iter__(
        self,
    ) -> typing.Iterator[QuestionSample]:
        return self.iterate()


if __name__ == "__main__":
    survey = Survey(
        name="roper",
        data_filename="data/roper/data.csv",
        demographic_filename="data/roper/demographics.json",
        question_filename="data/roper/questions.json",
    )

    prompt_info = next(iter(survey))
    prompt_info.completion = " C)"

    print(prompt_info)
