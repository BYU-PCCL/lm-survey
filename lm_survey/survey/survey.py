import typing
import pandas as pd

from lm_survey.survey import (
    IndependentVariable,
    DependentVariable,
    DependentVariableSample,
)
from lm_survey.prompt_templates import INDEPENDENT_VARIABLE_SUMMARY_TEMPLATE
import json
import functools
import argparse


class Survey:
    def __init__(
        self,
        name: str,
        data_filename: str,
        independent_variables_filename: str,
        dependent_variables_filename: str,
    ):
        self.name = name
        self.df = pd.read_csv(data_filename)

        with open(independent_variables_filename, "r") as file:
            self.independent_variables = [
                IndependentVariable(**independent_variable)
                for independent_variable in json.load(file)
            ]

        with open(dependent_variables_filename, "r") as file:
            self.dependent_variables = {
                dependent_variable["key"]: DependentVariable(**dependent_variable)
                for dependent_variable in json.load(file)
            }

    def _handle_missing_independent_variable(func: typing.Callable) -> typing.Callable:  # type: ignore
        @functools.wraps(func)
        def wrapper(self, row: pd.Series) -> str:
            try:
                return func(self, row)
            except ValueError as error:
                raise ValueError(
                    f"Row does not contain all fields for the independent variable summary. {error}"
                )

        return wrapper

    @_handle_missing_independent_variable
    def _create_independent_variable_summary(self, row: pd.Series) -> str:
        return " ".join(
            [
                independent_variable.to_sentence(row)
                for independent_variable in self.independent_variables
            ]
        )

    @_handle_missing_independent_variable
    def _get_independent_variable_dict(self, row: pd.Series) -> dict:
        return {
            independent_variable.name: independent_variable.to_option(row)
            for independent_variable in self.independent_variables
        }

    def _templatize(
        self, independent_variable_summary: str, dependent_variable: DependentVariable
    ) -> str:
        return INDEPENDENT_VARIABLE_SUMMARY_TEMPLATE.format(
            context_summary=independent_variable_summary,
            dependent_variable_prompt=dependent_variable.templatize(),
        )

    def iterate(
        self, n_samples_per_dependent_variable: typing.Optional[int] = None
    ) -> typing.Iterator[DependentVariableSample]:
        if n_samples_per_dependent_variable is None:
            n_samples_per_dependent_variable = len(self.df)

        n_sampled_per_dependent_variable = {
            key: 0 for key in self.dependent_variables.keys()
        }

        # The index from iterrows gives type errors when using it as a key in iloc.
        for i, (_, row) in enumerate(self.df.iterrows()):
            try:
                independent_variable_summary = (
                    self._create_independent_variable_summary(row)
                )
            except ValueError:
                continue

            for key, dependent_variable in self.dependent_variables.items():
                if n_sampled_per_dependent_variable[
                    key
                ] >= n_samples_per_dependent_variable or not dependent_variable.is_valid(
                    row
                ):
                    continue

                prompt = self._templatize(
                    independent_variable_summary, dependent_variable
                )
                correct_letter = dependent_variable.get_correct_letter(row)
                independent_variables = self._get_independent_variable_dict(row)

                yield DependentVariableSample(
                    question=dependent_variable.question,
                    independent_variables=independent_variables,
                    df_index=i,
                    key=key,
                    prompt=prompt,
                    correct_letter=correct_letter,
                )

                n_sampled_per_dependent_variable[key] += 1

    def __iter__(
        self,
    ) -> typing.Iterator[DependentVariableSample]:
        return self.iterate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_filename",
        type=str,
        default="data/roper/data.csv",
        help="The filename of the data.",
    )
    parser.add_argument(
        "--independent_variables_filename",
        type=str,
        default="data/roper/demographics.json",
        help="The filename of the independent variables.",
    )
    parser.add_argument(
        "--dependent_variables_filename",
        type=str,
        default="data/roper/questions.json",
        help="The filename of the dependent variables.",
    )
    args = parser.parse_args()

    survey = Survey(
        name="roper",
        data_filename=args.data_filename,
        independent_variables_filename=args.independent_variables_filename,
        dependent_variables_filename=args.dependent_variables_filename,
    )

    prompt_info = next(iter(survey))
    prompt_info.completion = " C)"

    print(prompt_info)