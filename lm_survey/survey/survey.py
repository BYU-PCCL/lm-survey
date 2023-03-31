import typing
import pandas as pd

from lm_survey.survey import (
    DependentVariableSample,
    Variable,
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
        config_filename: str,
        independent_variable_names: typing.List[str] = [],
        dependent_variable_names: typing.List[str] = [],
    ):
        self.name = name
        self.df = pd.read_csv(data_filename)

        self.variables = self._load_variables(config_filename=config_filename)

        self.set_independent_variables(
            independent_variable_names=independent_variable_names
        )

        self.set_dependent_variables(dependent_variable_names=dependent_variable_names)

    def _load_variables(self, config_filename: str) -> typing.List[Variable]:
        with open(config_filename, "r") as file:
            return [Variable(**variable) for variable in json.load(file)]

    def _filter_variables(
        self, variable_names: typing.List[str]
    ) -> typing.List[Variable]:
        variable_name_set = set(variable_names)

        return [
            variable
            for variable in self.variables
            if variable.name in variable_name_set
        ]

    def set_independent_variables(self, independent_variable_names: typing.List[str]):
        acceptable_names = set(independent_variable_names)

        self._independent_variables = [
            variable for variable in self.variables if variable.name in acceptable_names
        ]

    def set_dependent_variables(self, dependent_variable_names: typing.List[str]):
        acceptable_names = set(dependent_variable_names)

        self._dependent_variables = {
            variable.name: variable
            for variable in self.variables
            if variable.name in acceptable_names
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
            [variable.to_phrase(row) for variable in self._independent_variables]
        )

    @_handle_missing_independent_variable
    def _get_independent_variable_dict(self, row: pd.Series) -> typing.Dict[str, str]:
        return {
            variable.name: variable.to_text(row)
            for variable in self._independent_variables
        }

    def _templatize(
        self,
        independent_variable_summary: str,
        dependent_variable_prompt: str,
    ) -> str:
        return INDEPENDENT_VARIABLE_SUMMARY_TEMPLATE.format(
            context_summary=independent_variable_summary,
            dependent_variable_prompt=dependent_variable_prompt,
        )

    def iterate(
        self, n_samples_per_dependent_variable: typing.Optional[int] = None
    ) -> typing.Iterator[DependentVariableSample]:
        if n_samples_per_dependent_variable is None:
            n_samples_per_dependent_variable = len(self.df)

        n_sampled_per_dependent_variable = {
            key: 0 for key in self._dependent_variables.keys()
        }

        # The index from iterrows gives type errors when using it as a key in iloc.
        for i, (_, row) in enumerate(self.df.iterrows()):
            try:
                independent_variable_summary = (
                    self._create_independent_variable_summary(row)
                )
            except ValueError:
                continue

            for key, dependent_variable in self._dependent_variables.items():
                if n_sampled_per_dependent_variable[
                    key
                ] >= n_samples_per_dependent_variable or not dependent_variable.is_valid(
                    row
                ):
                    continue

                dependent_variable_prompt = dependent_variable.to_prompt(row)

                prompt = self._templatize(
                    independent_variable_summary=independent_variable_summary,
                    dependent_variable_prompt=dependent_variable_prompt,
                )
                correct_letter = dependent_variable.get_correct_letter(row)
                independent_variables = self._get_independent_variable_dict(row)

                yield DependentVariableSample(
                    question=dependent_variable.to_question(row),
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

    # survey = Survey(
    #     name="roper",
    #     data_filename=args.data_filename,
    #     independent_variables_filename=args.independent_variables_filename,
    #     dependent_variables_filename=args.dependent_variables_filename,
    # )

    # prompt_info = next(iter(survey))
    # prompt_info.completion = " C)"

    # print(prompt_info)
