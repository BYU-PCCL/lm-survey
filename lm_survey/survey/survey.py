import typing
import pandas as pd

from . import IndependentVariable, DependentVariable, DependentVariableSample
import json
import functools


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
                    f"Row does not contain all fields for backstory. {error}"
                )

        return wrapper

    @_handle_missing_independent_variable
    def _create_backstory(self, row: pd.Series) -> str:
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

    def _templatize(self, backstory: str, dependent_variable: DependentVariable) -> str:
        return "\n\n".join(
            [
                # TODO(alexgshaw): Make this more generalizable.
                f"Self-Identification: {backstory}",
                dependent_variable.templatize(),
            ]
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
                backstory = self._create_backstory(row)
            except ValueError:
                continue

            for key, dependent_variable in self.dependent_variables.items():
                if n_sampled_per_dependent_variable[
                    key
                ] >= n_samples_per_dependent_variable or not dependent_variable.is_valid(
                    row
                ):
                    continue

                prompt = self._templatize(backstory, dependent_variable)
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
    survey = Survey(
        name="roper",
        data_filename="data/roper/data.csv",
        independent_variables_filename="data/roper/demographics.json",
        dependent_variables_filename="data/roper/questions.json",
    )

    prompt_info = next(iter(survey))
    prompt_info.completion = " C)"

    print(prompt_info)
