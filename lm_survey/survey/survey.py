import os
import typing
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from lm_survey.survey.dependent_variable_sample import (
    Completion,
    DependentVariableSample,
)
from lm_survey.survey.question import Question, ValidOption
from lm_survey.survey.variable import Variable
from lm_survey.prompt_templates import INDEPENDENT_VARIABLE_SUMMARY_TEMPLATE
import json
import functools
import argparse


class Survey:
    def __init__(
        self,
        name: str,
        data_filename: str,
        config_filename: typing.Optional[str] = None,
        independent_variable_names: typing.List[str] = [],
        dependent_variable_names: typing.List[str] = [],
    ):
        self.name = name
        self.df = pd.read_csv(data_filename, dtype=str)

        if config_filename is not None and os.path.exists(config_filename):
            self.variables = self._load_variables(config_filename=config_filename)
        else:
            self.variables = []
            print(
                "No config file provided. You will need to generate one using the `generate_config` method."
            )

        self.set_independent_variables(
            independent_variable_names=independent_variable_names
        )

        self.set_dependent_variables(dependent_variable_names=dependent_variable_names)

    def _load_variables(self, config_filename: str) -> typing.List[Variable]:
        with open(config_filename, "r") as file:
            return [Variable(**variable) for variable in json.load(file)]

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
            [
                variable.to_natural_language(row)
                for variable in self._independent_variables
            ]
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

    def _get_question_text(self, key: str) -> str:
        return input(
            f"\nFor question {key}, what is the text for this"
            " question in the codebook?\n:"
        )

    def _process_option_index(self, index: str) -> typing.List[int]:
        if not "-" in index:
            return [int(index)]

        minimum, maximum = index.split("-")

        return list(range(int(minimum), int(maximum) + 1))

    def _process_option_indices(self, indices: str) -> typing.Set[int]:
        return {
            processed_index
            for index in indices.split(",")
            for processed_index in self._process_option_index(index)
        }

    def _process_valid_option(
        self, raw: str, natural_language_template: str, use_default: bool
    ) -> typing.Tuple[bool, ValidOption]:
        if use_default:
            return use_default, ValidOption(raw=raw, text=raw, natural_language=raw)

        text = input(
            f"\nText from codebook corresponding to option '{raw}' (hit ENTER to use the default value): "
        )

        if text == "skip":
            return True, ValidOption(raw=raw, text=raw, natural_language=raw)
        if text == "":
            text = raw

        rephrasing = input(
            f"Rephrasing for option '{raw}' to appear in the"
            f" {'{X}'} slot of your template, or to appear, if you"
            " didn't write a template (hit ENTER to use the text value): "
        )

        if rephrasing == "skip":
            return True, ValidOption(raw=raw, text=raw, natural_language=raw)
        elif rephrasing == "":
            natural_language = natural_language_template.format(X=text)
        else:
            natural_language = natural_language_template.format(X=rephrasing)

        return use_default, ValidOption(
            raw=raw, text=text, natural_language=natural_language
        )

    def _process_valid_option_exceptions(self, valid_options: typing.List[ValidOption]):
        while True:
            print(
                "\nHere is what you have so far:\n",
                "\n\n".join(
                    [
                        f"{i}.{valid_option}"
                        for i, valid_option in enumerate(valid_options)
                    ]
                ),
                sep="\n",
            )

            exception = input(
                "\nAre there any exceptions to the general format?\n(e.g., for"
                " the index, value pairs \n\n0 Republican\n1 Democrat\n2"
                " Independent\n\nyou would type 2 to provide an exception for"
                " Independent)\nType 'done' to finish\n:"
            )

            if exception == "done":
                break
            else:
                try:
                    option_index = int(exception)
                except ValueError:
                    print("Please enter a valid integer or 'done'")
                    continue

                text = input(
                    "What is the text corresponding to this code in the"
                    " codebook? Press ENTER to leave as is\n"
                )

                if text != "":
                    try:
                        valid_options[option_index].text = text
                    except IndexError:
                        print(
                            f"Index {option_index} is not a valid option. Please try again"
                        )
                        continue

                natural_language = input(
                    "How do you want to phrase this exception? Press ENTER to"
                    " leave as is\n"
                )

                if natural_language != "":
                    try:
                        valid_options[option_index].natural_language = natural_language
                    except IndexError:
                        print(
                            f"Index {option_index} is not a valid option. Please try again"
                        )
                        continue

        return valid_options

    def _get_natural_language_template(self) -> str:
        natural_language_template = input(
            "\nNow you will be asked whether you want to specify a general"
            " format for the data corresponding to this question to assume in"
            " your prompts and then whether there will be any exceptions to"
            ' that format. Do you want to specify a general format? (e.g. "I am'
            ' {X} years old." where {X} is a special token representing each'
            " specific answer)\nPress ENTER to skip\n:"
        )

        return natural_language_template if natural_language_template != "" else "{X}"

    def _process_options(
        self, valid_indices: typing.Set[int], unique_raw_options: np.ndarray
    ) -> typing.Tuple[typing.List[ValidOption], typing.List[str]]:
        valid_options = []
        invalid_options = []

        natural_language_template = self._get_natural_language_template()

        print(
            "Now you will be asked about the text in the codebook corresponding"
            " to each code, along with how to phrase each option to the"
            " language model.\nPress ENTER to skip this option, and type 'skip'"
            " to start specifying exceptions to the general rule\n:"
        )

        use_default = False

        for i, raw_option in enumerate(unique_raw_options):
            if i in valid_indices:
                use_default, valid_option = self._process_valid_option(
                    raw=raw_option,
                    natural_language_template=natural_language_template,
                    use_default=use_default,
                )
                valid_options.append(valid_option)
            else:
                invalid_options.append(raw_option)

        valid_options = self._process_valid_option_exceptions(
            valid_options=valid_options
        )

        return valid_options, invalid_options

    def _get_raw_sort_key(
        self, item: str
    ) -> typing.Tuple[int, typing.Union[str, float]]:
        try:
            num = float(item)
            return (0, num)
        except ValueError:
            return (1, item)

    def _get_options(
        self, key: str
    ) -> typing.Tuple[typing.List[ValidOption], typing.List[str]]:
        try:
            unique_raw_options = self.df[key].unique()
        except KeyError:
            raise ValueError(f"{key} is not a valid column name.")

        unique_raw_options = sorted(unique_raw_options, key=self._get_raw_sort_key)

        print(
            "\nFor that question, here is a list of the possible responses, each with its respective index.\n"
        )

        for i, raw_option in enumerate(unique_raw_options):
            print(f"{i}: {raw_option}")

        input_indices = input(
            "\nPlease give a comma-delimited list of indices for values you want"
            " to include (e.g., 1, 2, 5 ) or a range (e.g., 1-5 inclusive). The"
            " rest will be excluded\n:"
        )

        valid_indices = self._process_option_indices(input_indices)

        valid_options, invalid_options = self._process_options(
            valid_indices=valid_indices, unique_raw_options=unique_raw_options  # type: ignore
        )

        return valid_options, invalid_options

    def _create_question(
        self, column_names: typing.List[str]
    ) -> typing.Iterator[Question]:
        for column_name in column_names:
            question_text = self._get_question_text(key=column_name)

            try:
                valid_options, invalid_options = self._get_options(key=column_name)
            except ValueError as error:
                print(f"Skipping {column_name}: {error}.")
                continue

            yield Question(
                key=column_name,
                text=question_text,
                # TODO make it so that you can optionally input `ValidOption`s.
                valid_options=[
                    valid_option.to_dict() for valid_option in valid_options
                ],
                invalid_options=invalid_options,
            )

    def generate_config(self, config_filename: str):
        while True:
            restart = input("\nDo you want to start over? (y/n) :")
            if restart.lower() == "y":
                self.variables = []
                break
            elif restart.lower() == "n":
                break

        while True:
            variable_name = input(
                "\nWhich variables do you want to include (e.g., 'gender')? (type"
                " 'done' to finish and write a json objects for the variables"
                " you've specified)\n:"
            )

            if variable_name == "done":
                break

            variable = Variable(name=variable_name)

            column_names_input = input(
                "\nWhat columns correspond to this variable? (comma-delimited, e.g.,"
                " 'v102, v103, v104'; press ENTER to use the name of the variable)\n:"
            )

            if column_names_input == "":
                column_names_input = variable_name

            column_names = [
                column_name.strip() for column_name in column_names_input.split(",")
            ]

            for question in self._create_question(column_names=column_names):
                variable.upsert_question(question=question)

            self.variables.append(variable)

            # Export every time a variable is added to not accidentally lose progress.
            self.export_config(config_filename=config_filename)

    def generate_atp_config(self, config_filename: str):
        config_path = Path(config_filename)
        info_csv_path = config_path.parent / "info.csv"

        info_df = pd.read_csv(info_csv_path)

        for variable_name in list(info_df["key"]):
            variable_row = info_df[info_df["key"] == variable_name].iloc[0]
            question_text = str(variable_row["question"])
            option_mapping = eval(str(variable_row["option_mapping"]))

            original_options = list(option_mapping.values())
            valid_options = [o for o in original_options if o != "Refused"]
            invalid_options = [o for o in original_options if o == "Refused"]

            question = Question(
                key=variable_name,
                text=question_text,
                valid_options=[
                    ValidOption(raw=option, text=option).to_dict()
                    for option in valid_options
                ],
                invalid_options=invalid_options,
            ).to_dict()

            variable = Variable(name=variable_name, questions=[question])

            self.variables.append(variable)

            # Export every time a variable is added to not accidentally lose progress.
            self.export_config(config_filename=config_filename)

    def export_config(self, config_filename: str):
        with open(config_filename, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    def to_dict(self) -> typing.List[typing.Dict]:
        return [variable.to_dict() for variable in self.variables]

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

            for name, dependent_variable in self._dependent_variables.items():
                if n_sampled_per_dependent_variable[
                    name
                ] >= n_samples_per_dependent_variable or not dependent_variable.is_valid(
                    row
                ):
                    continue

                dependent_variable_prompt = dependent_variable.to_prompt(row)

                prompt = self._templatize(
                    independent_variable_summary=independent_variable_summary,
                    dependent_variable_prompt=dependent_variable_prompt,
                )

                correct_completion = dependent_variable.get_correct_letter(row)
                possible_completions = [
                    f" {letter}"
                    for letter in dependent_variable.get_possible_letters(row)
                ]
                independent_variables = self._get_independent_variable_dict(row)

                completion = Completion(
                    possible_completions=possible_completions,
                    correct_completion=correct_completion,
                )

                yield DependentVariableSample(
                    variable_name=name,
                    question=dependent_variable.to_question(row),
                    independent_variables=independent_variables,
                    index=i,
                    prompt=prompt,
                    completion=completion,
                )

                n_sampled_per_dependent_variable[name] += 1

    def mutual_info_stats(self, include_demographics=False) -> pd.DataFrame:
        mutual_info_dvs = {}
        independent_variable_names = [iv.name for iv in self._independent_variables]
        dependent_variable_names = [
            dv.name for dv in self._dependent_variables.values()
        ]

        columns = (
            dependent_variable_names + independent_variable_names
            if include_demographics
            else dependent_variable_names
        )

        for dv in columns:
            dv_mi = {}
            for demographic in independent_variable_names:
                # Create a new table with only the dv and demographic columns where
                # neither include nans or "Refused":
                compliant_responses = self.df[[dv, demographic]].dropna()
                compliant_responses = compliant_responses[
                    compliant_responses[dv] != "Refused"
                ]
                cleaned_dv_col = compliant_responses.iloc[:, 0].dropna()
                cleaned_demographic_col = compliant_responses.iloc[:, 1].dropna()
                mi = normalized_mutual_info_score(
                    cleaned_dv_col, cleaned_demographic_col
                )
                dv_mi[demographic] = mi
            mutual_info_dvs[dv] = {"average": np.mean(list(dv_mi.values())), **dv_mi}
        mi_df = pd.DataFrame.from_dict(mutual_info_dvs, orient="index")
        mi_df = mi_df.sort_values(by="average", ascending=False)
        return mi_df

    def __iter__(
        self,
    ) -> typing.Iterator[DependentVariableSample]:
        return self.iterate()


if __name__ == "__main__":
    # survey_name = "cces"

    # survey_directory = os.path.join("data", survey_name)

    # data_filename = os.path.join(survey_directory, "data.csv")
    # config_filename = os.path.join(survey_directory, "config.json")

    # survey = Survey(
    #     name=survey_name,
    #     data_filename=data_filename,
    #     config_filename=config_filename,
    # )

    # survey.generate_config(
    #     config_filename=config_filename,
    # )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--survey_name",
        type=str,
        default="roper",
    )

    args = parser.parse_args()

    survey_directory = os.path.join("data", args.survey_name)

    with open(
        os.path.join(survey_directory, "independent-variables.json"), "r"
    ) as file:
        independent_variable_names = json.load(file)

    with open(os.path.join(survey_directory, "dependent-variables.json"), "r") as file:
        dependent_variable_names = json.load(file)

    survey = Survey(
        name="roper",
        data_filename=os.path.join(survey_directory, "data.csv"),
        config_filename=os.path.join(survey_directory, "config.json"),
        independent_variable_names=independent_variable_names,
        dependent_variable_names=dependent_variable_names,
    )

    question_samples = list(iter(survey))

    print(
        f"{len(question_samples) / len(dependent_variable_names) / len(survey.df) * 100:.2f}%"
    )
