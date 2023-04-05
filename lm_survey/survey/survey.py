import typing
import numpy as np
import pandas as pd

from lm_survey.survey.dependent_variable_sample import DependentVariableSample
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
        self.df = pd.read_csv(data_filename)

        if config_filename is not None:
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
                option_index = int(exception)

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

    def _get_options(
        self, key: str
    ) -> typing.Tuple[typing.List[ValidOption], typing.List[str]]:
        try:
            unique_raw_options = self.df[key].unique()
        except KeyError:
            raise ValueError(f"{key} is not a valid column name.")

        unique_raw_options.sort()

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
            valid_indices=valid_indices, unique_raw_options=unique_raw_options
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
                " 'v102, v103, v104')\n:"
            )

            column_names = [
                column_name.strip() for column_name in column_names_input.split(",")
            ]

            for question in self._create_question(column_names=column_names):
                variable.upsert_question(question=question)

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
    survey = Survey(
        name="test",
        data_filename="data/roper/data.csv",
    )

    survey.generate_config(config_filename="data/roper/config-test.json")

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-d",
    #     "--data_filename",
    #     type=str,
    #     default="data/roper/data.csv",
    #     help="The filename of the data.",
    # )
    # parser.add_argument(
    #     "-c",
    #     "--config_filename",
    #     type=str,
    #     default="data/roper/config.json",
    #     help="The filename of the independent variables.",
    # )
    # args = parser.parse_args()

    # independent_variable_names = [
    #     "age",
    #     "party",
    #     "ideology",
    #     "religion",
    #     "marital",
    #     "employment",
    #     "education",
    #     "income",
    #     "ethnicity",
    #     "gender",
    # ]

    # dependent_variable_names = [
    #     "q1g",
    #     "q8a",
    #     "q8b",
    #     "q8c",
    #     "q9",
    #     "q10",
    #     "q11a",
    #     "q11b",
    #     "q11c",
    #     "q11d",
    #     "q11e",
    #     "q13",
    #     "q14",
    #     "q16",
    #     "q17",
    #     "q18",
    #     "q19",
    #     "q20",
    #     "q21",
    #     "q22",
    #     "q22a",
    #     "q23",
    #     "q23a",
    #     "q24",
    #     "q24a",
    #     "q25",
    #     "q26a",
    #     "q26b",
    #     "q26c",
    #     "q27",
    #     "q28",
    #     "q29",
    #     "q30",
    #     "q31",
    #     "q32a",
    #     "q32b",
    #     "q32c",
    #     "q33",
    #     "q34a",
    #     "q34b",
    #     "abort1",
    #     "abort2",
    # ]

    # survey = Survey(
    #     name="roper",
    #     data_filename=args.data_filename,
    #     config_filename=args.config_filename,
    #     independent_variable_names=independent_variable_names,
    #     dependent_variable_names=dependent_variable_names,
    # )

    # prompt_info = next(iter(survey))
    # prompt_info.completion = " C)"

    # print(prompt_info)
