import argparse
import json
import os
import typing
from pathlib import Path

from tqdm import tqdm

from lm_survey.survey import Survey


def check_survey_prompts(
    survey_name: str,
    experiment_name: str,
):
    data_dir = os.path.join("data", survey_name)
    variables_dir = os.path.join("variables", survey_name)
    experiment_dir = os.path.join("experiments", experiment_name, survey_name)

    with open(os.path.join(experiment_dir, "config.json"), "r") as file:
        config = json.load(file)

    print(os.path.join(variables_dir, "variables.json"))

    survey = Survey(
        name=survey_name,
        data_filename=os.path.join(data_dir, "data.csv"),
        variables_filename=os.path.join(variables_dir, "variables.json"),
        independent_variable_names=config["independent_variable_names"],
        dependent_variable_names=config["dependent_variable_names"],
    )

    next_survey_sample = next(survey.iterate())
    print(f"## EXAMPLE PROMPT FOR {data_dir}:")
    print()
    print('"""')
    print(
        f"{next_survey_sample.prompt}█{next_survey_sample.completion.correct_completion}"
    )
    print('"""')
    print()
    print(f"## DEMOGRAPHICS NATURAL LANGUAGE SUMMARY FOR {data_dir}:")
    print()
    survey.print_demographics_natural_language_summary()


def main(survey_directories: typing.List[Path], experiment_name: str) -> None:
    for survey_directory in survey_directories:
        check_survey_prompts(survey_directory, experiment_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Positional argument for survey dir(s)
    parser.add_argument(
        "survey_directory",
        nargs="+",
        type=Path,
    )
    parser.add_argument(
        "-e",
        "--experiment_name",
        type=str,
        default="default",
    )

    args = parser.parse_args()

    main(survey_directories=args.survey_directory, experiment_name=args.experiment_name)
