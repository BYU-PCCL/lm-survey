import argparse
import json
import os
import typing
from pathlib import Path

from tqdm import tqdm

from lm_survey.survey import Survey


def check_survey_prompts(survey_directory: Path):
    survey_name = survey_directory.name

    with (survey_directory / "independent-variables.json").open("r") as file:
        independent_variable_names = json.load(file)

    with (survey_directory / "dependent-variables.json").open("r") as file:
        dependent_variable_names = json.load(file)

    survey = Survey(
        name=survey_name,
        data_filename=survey_directory / "data.csv",
        variables_filename=survey_directory / "config.json",
        independent_variable_names=independent_variable_names,
        dependent_variable_names=dependent_variable_names,
    )

    print(f"## EXAMPLE PROMPT FOR {survey_directory}:")
    print()
    print('"""')
    print(f"{next(iter(survey)).prompt}█")
    print('"""')
    print()
    print(f"## DEMOGRAPHICS NATURAL LANGUAGE SUMMARY FOR {survey_directory}:")
    print()
    survey.print_demographics_natural_language_summary()


def main(survey_directories: typing.List[Path]) -> None:
    for survey_directory in survey_directories:
        check_survey_prompts(survey_directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Positional argument for survey dir(s)
    parser.add_argument(
        "survey_directory",
        nargs="+",
        type=Path,
    )

    args = parser.parse_args()

    main(survey_directories=args.survey_directory)
