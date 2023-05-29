import typing
import json
from pathlib import Path
from lm_survey.survey.dependent_variable_sample import DependentVariableSample


def json_to_DVS_list(results: typing.List[typing.Dict[str, typing.Any]]):
    DVS_list = [
        DependentVariableSample(
            **sample_dict,
        )
        for sample_dict in results
    ]
    return DVS_list


def filepath_to_DVS_list(filepath: typing.Union[str, Path]):
    with open(filepath, "r") as file:
        results = json.load(file)
    return json_to_DVS_list(results)


def DVS_list_to_json_file(
    DVS_list: typing.List[DependentVariableSample],
    filepath: typing.Union[str, Path],
):
    results = [question_sample.to_dict() for question_sample in DVS_list]
    with open(filepath, "w") as file:
        json.dump(results, file)
