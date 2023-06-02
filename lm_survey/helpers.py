import typing
import json
from pathlib import Path
from lm_survey.survey.dependent_variable_sample import DependentVariableSample


def json_to_dvs_list(results: typing.List[typing.Dict[str, typing.Any]]):
    dvs_list = [
        DependentVariableSample(
            **sample_dict,
        )
        for sample_dict in results
    ]
    return dvs_list


def filepath_to_dvs_list(filepath: typing.Union[str, Path]):
    with open(filepath, "r") as file:
        results = json.load(file)
    return json_to_dvs_list(results)


def dvs_list_to_json_file(
    dvs_list: typing.List[DependentVariableSample],
    filepath: typing.Union[str, Path],
):
    results = [question_sample.to_dict() for question_sample in dvs_list]
    with open(filepath, "w") as file:
        json.dump(results, file)
