import typing
import json
from pathlib import Path
from lm_survey.survey.dependent_variable_sample import DependentVariableSample
import pandas as pd


def json_to_dvs_list(
    results: typing.List[typing.Dict[str, typing.Any]],
    add_weights: bool = False,
):
    """add_weights looks at the original data corresponding to the results object and extracts the weights from there before generating DVSs"""

    if add_weights:
        wave = results[0]["variable_name"][-3:]
        input_filepath = (
            Path("data") / f"ATP/American_Trends_Panel_{wave}/responses.csv"
        )

        original_df = pd.read_csv(input_filepath)
        weight_key = [w for w in original_df.columns if w == f"WEIGHT_{wave}"]
        assert len(weight_key) == 1
        weight_key = weight_key[0]
        weights = original_df[weight_key]

        for result in results:
            result["weight"] = weights.loc[result["index"]]

    dvs_list = [
        DependentVariableSample(
            **sample_dict,
        )
        for sample_dict in results
    ]
    return dvs_list


def filepath_to_dvs_list(
    filepath: typing.Union[str, Path],
    add_weights: bool = False,
):
    with open(filepath, "r") as file:
        results = json.load(file)
    return json_to_dvs_list(results, add_weights=add_weights)


def dvs_list_to_json_file(
    dvs_list: typing.List[DependentVariableSample],
    filepath: typing.Union[str, Path],
):
    results = [question_sample.to_dict() for question_sample in dvs_list]
    with open(filepath, "w") as file:
        json.dump(results, file)
