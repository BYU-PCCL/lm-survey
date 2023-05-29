import argparse
import json
import os
import typing

import numpy as np
import pandas as pd
from tqdm import tqdm

from lm_survey.samplers import AutoSampler, BaseSampler
from lm_survey.survey import Survey


def estimate_survey_costs(
    sampler: BaseSampler,
    survey_name: str,
    experiment_name: str,
    *,
    n_samples_per_dependent_variable: typing.Optional[int] = None,
    n_top_mutual_info_dvs: typing.Optional[int] = None,
):
    data_dir = os.path.join("data", survey_name)
    variables_dir = os.path.join("variables", survey_name)
    experiment_dir = os.path.join("experiments", experiment_name, survey_name)

    with open(os.path.join(experiment_dir, "config.json"), "r") as file:
        config = json.load(file)

    survey = Survey(
        name=survey_name,
        data_filename=os.path.join(data_dir, "data.csv"),
        variables_filename=os.path.join(variables_dir, "variables.json"),
        independent_variable_names=config["independent_variable_names"],
        dependent_variable_names=config["dependent_variable_names"],
    )

    dependent_variable_samples = list(
        survey.iterate(
            n_samples_per_dependent_variable=n_samples_per_dependent_variable
        )
    )

    prompt_count = len(dependent_variable_samples)

    if hasattr(sampler, "batch_estimate_prompt_cost"):
        completion_costs = sampler.batch_estimate_prompt_cost(
            [
                dependent_variable_sample.completion_prompt
                for dependent_variable_sample in dependent_variable_samples
            ]
        )
    else:
        completion_costs = []
        for dependent_variable_sample in tqdm(dependent_variable_samples):
            completion_cost = sampler.estimate_prompt_cost(
                dependent_variable_sample.completion_prompt
            )
            completion_costs.append(completion_cost)

    total_completion_cost = np.sum(completion_costs)

    return {
        "prompt_count": prompt_count,
        "cost": total_completion_cost,
    }


def main(
    model_name: str,
    survey_names: typing.List[str],
    experiment_name: str,
    n_samples_per_dependent_variable: typing.Optional[int] = None,
    n_top_mutual_info_dvs: typing.Optional[int] = None,
) -> None:
    sampler = AutoSampler(model_name=model_name)

    survey_costs = {}
    for survey_name in tqdm(survey_names):
        estimate = estimate_survey_costs(
            sampler=sampler,
            survey_name=survey_name,
            experiment_name=experiment_name,
            n_samples_per_dependent_variable=n_samples_per_dependent_variable,
            n_top_mutual_info_dvs=n_top_mutual_info_dvs,
        )
        survey_costs[survey_name] = estimate

    total_cost = sum([estimate["cost"] for estimate in survey_costs.values()])

    total_prompt_count = sum(
        [estimate["prompt_count"] for estimate in survey_costs.values()]
    )

    if len(survey_names) > 1:
        print(f"Cost per survey:")
        for survey_name, survey_cost in survey_costs.items():
            print(
                f"{survey_name}: ${(survey_cost['cost'] / 100):.2f} ({survey_cost['prompt_count']}"
                " prompts)"
            )

    print(f"Total cost: ${(total_cost / 100):.2f} ({total_prompt_count} prompts)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-n",
        "--n_samples_per_dependent_variable",
        type=int,
    )
    parser.add_argument(
        "--n_top_mutual_info_dvs",
        type=int,
    )
    parser.add_argument(
        "-e",
        "--experiment_name",
        type=str,
        default="default",
    )
    # Positional argument for survey dir(s)
    parser.add_argument(
        "survey_name",
        nargs="+",
        type=str,
    )

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        survey_names=args.survey_name,
        experiment_name=args.experiment_name,
        n_samples_per_dependent_variable=args.n_samples_per_dependent_variable,
        n_top_mutual_info_dvs=args.n_top_mutual_info_dvs,
    )
