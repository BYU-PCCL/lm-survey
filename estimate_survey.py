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
    *,
    n_samples_per_dependent_variable: typing.Optional[int] = None,
    n_top_mutual_info_dvs: typing.Optional[int] = None,
):
    # TODO(vinhowe): fix this
    survey_directory = survey_name

    with open(
        os.path.join(survey_directory, "independent-variables.json"), "r"
    ) as file:
        independent_variable_names = json.load(file)

    with open(os.path.join(survey_directory, "dependent-variables.json"), "r") as file:
        dependent_variable_names = json.load(file)

    data_filename = os.path.join(survey_directory, "responses.csv")
    variables_filename = os.path.join(survey_directory, "config.json")

    survey = Survey(
        name=survey_name,
        data_filename=data_filename,
        variables_filename=variables_filename,
        independent_variable_names=independent_variable_names,
        dependent_variable_names=dependent_variable_names,
    )

    if n_top_mutual_info_dvs is not None:
        cached_mutual_info_stats_filename = os.path.join(
            survey_directory, "cached_mutual_info_stats.csv"
        )
        if os.path.exists(cached_mutual_info_stats_filename):
            mutual_info_stats = pd.read_csv(
                cached_mutual_info_stats_filename, index_col=0
            )
        else:
            mutual_info_stats = survey.mutual_info_stats()
            mutual_info_stats.to_csv(cached_mutual_info_stats_filename)
        dependent_variable_names = mutual_info_stats.index[:n_top_mutual_info_dvs]
        survey = Survey(
            name=survey_name,
            data_filename=data_filename,
            variables_filename=variables_filename,
            independent_variable_names=independent_variable_names,
            dependent_variable_names=dependent_variable_names,
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
                dependent_variable_sample.prompt
                for dependent_variable_sample in dependent_variable_samples
            ]
        )
    else:
        completion_costs = []
        for dependent_variable_sample in tqdm(dependent_variable_samples):
            completion_cost = sampler.estimate_prompt_cost(
                dependent_variable_sample.prompt
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
    n_samples_per_dependent_variable: typing.Optional[int] = None,
    n_top_mutual_info_dvs: typing.Optional[int] = None,
) -> None:
    sampler = AutoSampler(model_name=model_name)

    survey_costs = {}
    for survey_name in tqdm(survey_names):
        estimate = estimate_survey_costs(
            sampler=sampler,
            survey_name=survey_name,
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
        n_samples_per_dependent_variable=args.n_samples_per_dependent_variable,
        n_top_mutual_info_dvs=args.n_top_mutual_info_dvs,
    )
