import argparse
import json
import os
import typing

import numpy as np
from tqdm import tqdm

from lm_survey.samplers import AutoSampler
from lm_survey.survey import DependentVariableSample, Survey


def parse_model_name(model_name: str) -> str:
    if model_name.startswith("/") and model_name.endswith("/"):
        model_name = model_name.split("/")[-2]
    elif model_name.startswith("/"):
        model_name = model_name.split("/")[-1]
    else:
        model_name = model_name.replace("/", "-")

    return model_name


def get_commit_hash():
    commit_hash = os.popen("git rev-parse HEAD").read().strip()
    return commit_hash


def save_experiment(
    model_name: str,
    experiment_dir: str,
    dependent_variable_samples: typing.List[DependentVariableSample],
    prompt_name: str,
    n_samples_per_dependent_variable: typing.Optional[int] = None,
):
    parsed_model_name = parse_model_name(model_name)

    results = [
        question_sample.to_dict() for question_sample in dependent_variable_samples
    ]

    metadata = {
        "model_name": model_name,
        "n_samples_per_dependent_variable": n_samples_per_dependent_variable,
        "commit_hash": get_commit_hash(),
        "prompt_name": prompt_name,
    }

    experiment_metadata_dir = os.path.join(experiment_dir, parsed_model_name)

    if not os.path.exists(experiment_metadata_dir):
        os.makedirs(experiment_metadata_dir)

    with open(os.path.join(experiment_metadata_dir, "metadata.json"), "w") as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    with open(os.path.join(experiment_metadata_dir, "results.json"), "w") as file:
        json.dump(
            results,
            file,
            indent=4,
        )


def calculate_accuracy(
    dependent_variable_samples: typing.List[DependentVariableSample],
) -> float:
    scores = [
        dependent_variable_sample.completion.is_completion_correct
        for dependent_variable_sample in dependent_variable_samples
    ]

    return np.mean(scores)


def main(
    model_name: str,
    survey_name: str,
    experiment_name: str,
    prompt_name: str,
    n_samples_per_dependent_variable: typing.Optional[int] = None,
) -> None:
    data_dir = os.path.join("data", survey_name)
    variable_dir = os.path.join("variables", survey_name)
    experiment_dir = os.path.join("experiments", experiment_name, survey_name)

    with open(os.path.join(experiment_dir, "config.json"), "r") as file:
        config = json.load(file)

    survey = Survey(
        name=survey_name,
        data_filename=os.path.join(data_dir, "data.csv"),
        variables_filename=os.path.join(variable_dir, "variables.json"),
        independent_variable_names=config["independent_variable_names"],
        dependent_variable_names=config["dependent_variable_names"],
    )

    sampler = AutoSampler(model_name=model_name)

    dependent_variable_samples = list(
        survey.iterate(
            n_samples_per_dependent_variable=n_samples_per_dependent_variable,
            prompt_name=prompt_name,
        )
    )

    loop = tqdm(dependent_variable_samples)

    for dependent_variable_sample in loop:
        completion_log_probs = sampler.rank_completions(
            prompt=dependent_variable_sample.prompt,
            completions=dependent_variable_sample.completion.possible_completions,
        )
        dependent_variable_sample.completion.set_completion_log_probs(
            completion_log_probs
        )

        accuracy = calculate_accuracy(dependent_variable_samples)

        loop.set_description(f"Accuracy: {accuracy * 100:.2f}%")

    print(
        f"Accuracy: {accuracy * 100:.2f}% ({len(dependent_variable_samples)} samples)"
    )

    save_experiment(
        model_name=model_name,
        experiment_dir=experiment_dir,
        dependent_variable_samples=dependent_variable_samples,
        prompt_name=prompt_name,
        n_samples_per_dependent_variable=n_samples_per_dependent_variable,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-s",
        "--survey_name",
        type=str,
        default="roper",
    )
    parser.add_argument(
        "-n",
        "--n_samples_per_dependent_variable",
        type=int,
    )
    parser.add_argument(
        "-e",
        "--experiment_name",
        type=str,
        default="default",
    )
    parser.add_argument(
        "-p",
        "--prompt_name",
        type=str,
        default="first_person_natural_language_context",
    )

    args = parser.parse_args()

    # To prevent overbilling OpenAI.
    if (
        args.model_name.startswith("gpt3")
        and args.n_samples_per_dependent_variable is None
    ):
        args.n_samples_per_dependent_variable = 100

    main(
        model_name=args.model_name,
        survey_name=args.survey_name,
        experiment_name=args.experiment_name,
        n_samples_per_dependent_variable=args.n_samples_per_dependent_variable,
        prompt_name=args.prompt_name,
    )
