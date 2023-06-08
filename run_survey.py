import argparse
import json
import os
from pathlib import Path
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
    experiment_dir: Path,
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

    experiment_metadata_dir = experiment_dir / parsed_model_name

    if not experiment_metadata_dir.exists():
        experiment_metadata_dir.mkdir(parents=True)

    with open(experiment_metadata_dir / "metadata.json", "w") as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    with open(experiment_metadata_dir / "results.json", "w") as file:
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
        if dependent_variable_sample.completion.are_completion_log_probs_set()
    ]

    return np.mean(scores)


def calculate_baseline(
    dependent_variable_samples: typing.List[DependentVariableSample],
) -> float:
    scores = [
        1 / len(dependent_variable_sample.completion.possible_completions)
        for dependent_variable_sample in dependent_variable_samples
        if dependent_variable_sample.completion.are_completion_log_probs_set()
    ]

    return np.mean(scores)


def main(
    model_name: str,
    survey_name: str,
    experiment_name: str,
    prompt_name: str,
    n_samples_per_dependent_variable: typing.Optional[int] = None,
) -> None:
    data_dir = Path("data", survey_name)
    experiment_dir = Path("experiments", experiment_name, survey_name)

    with open(Path(experiment_dir, "config.json"), "r") as file:
        config = json.load(file)

    # If there is a variables file in the experiment directory, use that.
    variables_filename = experiment_dir / "variables.json"

    # Otherwise, use the default variables file for the survey.
    if not variables_filename.exists():
        variables_filename = Path("variables", survey_name, "variables.json")

    survey = Survey(
        name=survey_name,
        data_filename=Path(data_dir, "data.csv"),
        variables_filename=variables_filename,
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
    accuracy = 0.0

    for dependent_variable_sample in loop:
        completion_log_probs = sampler.rank_completions(
            prompt=dependent_variable_sample.prompt,
            completions=dependent_variable_sample.completion.possible_completions,
        )
        dependent_variable_sample.completion.set_completion_log_probs(
            completion_log_probs
        )

        accuracy = calculate_accuracy(dependent_variable_samples)
        baseline = calculate_baseline(dependent_variable_samples)

        loop.set_description(
            f"Accuracy: {accuracy * 100:.2f}%, Baseline: {baseline * 100:.2f}%"
        )

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
