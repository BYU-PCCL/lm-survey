import argparse
import asyncio
import json
import logging
import os
import typing

import numpy as np
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from lm_survey.samplers import AutoSampler
from lm_survey.survey import DependentVariableSample, Survey
from pathlib import Path


def infill_missing_responses(
    results_filepath,
) -> typing.Tuple[
    typing.List[DependentVariableSample],
    typing.List[DependentVariableSample],
]:
    with open(results_filepath, "r") as file:
        results = json.load(file)
    question_samples = [
        DependentVariableSample(
            **sample_dict,
        )
        for sample_dict in results
    ]
    # Find the question_samples that are missing responses
    missing_responses = [qs for qs in question_samples if not qs.has_response()]
    filled_responses = [qs for qs in question_samples if qs.has_response()]

    return missing_responses, filled_responses


def parse_model_name(model_name: str) -> str:
    if model_name.startswith("/"):
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
    n_samples_per_dependent_variable: typing.Optional[int] = None,
):
    parsed_model_name = parse_model_name(model_name)

    results = [
        question_sample.to_dict()
        for question_sample in dependent_variable_samples
    ]

    metadata = {
        "model_name": model_name,
        "n_samples_per_dependent_variable": n_samples_per_dependent_variable,
        "commit_hash": get_commit_hash(),
    }

    experiment_metadata_dir = os.path.join(experiment_dir, parsed_model_name)

    if not os.path.exists(experiment_metadata_dir):
        os.makedirs(experiment_metadata_dir)

    with open(
        os.path.join(experiment_metadata_dir, "metadata.json"), "w"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    with open(
        os.path.join(experiment_metadata_dir, "results.json"), "w"
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )


async def main(
    model_name: str,
    survey_name: str,
    experiment_name: str,
    n_samples_per_dependent_variable: typing.Optional[int] = None,
) -> None:
    data_dir = os.path.join("data", survey_name)
    variables_dir = os.path.join("variables", survey_name)
    experiment_dir = os.path.join("experiments", experiment_name, survey_name)

    if logging:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.ERROR)
        handler = logging.FileHandler(Path(experiment_dir) / "errors.log")
        handler.setLevel(logging.ERROR)
        formatter = logging.Formatter("%(asctime)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger = None

    with open(os.path.join(experiment_dir, "config.json"), "r") as file:
        config = json.load(file)

    survey = Survey(
        name=survey_name,
        data_filename=os.path.join(data_dir, "responses.csv"),
        variables_filename=os.path.join(variables_dir, "variables.json"),
        independent_variable_names=config["independent_variable_names"],
        dependent_variable_names=config["dependent_variable_names"],
    )

    sampler = AutoSampler(model_name=model_name, logger=logger)

    parsed_model_name = parse_model_name(model_name)
    experiment_metadata_dir = os.path.join(experiment_dir, parsed_model_name)
    # Use pathlib to check if experiment_metadata_dir/results.json exists.
    # If it does, then we default to filling in missing response objects instead of re-running the survey.
    results_path = Path(experiment_metadata_dir) / "results.json"
    if results_path.exists():
        (
            dependent_variable_samples,
            finished_samples,
        ) = infill_missing_responses(results_path)

    else:
        dependent_variable_samples = list(
            survey.iterate(
                n_samples_per_dependent_variable=n_samples_per_dependent_variable
            )
        )
        finished_samples = []

    # TODO: This is a really lame way to do this. We should probably do it another way,
    # especially because it seems obvious that the model name should not be tied to its
    # sampler implementation. This is a symptom of a very lazy async implementation.
    if sampler.model_name.startswith("async"):
        sample_coroutines = []
        for dependent_variable_sample in dependent_variable_samples:

            async def request_completion(dependent_variable_sample):
                (
                    completion_log_probs,
                    response_object,
                ) = await sampler.rank_completions(
                    prompt=dependent_variable_sample.prompt,
                    completions=dependent_variable_sample.completion.possible_completions,
                )  # type: ignore
                dependent_variable_sample.completion.set_completion_log_probs(
                    completion_log_probs
                )
                dependent_variable_sample.completion.response_object = (
                    response_object
                )

            sample_coroutines.append(
                request_completion(dependent_variable_sample)
            )

        await tqdm_asyncio.gather(*sample_coroutines)
    else:
        for dependent_variable_sample in tqdm(dependent_variable_samples):
            completion_log_probs, response_object = sampler.rank_completions(
                prompt=dependent_variable_sample.prompt,
                completions=dependent_variable_sample.completion.possible_completions,
            )  # type: ignore
            dependent_variable_sample.completion.set_completion_log_probs(
                completion_log_probs
            )
            dependent_variable_sample.completion.response_object = (
                response_object
            )

    accuracy = np.mean(
        [
            dependent_variable_sample.completion.is_completion_correct
            for dependent_variable_sample in dependent_variable_samples
        ]
    )

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
        f" ({len(dependent_variable_samples)} samples)"
    )

    dependent_variable_samples += finished_samples

    save_experiment(
        model_name=model_name,
        experiment_dir=experiment_dir,
        dependent_variable_samples=dependent_variable_samples,
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

    args = parser.parse_args()

    # To prevent overbilling OpenAI.
    # if (
    #     args.model_name.startswith("gpt3")
    #     and args.n_samples_per_dependent_variable is None
    # ):
    #     args.n_samples_per_dependent_variable = 100

    if args.survey_name == "all":
        paths = sorted(Path("data").glob("ATP/American*/"))
        for path in tqdm(paths):
            args.survey_name = str(path.relative_to("data"))
            asyncio.run(
                main(
                    model_name=args.model_name,
                    survey_name=args.survey_name,
                    experiment_name=args.experiment_name,
                    n_samples_per_dependent_variable=args.n_samples_per_dependent_variable,
                )
            )

    else:
        asyncio.run(
            main(
                model_name=args.model_name,
                survey_name=args.survey_name,
                experiment_name=args.experiment_name,
                n_samples_per_dependent_variable=args.n_samples_per_dependent_variable,
            )
        )
