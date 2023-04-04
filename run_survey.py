import typing
from lm_survey.survey import Survey, DependentVariableSample
from lm_survey.samplers import AutoSampler
from tqdm import tqdm
import json
import os
import argparse


def save_results(
    question_samples: typing.List[DependentVariableSample],
    model_name: str,
    survey_name: str,
):
    new_results = {
        model_name: [
            {
                **question_sample.to_dict(),
            }
            for question_sample in question_samples
        ]
    }

    output_filepath = os.path.join(
        "results",
        survey_name,
        "results.json",
    )

    if not os.path.exists(output_filepath):
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        results = {}
    else:
        with open(output_filepath, "r") as file:
            results = json.load(file)

    results.update(new_results)

    with open(output_filepath, "w") as file:
        json.dump(results, file, indent=4)


def main(
    model_name: str,
    survey_name: str,
    n_samples_per_dependent_variable: typing.Optional[int] = None,
) -> None:
    survey_directory = os.path.join("data", survey_name)

    with open(
        os.path.join(survey_directory, "independent-variables.json"), "r"
    ) as file:
        independent_variable_names = json.load(file)

    with open(os.path.join(survey_directory, "dependent-variables.json"), "r") as file:
        dependent_variable_names = json.load(file)

    survey = Survey(
        name=survey_name,
        data_filename=os.path.join(survey_directory, "data.csv"),
        config_filename=os.path.join(survey_directory, "config.json"),
        independent_variable_names=independent_variable_names,
        dependent_variable_names=dependent_variable_names,
    )

    sampler = AutoSampler(model_name=model_name)

    question_samples = list(
        survey.iterate(
            n_samples_per_dependent_variable=n_samples_per_dependent_variable
        )
    )

    for question_sample in tqdm(question_samples):
        completion = sampler.get_best_next_token(prompt=question_sample.prompt)
        question_sample.completion = completion

    save_results(
        question_samples=question_samples,
        model_name=model_name,
        survey_name=survey_name,
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
        n_samples_per_dependent_variable=args.n_samples_per_dependent_variable,
    )
