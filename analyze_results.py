import argparse
import json
from pathlib import Path
import typing

from lm_survey.survey import SurveyResults


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="The name of the experiment to run.",
        default="default",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="The name of the model to run.",
        default="llama-65b-hf",
    )

    parser.add_argument(
        "-s",
        "--survey",
        type=str,
        default="roper",
        help="The name of the survey to run.",
    )

    args = parser.parse_args()

    experiment_dir = Path("experiments", args.experiment, args.survey)
    results_filename = Path("results.json")

    paths: typing.Dict[str, Path] = {
        "center": experiment_dir / args.model / results_filename,
        # "left": experiment_dir / f"{args.model}-left" / results_filename,
        # "right": experiment_dir / f"{args.model}-right" / results_filename,
    }

    results = {}
    for ideology, path in paths.items():
        if not path.exists():
            raise ValueError(f"Path {path} does not exist.")

        with open(path, "r") as file:
            ideology_results = json.load(file)

        results[ideology] = ideology_results

    survey_results = {
        ideology: SurveyResults(question_samples=ideology_results)
        for ideology, ideology_results in results.items()
    }

    slices = []

    # for ideology, ideology_survey_results in survey_results.items():
    #     print(ideology)
    #     print(
    #         ideology_survey_results.get_mean_score(slice_by=slices).nlargest(
    #             3, "95%_lower_bound_gain"
    #         ),
    #         sep="\n",
    #         end="\n\n",
    #     )

    analysis_path = experiment_dir / args.model / "analysis.json"

    summary_df = (
        survey_results["center"]
        .summarize(slice_by=slices)
        .round(2)
        .to_json(analysis_path, indent=4)
    )
