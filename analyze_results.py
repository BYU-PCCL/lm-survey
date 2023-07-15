import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

from lm_survey.survey import SurveyResults


def analyze_results(survey_results: SurveyResults, slices: List[str]) -> pd.DataFrame:
    return survey_results.summarize(slice_by=slices).round(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--experiment_results",
        type=str,
        help="The name of the experiment to run.",
        required=True,
    )

    parser.add_argument(
        "-s",
        "--slices",
        nargs="+",
        type=str,
        help="The slices to analyze.",
        default=[],
    )

    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Whether to run in debug mode.",
    )

    args = parser.parse_args()

    experiment_results = Path(args.experiment_results)

    if not experiment_results.exists():
        raise ValueError(f"Path {experiment_results} does not exist.")

    with open(experiment_results, "r") as file:
        question_samples = json.load(file)

    survey_results = SurveyResults(question_samples=question_samples)

    analysis_path = experiment_results.parent / "analysis.json"

    summary_df = analyze_results(survey_results, args.slices)

    if args.debug:
        print(summary_df)
    else:
        summary_df.to_json(analysis_path, indent=4)
