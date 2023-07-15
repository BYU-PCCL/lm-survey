from pathlib import Path
from typing import List

import numpy as np
from lm_survey.survey.results import SurveyResults
from matplotlib import pyplot as plt
import pandas as pd
import json
import argparse
from analyze_results import analyze_results

plt.style.use("ggplot")

atp_label_map = {
    "CREGION": "Region",
    "AGE": "Age",
    "SEX": "Sex",
    "EDUCATION": "Education",
    "CITIZEN": "Citizenship",
    "MARITAL": "Marital Status",
    "RELIG": "Religion",
    "RELIGATTEND": "Religious Attendance",
    "POLPARTY": "Political Party",
    "INCOME": "Income",
    "POLIDEOLOGY": "Political Ideology",
    "RACE": "Race",
}


def get_plot_name(slices: List[str]) -> str:
    return f'{"-".join(slices).lower()}.png' if slices else "all.png"


def load_question_samples(results_path: Path) -> SurveyResults:
    with open(results_path, "r") as file:
        question_samples = json.load(file)

    return SurveyResults(question_samples=question_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--experiments",
        type=str,
        default="/home/ashaw8/compute/lm-survey/experiments/guns/atp_w92/llama-30b-hf/instruct",
        help="Path to results file",
    )

    parser.add_argument(
        "-s",
        "--slices",
        nargs="+",
        default=[],
        type=str,
        help="Path to slices file",
    )

    args = parser.parse_args()

    experiment_path = Path(args.experiments)

    results_paths = list(experiment_path.rglob("results.json"))

    question_samples_list = [
        load_question_samples(results_path) for results_path in results_paths
    ]

    dfs = [
        analyze_results(question_samples, args.slices)
        for question_samples in question_samples_list
    ]

    for i in range(len(dfs)):
        dfs[i]["ideology"] = results_paths[i].parent.name

    df = pd.concat(dfs, axis=0)

    groups = df.groupby("ideology")

    # Define bar width
    bar_width = 0.1
    ideologies = ["left", "center", "right", "none"]

    baseline = (
        df.groupby(args.slices)["baseline"].mean().values
        if args.slices != []
        else df["baseline"].values
    )

    plt.figure(figsize=(12, 5))
    # Generate grouped bar chart
    for ideology, data in groups:
        x = np.arange(len(data))

        i = ideologies.index(ideology)  # type: ignore

        plt.bar(
            x + i * bar_width,
            data["mean"],
            yerr=data["std_error"],
            align="center",
            alpha=0.7,
            ecolor="black",
            capsize=3,
            width=bar_width,
            label=ideology.title(),  # type: ignore
        )

    plt.bar(
        np.arange(len(next(iter(groups))[1])) + len(ideologies) * bar_width,
        baseline,  # type: ignore
        align="center",
        alpha=0.7,
        width=bar_width,
        label="Baseline",
    )

    # Set labels and title
    plt.ylabel("Accuracy")
    plt.xlabel(", ".join([atp_label_map[slice] for slice in args.slices]).title())
    plt.legend()

    x = np.arange(len(next(iter(groups))[1]))

    # Setting xticks in the middle of the grouped bars
    xticks_location = x + bar_width * (len(groups)) / 2
    plt.xticks(xticks_location, next(iter(groups))[1].index, rotation=45)
    plt.tight_layout()

    output_path = Path(
        "plots",
        *experiment_path.parts[-4:-2],
        "demographic-slices",
        get_plot_name(args.slices),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=200)
