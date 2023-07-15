import argparse
import json
from pathlib import Path

import numpy as np

from analyze_results import analyze_results
from lm_survey.survey.results import SurveyResults
from matplotlib import pyplot as plt

plt.style.use("ggplot")


def load_question_samples(results_path: Path) -> SurveyResults:
    with open(results_path, "r") as file:
        question_samples = json.load(file)

    return SurveyResults(question_samples=question_samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--survey",
        type=str,
        default="atp",
    )

    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        default="abortion",
    )

    parser.add_argument(
        "-i",
        "--ideology",
        type=str,
        default="right",
        help="The ideology of the persona",
        choices=["left", "center", "right", "none"],
    )

    args = parser.parse_args()

    experiment_path = Path(
        "experiments",
        args.experiment,
        args.survey,
        "llama-30b-hf/instruct",
        args.ideology,
    )
    results_path = experiment_path / "results.json"

    question_samples = load_question_samples(results_path)

    df = analyze_results(question_samples, ["variable_name"])

    df.sort_values(by="95%_lower_bound_gain", inplace=True, ascending=False)

    # Plot a bar chart of the question distribution

    plt.figure(figsize=(15, 5))

    bar_width = 0.3

    plt.bar(
        np.arange(df.index.size) - bar_width / 2,
        df["mean"],
        yerr=df["std_error"],
        capsize=2,
        ecolor="black",
        alpha=0.7,
        width=bar_width,
        label=f"LLaMA 33B + Instruct + {experiment_path.name.title()}",
    )
    plt.bar(
        np.arange(df.index.size) + bar_width / 2,
        df["baseline"],
        alpha=0.7,
        width=bar_width,
        label="Baseline",
    )

    plt.xticks(np.arange(df.index.size), df.index, rotation=90)
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()

    output_path = Path(
        "plots",
        args.experiment,
        args.survey,
        "question-distribution",
        f"{args.ideology}.png",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path, dpi=200)

    with open(experiment_path / "question-rankings.json", "w") as file:
        question_rankings = dict(
            zip(
                df.index.tolist(),
                df["95%_lower_bound_gain"].tolist(),
            )
        )
        json.dump(question_rankings, file, indent=2)
