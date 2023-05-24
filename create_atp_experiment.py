import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from lm_survey.samplers import AutoSampler
from lm_survey.survey import DependentVariableSample, Survey


def main(survey_name: str, experiment_name: str) -> None:
    data_dir = Path("data") / survey_name
    experiment_dir = Path("experiments") / experiment_name / survey_name

    # create experiment dir
    if not experiment_dir.exists():
        experiment_dir.mkdir(parents=True, exist_ok=True)

    info_csv_path = data_dir / "info.csv"
    metadata_csv_path = data_dir / "metadata.csv"

    info_df = pd.read_csv(info_csv_path)
    metadata_df = pd.read_csv(metadata_csv_path)

    experiment_config = {
        "independent_variable_names": list(metadata_df["key"]),
        "dependent_variable_names": list(info_df["key"])
    }

    with (experiment_dir / "config.json").open("w") as file:
        json.dump(experiment_config, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s",
        "--survey_name",
        type=str,
        default="all",
    )
    parser.add_argument(
        "-e",
        "--experiment_name",
        type=str,
        default="default",
    )

    args = parser.parse_args()

    if args.survey_name == "all":
        paths = sorted(Path("data").glob("ATP/American*/"))
        for path in tqdm(paths):
            args.survey_name = str(path.relative_to("data"))
            main(survey_name=args.survey_name, experiment_name=args.experiment_name)
    else:
        main(survey_name=args.survey_name, experiment_name=args.experiment_name)
