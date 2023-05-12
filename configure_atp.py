import argparse
import os
from pathlib import Path

from lm_survey.survey.survey import Survey

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wave", type=int, help="Wave of ATP to configure")
    args = parser.parse_args()

    survey_directory = Path("data", "ATP", f"American_Trends_Panel_W{args.wave}")

    survey = Survey(
        name="ATP_W92",
        data_filename=survey_directory / "responses.csv",
        config_filename=survey_directory / "config.json",
    )

    survey.generate_dv_config(survey_directory / "config.json")
