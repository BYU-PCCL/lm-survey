import argparse
import json
import os
from pathlib import Path

from lm_survey.survey.survey import Survey

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "wave", type=Path, nargs="+", help="Path(s) to wave of ATP to configure"
    )
    parser.add_argument("output_path", type=Path, help="Path to output directory")
    parser.add_argument(
        "--base-variables", type=Path, help="Path to optional base variables"
    )
    args = parser.parse_args()

    for wave in args.wave:
        variables_path = wave / "variables.json"

        survey = Survey(name="ATP_W92", data_filename=wave / "data.csv")

        wave_output_dir = args.output_path / wave
        wave_output_dir.mkdir(parents=True, exist_ok=True)

        survey.generate_atp_schema(wave, wave_output_dir / "variables.json")

        # This is a simple way to put some extra stuff in the variables file
        if args.base_variables:
            with args.base_variables.open("r") as f:
                base_variables = json.load(f)
            with variables_path.open("r") as f:
                config = json.load(f)
            config.extend(base_variables)
            with variables_path.open("w") as f:
                json.dump(config, f, indent=2)
