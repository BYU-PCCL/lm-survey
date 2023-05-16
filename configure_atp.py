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
    parser.add_argument("--base-config", type=Path, help="Path to optional base config")
    args = parser.parse_args()

    for wave in args.wave:
        config_path = wave / "config.json"

        survey = Survey(name="ATP_W92", data_filename=wave / "responses.csv")

        survey.generate_atp_config(config_path)

        # This is a simple way to put some extra stuff in the config
        if args.base_config:
            with args.base_config.open("r") as f:
                base_config = json.load(f)
            with config_path.open("r") as f:
                config = json.load(f)
            config.extend(base_config)
            with config_path.open("w") as f:
                json.dump(config, f, indent=2)
