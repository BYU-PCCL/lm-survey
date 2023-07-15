import argparse
import json
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-e",
        "--experiment",
        type=str,
        help="The name of the experiment to run.",
        required=True,
        choices=["abortion", "guns", "immigration"],
    )

    args = parser.parse_args()

    experiment_path = Path("experiments", args.experiment)

    combined_config_path = experiment_path / "atp/config.json"
    combined_variables_path = experiment_path / "atp/variables.json"

    combined_config_path.parent.mkdir(parents=True, exist_ok=True)

    atp_configs = experiment_path.rglob("atp*/config.json")
    atp_variables = experiment_path.rglob("atp*/variables.json")

    combined_config = {
        "dependent_variable_names": [],
        "independent_variable_names": [],
    }
    combined_variables = []

    for i, (config_path, variable_path) in enumerate(zip(atp_configs, atp_variables)):
        with open(config_path, "r") as file:
            config = json.load(file)

        with open(variable_path, "r") as file:
            variables = json.load(file)

        if i == 0:
            combined_config["independent_variable_names"].extend(
                config["independent_variable_names"]
            )
            combined_variables.extend(variables)
        else:
            combined_variables.extend(
                [
                    variable
                    for variable in variables
                    if variable["name"] not in config["independent_variable_names"]
                ]
            )

        combined_config["dependent_variable_names"].extend(
            config["dependent_variable_names"]
        )

    with open(combined_config_path, "w") as file:
        json.dump(combined_config, file, indent=4)

    with open(combined_variables_path, "w") as file:
        json.dump(combined_variables, file, indent=4)

    ideologies = ["left", "center", "right", "none"]

    for ideology in ideologies:
        combined_results_path = (
            experiment_path / f"atp/llama-30b-hf/instruct/{ideology}/results.json"
        )
        combined_results_path.parent.mkdir(parents=True, exist_ok=True)

        instruct_results_paths = experiment_path.rglob(
            f"atp*/llama-30b-hf/instruct/{ideology}/results.json"
        )

        combined_results = []
        for results_path in instruct_results_paths:
            with open(results_path, "r") as file:
                results = json.load(file)

            combined_results.extend(results)

        with open(combined_results_path, "w") as file:
            json.dump(combined_results, file, indent=4)
