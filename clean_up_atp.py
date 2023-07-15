from pathlib import Path
import json


def replace(text):
    if text.startswith("I am"):
        text = text.replace("I am", "You are")

    if text.startswith("I "):
        text = text.replace("I ", "You ")

    text = (
        text.replace("my ", "your ")
        .replace("My ", "Your ")
        .replace("Mine ", "Yours ")
        .replace("mine", "yours")
        .replace("I am", "you are")
        .replace("I ", "you ")
    )

    return text


def switch_to_second_person(variable):
    for i in range(len(variable["questions"][0]["valid_options"])):
        variable["questions"][0]["valid_options"][i]["natural_language"] = replace(
            variable["questions"][0]["valid_options"][i]["natural_language"]
        )

    return variable


experiment_dir = Path("experiments")
variable_dir = Path("variables")

experiments = ["abortion", "guns", "immigration"]

configs = [
    (
        experiment_path,
        variable_dir / "ATP" / experiment_path.parent.name / "variables.json",
    )
    for experiment in experiments
    for experiment_path in (experiment_dir / experiment).rglob("atp*/config.json")
]

for config_path, variables_path in configs:
    with open(config_path, "r") as f:
        config = json.load(f)

    keys = set(config["dependent_variable_names"]) | set(
        config["independent_variable_names"]
    )

    with open(variables_path, "r") as f:
        variables = json.load(f)

    filtered_variables = [
        variable for variable in variables if variable["name"] in keys
    ]

    for i in range(len(filtered_variables)):
        if filtered_variables[i]["name"] in config["independent_variable_names"]:
            filtered_variables[i] = switch_to_second_person(filtered_variables[i])

    with open(config_path.parent / "variables.json", "w") as f:
        json.dump(filtered_variables, f, indent=2)
