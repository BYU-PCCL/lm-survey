import subprocess
import sys

sys.path.append(".")
from lm_survey.helpers import *
from pathlib import Path
import json
import shutil


WAVES = [26, 27]
EXPERIMENT_NAME = "test/test_overwrite_infill"


def generate_test_experiment(n_dvs: int = 3, n_instances: int = 6):
    for wave in WAVES:
        exp_dir = Path(f"experiments/breadth/ATP/American_Trends_Panel_W{wave}")

        results_path = exp_dir / "async-gpt3-text-davinci-003/results.json"
        results = filepath_to_DVS_list(results_path)

        config_path = exp_dir / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        n_dvs = 3
        n_instances = 6

        # First find the DVs
        dvs = list(set([r.variable_name for r in results]))[:n_dvs]

        config["dependent_variable_names"] = dvs

        full_results = []
        partial_results = []
        # Now we need 3 results for each DV that have
        for dv in dvs:
            dv_results = [r for r in results if r.variable_name == dv]
            has_ro = [r for r in dv_results if r.has_response()]
            missing_ro = [r for r in dv_results if not r.has_response()]
            if len(missing_ro) < n_instances / 2:
                print(
                    "Not enough null responses. Removing completions from DVS's"
                    " in has_ro"
                )
                missing_ro = [r.copy().remove_response() for r in has_ro][::-1]

            full_results.extend(has_ro[:n_instances])

            partial_results.extend(missing_ro[: int(n_instances / 2)])
            partial_results.extend(has_ro[: int(n_instances / 2)])

        exp_dest = Path(
            f"experiments/{EXPERIMENT_NAME}/ATP/American_Trends_Panel_W{wave}"
        )

        Path(
            f"experiments/{EXPERIMENT_NAME}/ATP/American_Trends_Panel_W{wave}/async-gpt3-text-davinci-003"
        ).mkdir(parents=True, exist_ok=True)

        full_results_path = exp_dest / Path(
            "async-gpt3-text-davinci-003/full_results.json"
        )
        DVS_list_to_json_file(full_results, full_results_path)

        partial_results_path = exp_dest / Path(
            f"async-gpt3-text-davinci-003/partial_results.json"
        )
        DVS_list_to_json_file(partial_results, partial_results_path)

        config_path = exp_dest / Path("config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)


def test_generate_experiment():
    n_dvs = 3
    n_instances = 6
    generate_test_experiment(n_dvs, n_instances)
    for wave in WAVES:
        exp_dir = Path(
            f"experiments/{EXPERIMENT_NAME}/ATP/American_Trends_Panel_W{wave}"
        )
        full_results_path = exp_dir / Path(
            "async-gpt3-text-davinci-003/full_results.json"
        )
        full_results = filepath_to_DVS_list(full_results_path)

        partial_results_path = exp_dir / Path(
            f"async-gpt3-text-davinci-003/partial_results.json"
        )
        partial_results = filepath_to_DVS_list(partial_results_path)

        assert (
            sum([r.has_response() for r in full_results]) == n_dvs * n_instances
        )
        assert (
            sum([r.has_response() for r in partial_results])
            == n_dvs * n_instances / 2
        )


def test_infill_full_results():
    for wave in WAVES:
        exp_dir = Path(
            f"experiments/{EXPERIMENT_NAME}/ATP/American_Trends_Panel_W{wave}"
        )
        full_results_path = exp_dir / Path(
            f"async-gpt3-text-davinci-003/full_results.json"
        )
        results_path = exp_dir / Path(
            f"async-gpt3-text-davinci-003/results.json"
        )
        shutil.copy(full_results_path, results_path)
    # Run infill
    subprocess.run(
        "python3 run_survey.py -m async-gpt3-text-davinci-003 -s all -e"
        " test/test_overwrite_infill -i".split()
    )
    for wave in WAVES:
        exp_dir = Path(
            f"experiments/{EXPERIMENT_NAME}/ATP/American_Trends_Panel_W{wave}"
        )
        full_results_path = exp_dir / Path(
            f"async-gpt3-text-davinci-003/full_results.json"
        )
        results_path = exp_dir / Path(
            f"async-gpt3-text-davinci-003/results.json"
        )
        full_results = filepath_to_DVS_list(full_results_path)
        results = filepath_to_DVS_list(results_path)
        ids_eq = [
            r.completion.response_object["id"]
            == fr.completion.response_object["id"]
            for r, fr in zip(results, full_results)
        ]
        assert sum(ids_eq) == len(ids_eq)
        results_path.unlink()


def test_infill_partial_results(n_dvs=3, n_instances=6):
    for wave in WAVES:
        exp_dir = Path(
            f"experiments/{EXPERIMENT_NAME}/ATP/American_Trends_Panel_W{wave}"
        )
        partial_results_path = exp_dir / Path(
            f"async-gpt3-text-davinci-003/partial_results.json"
        )
        results_path = exp_dir / Path(
            f"async-gpt3-text-davinci-003/results.json"
        )
        shutil.copy(partial_results_path, results_path)
    # Run infill
    subprocess.run(
        "python3 run_survey.py -m async-gpt3-text-davinci-003 -s all -e"
        " test/test_overwrite_infill -i".split()
    )
    for wave in WAVES:
        exp_dir = Path(
            f"experiments/{EXPERIMENT_NAME}/ATP/American_Trends_Panel_W{wave}"
        )
        partial_results_path = exp_dir / Path(
            f"async-gpt3-text-davinci-003/partial_results.json"
        )
        results_path = exp_dir / Path(
            f"async-gpt3-text-davinci-003/results.json"
        )
        partial_results = filepath_to_DVS_list(partial_results_path)
        results = filepath_to_DVS_list(results_path)

        missing_ros = [r.has_response() for r in partial_results]
        n_missing = sum(missing_ros)

        # Check that the right number of responses are missing for partial results
        assert n_missing == n_dvs * n_instances / 2

        # Check that all the DVSs with responses in the partial results
        # have same response in the full results
        unique_pr_ids = set(
            [
                r.completion.response_object["id"]
                if r.has_response()
                else "MISSING"
                for r in partial_results
            ]
        )
        unique_r_ids = set(
            [r.completion.response_object["id"] for r in results]
        )
        assert (
            len(unique_pr_ids.intersection(unique_r_ids))
            == n_dvs * n_instances / 2
        )
        results_path.unlink()


if __name__ == "__main__":
    # test_generate_experiment()
    test_infill_full_results()
    # test_infill_partial_results()
