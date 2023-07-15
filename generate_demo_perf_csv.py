from pathlib import Path
from typing import Dict
import pandas as pd
from tqdm import tqdm
from plot_slices import load_question_samples, atp_label_map
from analyze_results import analyze_results


def get_model_id(results_path: Path) -> str:
    return f"{results_path.parts[-5]}-{results_path.parent.name}"


def save_demo_df(results_path: Path, demo_df: pd.DataFrame, demographic: str):
    demographic_path = results_path.parent / "demographics" / f"{demographic}.csv"
    demographic_path.parent.mkdir(exist_ok=True, parents=True)
    demo_df.to_csv(demographic_path)


def create_summary_df(
    model_to_df: Dict[str, pd.DataFrame], column: str = "95%_lower_bound_gain"
) -> pd.DataFrame:
    return pd.concat(
        [df[column].rename(model_id) for model_id, df in model_to_df.items()], axis=1
    )


experiment_dir = Path("experiments")
model_dir = "llama-30b-hf/instruct"

keep_columns = [
    "mean",
    "std_error",
    "baseline",
    "95%_lower_bound_gain",
]

demographics = [
    "CREGION",
    "SEX",
    "EDUCATION",
    "CITIZEN",
    "MARITAL",
    "RELIG",
    "RELIGATTEND",
    "POLPARTY",
    "INCOME",
    "POLIDEOLOGY",
    "RACE",
]

experiment_paths = [
    experiment_dir / "guns" / "atp" / model_dir,
    experiment_dir / "immigration" / "atp" / model_dir,
    experiment_dir / "abortion" / "atp" / model_dir,
    # experiment_dir / "abortion" / "kaiser_family_foundation" / model_dir,
]

results_paths = [
    results_path
    for experiment_path in experiment_paths
    for results_path in experiment_path.rglob("results.json")
]

model_to_df = {}

loop = tqdm(total=len(results_paths) * len(demographics))

for results_path in results_paths:
    question_samples = load_question_samples(results_path)
    model_id = get_model_id(results_path)

    demo_dfs = []
    for demographic in demographics:
        demo_df = analyze_results(question_samples, [demographic])
        save_demo_df(results_path, demo_df, demographic)

        demo_dfs.append(demo_df[keep_columns])

        loop.update(1)

    model_to_df[model_id] = pd.concat(demo_dfs, axis=0)

summary_df = create_summary_df(model_to_df)
summary_df.to_csv("experiments/survey-summary.csv")
