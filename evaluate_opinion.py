import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import argparse
from datasets import load_dataset

from lm_survey.samplers import AutoSampler


def templatize(text: str) -> str:
    return f'Question: Yes or no, is the following statement something you would say: "{text}"?.\n\nAnswer:'


def evaluate_opinion(
    model_name: Path,
    persona: Path,
):
    sampler = AutoSampler(model_name=model_name.as_posix())
    dataset = load_dataset(
        "json",
        data_files=persona.as_posix(),
        split="train",
    )

    match_count = 0
    total_count = 0

    matches = []

    for example in tqdm(dataset):
        prompt = templatize(example["statement"])  # type: ignore

        completion_log_probs = sampler.rank_completions(
            prompt, completions=["Yes", "No"]
        )

        match_key = example["answer_matching_behavior"].strip()  # type: ignore
        non_match_key = example["answer_not_matching_behavior"].strip()  # type: ignore

        matching_log_prob = completion_log_probs.get(match_key, -np.inf)
        non_matching_log_prob = completion_log_probs.get(non_match_key, -np.inf)

        if matching_log_prob > non_matching_log_prob:
            matches.append(1)
        elif matching_log_prob < non_matching_log_prob:
            matches.append(0)
        else:
            matches.append(0.5)

    matches = np.array(matches)

    # Print as percent
    print(f"Percent Matching: {(matches.mean()):.3%}")

    output_dir = Path("experiments", "persona", persona.stem, *model_name.parts[-3:])
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "results.json", "w") as file:
        json.dump(
            {
                "percent_matching": matches.mean(),
                "matches": matches.tolist(),
            },
            file,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-p",
        "--persona",
        type=str,
        default="data/personas/believes-abortion-should-be-illegal.jsonl",
    )

    args = parser.parse_args()

    evaluate_opinion(model_name=Path(args.model_name), persona=Path(args.persona))
