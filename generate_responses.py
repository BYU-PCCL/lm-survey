import argparse
import json
from pathlib import Path
from typing import Tuple

from tqdm import tqdm

from lm_survey.samplers import AutoSampler


def to_prompt(text: str) -> str:
    return f"Question: {text}\n\nAnswer:"


def get_relevant_model_path_parts(model_path: Path) -> Tuple[str]:
    return (
        (model_path.name, "base")
        if "models" in model_path.parts
        else model_path.parts[-3:]
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
        "-q",
        "--questions",
        type=str,
        default="experiments/open-ended/abortion/questions.json",
    )

    args = parser.parse_args()

    sampler = AutoSampler(model_name=args.model_name)

    model_path = Path(args.model_name)

    question_path = Path(args.questions)
    with open(question_path) as file:
        questions = json.load(file)

    resonses = [
        {
            "question": question,
            "response": sampler.sample_several(
                to_prompt(question), temperature=0.7, n_tokens=300
            ),
        }
        for question in tqdm(questions)
    ]

    output_path = Path(
        question_path.parent,
        *get_relevant_model_path_parts(model_path),
        "responses.json",
    )

    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(
        output_path,
        "w",
    ) as file:
        json.dump(resonses, file, indent=2)
