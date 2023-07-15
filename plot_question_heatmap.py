from pathlib import Path
from typing import Dict, List
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kendalltau

plt.style.use("ggplot")


def merge_question_rankings(question_rankings_list: List[Dict[str, float]]):
    merged_rankings = {
        k: v
        for question_rankings in question_rankings_list
        for k, v in question_rankings.items()
    }

    return dict(sorted(merged_rankings.items(), key=lambda item: item[1]))


experiment_dir = Path("experiments")

topics = [
    "guns",
    "abortion",
    "immigration",
]

topic_to_title = {
    "guns": "Gun Control",
    "abortion": "Abortion",
    "immigration": "Immigration",
}

topic_paths = [experiment_dir / topic / "atp" for topic in topics]

paths_per_topic = {
    topic: list(topic_path.rglob("question-rankings.json"))
    for topic, topic_path in zip(topics, topic_paths)
}

paths_per_topic["abortion"].extend(
    (experiment_dir / "abortion" / "kaiser_family_foundation").rglob(
        "question-rankings.json"
    )
)

rankings_per_topic = {}

for topic, paths in paths_per_topic.items():
    for path in paths:
        with path.open() as file:
            rankings = json.load(file)

        # rankings_per_topic.setdefault(topic, {})[path.parent.name] = rankings.keys()
        rankings_per_topic.setdefault(topic, {}).setdefault(
            path.parent.name, {}
        ).update(rankings)

# Sort each of the rankings and only keep the keys

for topic, rankings_per_ideology in rankings_per_topic.items():
    for ideology, rankings in rankings_per_ideology.items():
        rankings_per_topic[topic][ideology] = list(
            dict(sorted(rankings.items(), key=lambda item: item[1])).keys()
        )

plt.figure(figsize=(15, 5))
for i, (topic, rankings_per_topic) in enumerate(rankings_per_topic.items(), 1):
    plt.subplot(1, len(topics), i)
    rankings = np.array(list(rankings_per_topic.values()))

    taus = np.zeros((len(rankings), len(rankings)))

    for i, ranking1 in enumerate(rankings):
        for j, ranking2 in enumerate(rankings):
            taus[i, j] = kendalltau(ranking1, ranking2)[0]

    labels = [label.title() for label in rankings_per_topic.keys()]

    # plot the heatmap
    sns.heatmap(
        taus,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        # cmap="crest",
    )

    # Make the heatmap square
    plt.gca().set_aspect("equal")

    plt.title(topic_to_title[topic])

output_dir = Path("plots/question_heatmap.png")

plt.tight_layout()
plt.savefig(output_dir, dpi=200)
