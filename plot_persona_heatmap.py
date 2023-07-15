from pathlib import Path
import numpy as np
import json

import matplotlib.pyplot as plt
import seaborn as sns
import argparse

plt.style.use("ggplot")


parser = argparse.ArgumentParser()

experiment_path = Path("experiments/persona")

personas = [
    "anti-immigration",
    "believes-in-gun-rights",
    "believes-abortion-should-be-illegal",
]

persona_paths = [experiment_path / persona for persona in personas]

paths_per_persona = {
    persona: persona_path.rglob("results.json")
    for persona, persona_path in zip(personas, persona_paths)
}

matches_per_persona = {}

for persona, paths in paths_per_persona.items():
    for path in paths:
        with path.open() as file:
            result = json.load(file)
            matches_per_persona.setdefault(persona, {})[path.parent.name] = result[
                "matches"
            ]

plt.figure(figsize=(15, 5))
for i, (persona, matches_per_ideology) in enumerate(matches_per_persona.items(), 1):
    plt.subplot(1, len(personas), i)
    matches = np.array(list(matches_per_ideology.values()))

    correlations = np.corrcoef(matches)

    labels = [label.title() for label in matches_per_ideology.keys()]

    # plot the heatmap
    sns.heatmap(
        correlations,
        annot=True,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        # cmap="coolwarm",
    )

    # Make the heatmap square
    plt.gca().set_aspect("equal")

    plt.title(persona)

output_dir = Path("plots/persona_heatmap.png")

plt.tight_layout()
plt.savefig(output_dir, dpi=200)
