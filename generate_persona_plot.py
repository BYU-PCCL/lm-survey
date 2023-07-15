from pathlib import Path
from matplotlib import pyplot as plt
import json

plt.style.use("ggplot")

import pandas as pd

filepaths = Path("experiments/persona").rglob("results.json")

topic_to_persona = {
    "abortion": "believes-abortion-should-be-illegal",
    "guns": "believes-in-gun-rights",
    "immigration": "anti-immigration",
}

persona_to_topic = {v: k for k, v in topic_to_persona.items()}


results = {}
for filepath in filepaths:
    with open(filepath, "r") as f:
        persona_results = json.load(f)

    persona = filepath.parts[-5]

    if persona not in results:
        results[persona] = {
            "left": 0,
            "center": 0,
            "right": 0,
        }

    results[persona][filepath.parent.name] = persona_results["percent_matching"]

df = pd.DataFrame(results).T

# Make the index title case
df.columns = df.columns.str.title()

df.plot(kind="bar", figsize=(10, 4), alpha=0.7)

plt.xlabel("Persona")
plt.ylabel("Percent of Responses Matching Persona")

plt.xticks(rotation=0)

plt.tight_layout()
plt.legend()
plt.savefig("plots/persona_matching.png", dpi=200)
