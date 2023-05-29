from lm_survey.survey.results import SurveyResults
import numpy as np
import os
import json
from lm_survey.survey.dependent_variable_sample import DependentVariableSample
import glob
import pandas as pd
import sys


# Grab all the files called "results.json" in the "experiments" directory
input_filepaths = glob.glob(
    os.path.join("experiments/breadth", "**", "results.json"), recursive=True
)

print(input_filepaths)

# read input filepaths into pandas dfs
dfs = []
mean_reps = {}
for input_filepath in input_filepaths:
    with open(input_filepath, "r") as file:
        results = json.load(file)
    question_samples = [
        DependentVariableSample(
            **sample_dict,
        )
        for sample_dict in results
    ]

    survey_results = SurveyResults(question_samples=question_samples)
    wave = input_filepath.split("/")[3][-3:]
    # mean_reps[wave] = survey_results.get_representativeness()
    print(f"{wave}: {survey_results.calculate_avg_samples()}")

# print("Average representativeness: ", np.mean(list(mean_reps.values())))
# print(
#     "Average representativeness per : \n",
#     [f"{k}: {v}\n" for k, v in mean_reps.items()],
# )


# with open(input_filepath, "r") as file:
#     results = json.load(file)

# question_samples = [
#     DependentVariableSample(
#         **sample_dict,
#     )
#     for sample_dict in results["llama-7b-hf"]
# ]

# survey_results = SurveyResults(question_samples=question_samples)

# # Print with 2 decimal places
# print(survey_results.get_mean_score(slice_by=["gender"]).round(2))
