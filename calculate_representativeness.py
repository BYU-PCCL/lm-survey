from lm_survey.survey import SurveyResults
import numpy as np
from lm_survey.helpers import *
import os
import json
from lm_survey.survey import DependentVariableSample
import glob
import pandas as pd
import sys


# Grab all the files called "results.json" in the "experiments" directory
input_filepaths = Path("experiments/breadth").glob(
    "**/*davinci*/results.json",
)


# read input filepaths into pandas dfs
mean_reps = {}

question_samples = []
for input_filepath in list(input_filepaths):
    question_samples += filepath_to_dvs_list(input_filepath, add_weights=True)
    # wave = input_filepath.split("/")[3][-3:]
    # mean_reps[wave] = survey_results.get_representativeness()

    survey_results = SurveyResults(question_samples=question_samples)
    rep = survey_results.get_representativeness(
        survey_results.df.groupby("variable_name")
    )
    print(rep.mean())


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
