from lm_survey.survey.survey import Survey
import argparse
import os


if __name__ == "__main__":

    survey_directory = os.path.join("data", "ATP", "American_Trends_Panel_W92")

    survey = Survey(
        name="ATP_W92",
        data_filename=os.path.join(survey_directory, "responses.csv"),
        variables_filename=os.path.join(survey_directory, "config.json"),
    )
    survey.generate_atp_config(os.path.join(survey_directory, "config.json"))
