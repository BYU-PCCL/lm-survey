import typing
from lm_survey.survey.prompts.prompt import (
    EnumeratedContextPrompt,
    NaturalLanguageContextPrompt,
    InterviewContextPrompt,
)
import pytest
import pandas as pd

from lm_survey.survey.variable import Variable


@pytest.fixture
def independent_variables():
    raw_variables = [
        {
            "name": "gender",
            "questions": [
                {
                    "key": "d16",
                    "text": "What is your gender? ",
                    "valid_options": [
                        {
                            "raw": "Male",
                            "text": "Male",
                            "natural_language": "I'm male.",
                        },
                        {
                            "raw": "Female",
                            "text": "Female",
                            "natural_language": "I'm female.",
                        },
                        {
                            "raw": "Other (Vol. for phone)",
                            "text": "Other (Vol. for phone)",
                            "natural_language": "I'm non-binary.",
                        },
                    ],
                    "invalid_options": [],
                }
            ],
        },
        {
            "name": "ethnicity",
            "questions": [
                {
                    "key": "d10",
                    "text": "Are you of Hispanic or Latino origin or descent?",
                    "valid_options": [
                        {
                            "raw": "Yes",
                            "text": "Yes",
                            "natural_language": "I'm Hispanic.",
                        }
                    ],
                    "invalid_options": ["Don't know", "No", "Refused/Web blank"],
                },
                {
                    "key": "d11",
                    "text": "Do you consider yourself white, black or African American, Asian, Native American, Pacific Islander, mixed race or some other race?",
                    "valid_options": [
                        {
                            "raw": "White",
                            "text": "White",
                            "natural_language": "I'm White.",
                        },
                        {
                            "raw": "Black or African American",
                            "text": "Black or African American",
                            "natural_language": "I'm African American.",
                        },
                        {
                            "raw": "Native American/American Indian/Alaska Native",
                            "text": "Native American/American Indian/Alaska Native",
                            "natural_language": "I'm Native American.",
                        },
                        {
                            "raw": "Asian/Chinese/Japanese",
                            "text": "Asian/Chinese/Japanese",
                            "natural_language": "I'm Asian.",
                        },
                        {
                            "raw": "Mixed",
                            "text": "Mixed",
                            "natural_language": "I'm Mixed ethnicity.",
                        },
                    ],
                    "invalid_options": ["Another race", "Refused/Web blank"],
                },
            ],
        },
    ]
    return [Variable(**raw_variable) for raw_variable in raw_variables]


@pytest.fixture
def dependent_variables():
    raw_variables = [
        {
            "name": "q24",
            "questions": [
                {
                    "key": "q24",
                    "text": "Do you (support) or (oppose) laws prohibiting abortions once cardiac activity, sometimes known as a fetal heartbeat, is detected?",
                    "valid_options": [
                        {"raw": "Oppose", "text": "Oppose", "natural_language": " "},
                        {"raw": "Support", "text": "Support", "natural_language": " "},
                    ],
                    "invalid_options": ["Don't know", "Refused/Web blank"],
                }
            ],
        },
        {
            "name": "q24a",
            "questions": [
                {
                    "key": "q24a",
                    "text": "What if you heard that cardiac activity is usually detectable around six weeks into pregnancy, which is before many women know they are pregnant. Do you (still support) or do you (now oppose) placing this restriction on women seeking abortions?",
                    "valid_options": [
                        {"raw": "Oppose", "text": "Oppose", "natural_language": " "},
                        {"raw": "Support", "text": "Support", "natural_language": " "},
                    ],
                    "invalid_options": [" ", "Don't know", "Refused/Web blank"],
                }
            ],
        },
        {
            "name": "q25",
            "questions": [
                {
                    "key": "q25",
                    "text": "Have you heard of emergency contraceptive pills, sometimes called morning after pills or \u201cPlan B\u201d, or is this not something you\u2019ve heard of?",
                    "valid_options": [
                        {
                            "raw": "Yes, have heard of it",
                            "text": "Yes, have heard of it",
                            "natural_language": " ",
                        },
                        {
                            "raw": "No, have not heard of it",
                            "text": "No, have not heard of it",
                            "natural_language": " ",
                        },
                    ],
                    "invalid_options": ["Refused/Web blank"],
                }
            ],
        },
    ]
    return [Variable(**raw_variable) for raw_variable in raw_variables]


@pytest.fixture
def row():
    return pd.Series(
        {
            "d16": "Male",
            "d10": "No",
            "d11": "White",
            "q24": "Support",
            "q24a": "Support",
            "q25": "Yes, have heard of it",
        }
    )


def test_first_person_natural_language_context_prompt_format(
    row: pd.Series,
    dependent_variables: Variable,
    independent_variables: typing.List[Variable],
) -> str:
    prompt = NaturalLanguageContextPrompt()

    formatted_prompt = prompt.format(
        row=row,
        dependent_variable=dependent_variables[0],
        independent_variables=independent_variables,
    )

    expected_prompt = """
Context: I'm male. I'm White.

Question: Do you (support) or (oppose) laws prohibiting abortions once cardiac activity, sometimes known as a fetal heartbeat, is detected?

A) Oppose
B) Support

Please respond with the letter of the answer choice that best fits the context.

Answer:
    """.strip()

    assert (
        formatted_prompt == expected_prompt
    ), f"Expected prompt to be {expected_prompt}, but got {formatted_prompt}"


def test_second_person_enumerated_context_prompt_format(
    row: pd.Series,
    dependent_variables: Variable,
    independent_variables: typing.List[Variable],
) -> str:
    prompt = EnumeratedContextPrompt()

    formatted_prompt = prompt.format(
        row=row,
        dependent_variable=dependent_variables[0],
        independent_variables=independent_variables,
    )

    print(formatted_prompt)

    expected_prompt = """
Context: {
    gender: male,
    ethnicity: white,
}

Question: Do you (support) or (oppose) laws prohibiting abortions once cardiac activity, sometimes known as a fetal heartbeat, is detected?

A) Oppose
B) Support

Please respond with the letter of the answer choice that best fits the context.

Answer:
    """.strip()

    assert (
        formatted_prompt == expected_prompt
    ), f"Expected prompt to be {expected_prompt}, but got {formatted_prompt}"


def test_second_person_interview_context_prompt_format(
    row: pd.Series,
    dependent_variables: Variable,
    independent_variables: typing.List[Variable],
) -> str:
    prompt = InterviewContextPrompt()

    formatted_prompt = prompt.format(
        row=row,
        dependent_variable=dependent_variables[0],
        independent_variables=independent_variables,
    )

    expected_prompt = """
Question: What is your gender? 

A) Male
B) Female
C) Other (Vol. for phone)

Please respond with the letter of the answer choice that best fits your opinion.

Answer: A

###

Question: Do you consider yourself white, black or African American, Asian, Native American, Pacific Islander, mixed race or some other race?

A) White
B) Black or African American
C) Native American/American Indian/Alaska Native
D) Asian/Chinese/Japanese
E) Mixed

Please respond with the letter of the answer choice that best fits your opinion.

Answer: A

###

Question: Do you (support) or (oppose) laws prohibiting abortions once cardiac activity, sometimes known as a fetal heartbeat, is detected?

A) Oppose
B) Support

Please respond with the letter of the answer choice that best fits your opinion.

Answer:
    """.strip()

    assert (
        formatted_prompt == expected_prompt
    ), f"Expected prompt to be {expected_prompt}, but got {formatted_prompt}"
