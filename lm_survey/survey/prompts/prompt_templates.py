import typing

from lm_survey.constants import MULTIPLE_CHOICE_LIST


CONTEXT_AND_QUESTION_TEMPLATE = """Context: {context_summary}

{dependent_variable_prompt}"""

QUESTION_REFERENCING_CONTEXT_TEMPLATE = """Question: {question}

{choices}

Please respond with the letter of the answer choice that best fits the context.

Answer:"""

MULTIPLE_CHOICE_QUESTION_TEMPLATE = """Question: {question}

{choices}

Please respond with the letter of the answer choice that best fits your opinion.

Answer:"""

OPEN_RESPONSE_QUESTION_TEMPLATE = """Question: {question}

Please respond with your answer.

Answer:"""


def format_enumerated_iv_summary(independent_variables: typing.Dict[str, str]) -> str:
    return "\n".join(
        [
            f"{variable_name.title()}: {variable_value.title()}"
            for variable_name, variable_value in independent_variables.items()
        ]
    )


def format_interview_prompt(
    questions: typing.List[typing.Dict[str, typing.List[typing.Union[str, None]]]],
) -> str:
    return "\n\n".join(
        [
            (
                MULTIPLE_CHOICE_QUESTION_TEMPLATE.format(
                    question=question["question"],
                    choices=format_multiple_choice_options(question["choices"]),
                )
                + f' {question["answer"]}'
            ).strip()
            for question in questions
        ]
    )


def format_multiple_choice_options(
    options: typing.Union[typing.List[str], typing.List[typing.Union[str, None]]]
) -> str:
    return "\n".join(
        [f"{MULTIPLE_CHOICE_LIST[i]}) {option}" for i, option in enumerate(options)]
    )


if __name__ == "__main__":
    independent_variables = {
        "gender": "male",
        "race": "white",
        "age": "18-29",
    }

    iv_summary = format_enumerated_iv_summary(independent_variables)

    questions = [
        {
            "question": "What is your opinion of the president?",
            "choices": [
                "Very favorable",
                "Somewhat favorable",
                "Somewhat unfavorable",
                "Very unfavorable",
                "Don't know",
            ],
            "answer": "A",
        }
    ]

    interview_prompt = format_interview_prompt(questions)

    print(interview_prompt)
