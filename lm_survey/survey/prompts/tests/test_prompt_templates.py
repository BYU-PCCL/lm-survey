from lm_survey.survey.prompts.prompt_templates import (
    format_enumerated_iv_summary,
    format_interview_prompt,
)


def test_format_interview_prompt():
    questions = [
        {
            "question": "What is your name?",
            "choices": ["Chris", "Alex"],
            "answer": "A",
        },
        {
            "question": "What is your favorite color?",
            "choices": ["Red", "Green", "Blue"],
            "answer": "B",
        },
    ]
    expected_output = """
Question: What is your name?

A) Chris
B) Alex

Please respond with the letter of the answer choice that best fits your opinion.

Answer: A

Question: What is your favorite color?

A) Red
B) Green
C) Blue

Please respond with the letter of the answer choice that best fits your opinion.

Answer: B
    """.strip()
    assert format_interview_prompt(questions) == expected_output


def test_format_enumerated_iv_summary():
    independent_variables = {"age": "young", "gender": "male", "income": "low"}
    expected_output = """
Age: Young
Gender: Male
Income: Low
    """.strip()
    assert format_enumerated_iv_summary(independent_variables) == expected_output
