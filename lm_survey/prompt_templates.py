import typing

from lm_survey.constants import MULTIPLE_CHOICE_LIST


INDEPENDENT_VARIABLE_SUMMARY_TEMPLATE = """Context: {context_summary}

{dependent_variable_prompt}"""

DEPENDENT_VARIABLE_TEMPLATE = """Question: {question}

{choices}

Please respond with the letter of the answer choice that best fits the context.

Answer:"""


def format_multiple_choice_options(options: typing.List[str]) -> str:
    return "\n".join(
        [f"{MULTIPLE_CHOICE_LIST[i]}) {option}" for i, option in enumerate(options)]
    )
