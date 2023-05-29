import typing

from lm_survey.constants import MULTIPLE_CHOICE_LIST

DEPENDENT_VARIABLE_TEMPLATE = """Question: {question}

{choices}"""

COMPLETION_TEMPLATE = """Context: {context_summary}

{dependent_variable_prompt}

Please respond with the letter of the answer choice that best fits the context.

Answer:"""

CHAT_SYSTEM_TEMPLATE = """
As a large language model, your training data contains information about online \
conversation between individuals with various demographics. In this example, you are \
asked to answer a survey question given a demographic description. You may elaborate \
on your answer,  but you must start your response with the letter corresponding the \
correct response. An example: "Z, because..."

'''

Context: {context_summary}

{dependent_variable_prompt}

Please respond with the letter of the answer choice that best fits the context.

Answer:
"""


def format_multiple_choice_options(
    options: typing.List[typing.Union[str, None]]
) -> str:
    return "\n".join(
        [f"{MULTIPLE_CHOICE_LIST[i]}) {option}" for i, option in enumerate(options)]
    )
