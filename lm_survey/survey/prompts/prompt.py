from abc import ABC, abstractmethod
import typing

import pandas as pd
from lm_survey.survey.variable import Variable
from lm_survey.survey.prompts.prompt_templates import (
    CONTEXT_AND_QUESTION_TEMPLATE,
    MULTIPLE_CHOICE_QUESTION_TEMPLATE,
    OPEN_RESPONSE_QUESTION_TEMPLATE,
    format_multiple_choice_options,
    QUESTION_REFERENCING_CONTEXT_TEMPLATE,
)


class BasePrompt(ABC):
    @abstractmethod
    def format(
        self, independent_variables: typing.List[Variable], dependent_variable: Variable
    ) -> str:
        pass


class FirstPersonNaturalLanguageContextPrompt(BasePrompt):
    def _create_independent_variable_summary(
        self, independent_variables: typing.List[Variable], row: pd.Series
    ) -> str:
        return " ".join(
            [variable.to_natural_language(row) for variable in independent_variables]
        )

    def format(
        self,
        row: pd.Series,
        dependent_variable: Variable,
        independent_variables: typing.List[Variable],
    ) -> str:
        independent_variable_summary = self._create_independent_variable_summary(
            independent_variables=independent_variables, row=row
        )

        choices = format_multiple_choice_options(
            options=dependent_variable.to_options(row)
        )

        dependent_variable_prompt = QUESTION_REFERENCING_CONTEXT_TEMPLATE.format(
            question=dependent_variable.to_question_text(row),
            choices=choices,
        )

        return CONTEXT_AND_QUESTION_TEMPLATE.format(
            context_summary=independent_variable_summary,
            dependent_variable_prompt=dependent_variable_prompt,
        )


class SecondPersonEnumeratedContextPrompt(BasePrompt):
    def _create_independent_variable_summary(
        self, independent_variables: typing.List[Variable], row: pd.Series
    ) -> str:
        summary = "\n".join(
            [
                f"    {variable.name.lower()}: {variable.to_text(row).lower()},"
                for variable in independent_variables
            ]
        )

        return "{\n" + summary + "\n}"

    def format(
        self,
        row: pd.Series,
        dependent_variable: Variable,
        independent_variables: typing.List[Variable],
    ) -> str:
        independent_variable_summary = self._create_independent_variable_summary(
            independent_variables=independent_variables, row=row
        )

        choices = format_multiple_choice_options(
            options=dependent_variable.to_options(row)
        )

        dependent_variable_prompt = QUESTION_REFERENCING_CONTEXT_TEMPLATE.format(
            question=dependent_variable.to_question_text(row),
            choices=choices,
        )

        return CONTEXT_AND_QUESTION_TEMPLATE.format(
            context_summary=independent_variable_summary,
            dependent_variable_prompt=dependent_variable_prompt,
        )


class SecondPersonInterviewContextPrompt(BasePrompt):
    def _format_long_variable(
        self, variable: Variable, row: pd.Series, include_answer: bool = False
    ):
        variable_prompt = OPEN_RESPONSE_QUESTION_TEMPLATE.format(
            question=variable.to_question_text(row)
        )

        if include_answer:
            variable_prompt = variable_prompt + f" {variable.to_natural_language(row)}"

        return variable_prompt

    def _format_variable(
        self, variable: Variable, row: pd.Series, include_answer: bool = False
    ):
        options = variable.to_options(row)

        if len(options) > 10:
            return self._format_long_variable(
                variable=variable, row=row, include_answer=include_answer
            )

        choices = format_multiple_choice_options(options=options)

        variable_prompt = MULTIPLE_CHOICE_QUESTION_TEMPLATE.format(
            question=variable.to_question_text(row),
            choices=choices,
        )

        if include_answer:
            variable_prompt = variable_prompt + f" {variable.get_correct_letter(row)}"

        return variable_prompt

    def format(
        self,
        row: pd.Series,
        dependent_variable: Variable,
        independent_variables: typing.List[Variable],
    ) -> str:
        sep = "\n\n###\n\n"
        preliminary_interview = sep.join(
            [
                f"{self._format_variable(variable=variable, row=row, include_answer=True)}"
                for variable in independent_variables
            ]
        )

        return (
            preliminary_interview
            + sep
            + self._format_variable(
                variable=dependent_variable, row=row, include_answer=False
            )
        )
