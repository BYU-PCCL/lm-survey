from abc import ABC, abstractmethod
import typing

import pandas as pd
from lm_survey.survey.variable import Variable
from lm_survey.survey.prompts.prompt_templates import (
    CONTEXT_AND_QUESTION_TEMPLATE,
    QUESTION_TEMPLATE,
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
            question=dependent_variable.to_natural_language(row),
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
        return "\n".join(
            [
                f"{variable.name.title()}: {variable.to_text(row).title()}"
                for variable in independent_variables
            ]
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
            question=dependent_variable.to_natural_language(row),
            choices=choices,
        )

        return CONTEXT_AND_QUESTION_TEMPLATE.format(
            context_summary="\n" + independent_variable_summary,
            dependent_variable_prompt=dependent_variable_prompt,
        )


class SecondPersonInterviewContextPrompt(BasePrompt):
    def _format_variable(self, variable: Variable, row: pd.Series):
        choices = format_multiple_choice_options(variable.to_options(row))

        return QUESTION_TEMPLATE.format(
            question=variable.to_question_text(row),
            choices=choices,
        )

    def format(
        self,
        row: pd.Series,
        dependent_variable: Variable,
        independent_variables: typing.List[Variable],
    ) -> str:
        sep = "\n\n###\n\n"
        preliminary_interview = sep.join(
            [
                f"{self._format_variable(variable, row)} {variable.get_correct_letter(row)}"
                for variable in independent_variables
            ]
        )

        return (
            preliminary_interview + sep + self._format_variable(dependent_variable, row)
        )
