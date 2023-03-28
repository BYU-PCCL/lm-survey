import json
import typing
import pandas as pd
import pandas.core.groupby.generic
from lm_survey.survey import DependentVariableSample


class SurveyResults:
    def __init__(
        self, dependent_variable_samples: typing.List[DependentVariableSample]
    ):
        df = pd.DataFrame(
            data=[
                self._flatten_dict(sample.to_dict())
                for sample in dependent_variable_samples
            ]
        )
        # Make the index the df_index column and sort by it
        df.set_index("df_index", inplace=True)
        self.df = df.sort_index()

    def _flatten_dict(self, input_dict: dict) -> dict:
        output_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, dict):
                output_dict.update(self._flatten_dict(value))
            else:
                output_dict[key] = value
        return output_dict

    def slice(
        self, columns: typing.List[str]
    ) -> pandas.core.groupby.generic.DataFrameGroupBy:
        return self.df.groupby(columns)

    def _compute_mean(
        self,
        groups: pandas.core.groupby.generic.DataFrameGroupBy,
    ) -> pd.Series:
        return groups.is_completion_correct.mean()

    def _bootstrap_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sample(frac=1, replace=True)  # type: ignore

    def _bootstrap(
        self,
        groups: pandas.core.groupby.generic.DataFrameGroupBy,
    ) -> pd.Series:
        sampled_groups = groups.apply(self._bootstrap_sample).groupby(level=0)
        return self._compute_mean(sampled_groups)

    def _estimate_standard_error(
        self,
        groups: pandas.core.groupby.generic.DataFrameGroupBy,
        n_bootstraps: int = 1000,
    ) -> pd.Series:
        bootstraps = pd.concat([self._bootstrap(groups) for _ in range(n_bootstraps)])

        return bootstraps.groupby(level=0).std()

    def get_mean_score(self, slice_by: typing.List[str]) -> pd.DataFrame:
        groups = self.slice(columns=slice_by)

        means = self._compute_mean(groups).rename("mean")
        errors = self._estimate_standard_error(groups).rename("std_error")
        counts = groups.count().is_completion_correct.rename("n_samples")

        scores_df = pd.concat([means, errors, counts], axis=1)

        return scores_df


if __name__ == "__main__":
    input_filepath = "data/roper/results.json"

    with open(input_filepath, "r") as file:
        results = json.load(file)

    dependent_variable_samples = [
        DependentVariableSample(
            **sample_dict,
        )
        for sample_dict in results["gpt3-text-davinci-003"]
    ]

    survey_results = SurveyResults(
        dependent_variable_samples=dependent_variable_samples
    )

    print(survey_results.get_mean_score(slice_by=["gender"]))
