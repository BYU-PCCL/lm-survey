import json
import os
import typing
import pandas as pd
import pandas.core.groupby.generic
from lm_survey.survey.dependent_variable_sample import DependentVariableSample


class SurveyResults:
    def __init__(self, question_samples: typing.List[DependentVariableSample]):
        df = pd.DataFrame(
            data=[self._flatten_layer(sample.to_dict()) for sample in question_samples]
        )
        # Make the index the index column and sort by it
        df.set_index("index", inplace=True)

        self.df = df.sort_index()
        self._engineer_columns()

    def _engineer_columns(self):
        self.df["random_guess_prob"] = self.df["possible_completions"].apply(
            lambda x: 1 / len(x)
        )

    def _flatten_dict(self, input_dict: dict) -> dict:
        """Flattens the entire dict so it is only 1 layer deep."""
        output_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, dict):
                output_dict.update(self._flatten_dict(value))
            else:
                output_dict[key] = value
        return output_dict

    def _flatten_layer(self, input_dict: dict) -> dict:
        """Only flattens the top layer of the dict."""
        output_dict = {}
        for key, value in input_dict.items():
            if isinstance(value, dict):
                output_dict.update(value)
            else:
                output_dict[key] = value
        return output_dict

    def _calculate_random_guess(
        self, groups: pandas.core.groupby.generic.DataFrameGroupBy
    ) -> pd.Series:
        return groups.random_guess_prob.mean()

    def slice(
        self, columns: typing.List[str]
    ) -> pandas.core.groupby.generic.DataFrameGroupBy:
        if columns == []:
            return self.df.groupby(lambda _: "all")
        else:
            return self.df.groupby(by=columns)

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

    def get_mean_score(self, slice_by: typing.List[str] = []) -> pd.DataFrame:
        groups = self.slice(columns=slice_by)

        means = self._compute_mean(groups)
        errors = self._estimate_standard_error(groups)
        baselines = self._calculate_random_guess(groups)
        n_samples = groups.size()
        improvement_lower_bounds = means - baselines - errors * 1.96

        scores_df = pd.concat(
            [means, errors, baselines, improvement_lower_bounds, n_samples], axis=1
        )
        scores_df.columns = [
            "mean",
            "std_error",
            "baseline",
            "95%_lower_bound_gain",
            "n_samples",
        ]

        return scores_df


if __name__ == "__main__":
    experiment_dir = "experiments/default/roper"
    center_path = os.path.join(experiment_dir, "llama-65b-hf","results.json")
    left_path = os.path.join(experiment_dir, "llama-65b-hf-left","results.json")

    with open(center_path, "r") as file:
        center_results = json.load(file)

    with open(left_path, "r") as file:
        left_results = json.load(file)

    center_question_samples = [
        DependentVariableSample(
            **sample_dict,
        )
        for sample_dict in center_results
    ]

    left_question_samples = [
        DependentVariableSample(
            **sample_dict,
        )
        for sample_dict in left_results
    ]

    center_survey_results = SurveyResults(question_samples=center_question_samples)
    left_survey_results = SurveyResults(question_samples=left_question_samples)

    slices = ["ethnicity"]

    print("CENTER", center_survey_results.get_mean_score(slice_by=slices).round(2), sep="\n", end="\n\n")
    print("LEFT", left_survey_results.get_mean_score(slice_by=slices).round(2), sep="\n", end="\n\n")

    # Give battery of standard tests and view performance
