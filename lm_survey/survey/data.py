import typing
from torch.utils.data import Dataset
from lm_survey.survey.dependent_variable_sample import DependentVariableSample
from lm_survey.survey.survey import Survey

SurveyItem = typing.Tuple[str, typing.List[str], DependentVariableSample]


class SurveyDataset(Dataset):
    def __init__(
        self,
        survey: Survey,
        n_samples_per_dependent_variable: typing.Optional[int] = None,
    ):
        self.data = list(
            survey.iterate(
                n_samples_per_dependent_variable=n_samples_per_dependent_variable
            )
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> typing.Union[SurveyItem, typing.List[SurveyItem]]:
        if isinstance(idx, slice):
            return [
                (sample.prompt, sample.completion.possible_completions, sample)
                for sample in self.data[idx]
            ]
        else:
            sample = self.data[idx]

            return (
                sample.prompt,
                sample.completion.possible_completions,
                sample,
            )


class SurveyDataLoader:
    """
    We don't inherit from torch.utils.data.DataLoader because we need more granular control
    over the batching.
    """

    def __init__(
        self,
        dataset: SurveyDataset,
        batch_size: int,
    ):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i : i + self.batch_size]
            prompts, completions, samples = zip(*batch)

            yield prompts, completions, samples
