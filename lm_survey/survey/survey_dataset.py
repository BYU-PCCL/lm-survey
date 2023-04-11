import typing
from torch.utils.data import Dataset
from lm_survey.survey.dependent_variable_sample import DependentVariableSample
from lm_survey.survey.survey import Survey


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

    def __getitem__(self, idx) -> typing.Tuple[str, DependentVariableSample]:
        sample = self.data[idx]

        return sample.prompt, sample
