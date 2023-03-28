import typing
import pandas as pd


class IndependentVariable:
    def __init__(
        self,
        name: str,
        keys: typing.List[str],
        invalid_options: typing.Dict[str, typing.List[str]],
        valid_options: typing.Dict[str, typing.Dict[str, str]],
    ):
        self.name = name
        self.keys = keys
        self.invalid_options = {
            key: set(options) for key, options in invalid_options.items()
        }
        self.valid_options = valid_options

    def _templatize(self, row: pd.Series) -> str:
        key = self._get_key(row)

        return self.valid_options[key][row[key]]

    def _get_key(self, row: pd.Series) -> str:
        for key in self.keys:
            if row[key].strip() not in self.invalid_options[key]:
                return key

        raise ValueError(
            f"This row has no key containing a valid value for the independent variable: {self.name})."
        )

    def to_sentence(self, row: pd.Series) -> str:
        return self._templatize(row)

    def to_option(self, row: pd.Series) -> str:
        key = self._get_key(row)
        return row[key]
