from abc import ABCMeta, abstractmethod
import typing


class BaseSampler(metaclass=ABCMeta):
    def __init__(
        self,
        model_name: str,
        state_dict_path: typing.Optional[str] = None,
        config_path: typing.Optional[str] = None,
    ):
        self.model_name = model_name
        self.state_dict_path = state_dict_path
        self.config_path = config_path

    @abstractmethod
    def send_prompt(self, prompt: str, n_probs: int, **kwargs) -> typing.Dict[str, int]:
        """
        Sends the given prompt to a LM.
        Arguments:
            prompt (str) a prompt to be sent to LM
            n_probs (int) number of desired output probalities.
        Return:
            dict (str:int) a dictionary of log probabilities of length n_probs
        """
        pass

    @abstractmethod
    def rank_completions(
        self, prompt: str, completions: typing.List[str]
    ) -> typing.Dict[str, float]:
        pass

    def get_best_next_token(self, prompt: str, **kwargs) -> str:
        """
        Generates a sequence of tokens from a prompt.
        Arguments:
            prompt (str) a prompt to be sent to LM
            n_probs (int) number of desired output probalities.
        Return:
            str a generated sequence
        """
        logprobs = self.send_prompt(prompt=prompt, n_probs=1, **kwargs)
        return list(logprobs.keys())[0]
