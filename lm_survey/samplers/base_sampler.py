from abc import ABCMeta, abstractmethod
import typing
import logging

T = typing.TypeVar("T")
MaybeAwaitable = typing.Union[T, typing.Awaitable[T]]


class BaseSampler(metaclass=ABCMeta):
    def __init__(
        self,
        model_name: str,
        state_dict_path: typing.Optional[str] = None,
        config_path: typing.Optional[str] = None,
        logger: typing.Optional[logging.Logger] = None,
    ):
        self.model_name = model_name
        self.state_dict_path = state_dict_path
        self.config_path = config_path
        self.logger = logger

    @abstractmethod
    def send_prompt(
        self, prompt: str, n_probs: int, **kwargs
    ) -> MaybeAwaitable[typing.Tuple[typing.Dict[str, int], typing.Any]]:
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
    ) -> MaybeAwaitable[typing.Tuple[typing.Dict[str, float], typing.Any]]:
        pass

    def get_best_next_token(
        self, prompt: str, **kwargs
    ) -> MaybeAwaitable[typing.Tuple[str, typing.Any]]:
        """
        Generates a sequence of tokens from a prompt.
        Arguments:
            prompt (str) a prompt to be sent to LM
            n_probs (int) number of desired output probalities.
        Return:
            str a generated sequence
        """
        # TODO(vinhowe): This is an AWFUL way to do this and it is SO FRAGILE
        if not self.model_name.startswith("async"):
            logprobs, response = self.send_prompt(
                prompt=prompt, n_probs=1, **kwargs
            )
            return list(logprobs.keys())[0], response

        async def _get_best_next_token():
            logprobs, response = await self.send_prompt(
                prompt=prompt, n_probs=1, **kwargs
            )
            return list(logprobs.keys())[0], response

        return _get_best_next_token()

    @abstractmethod
    def estimate_prompt_cost(
        self, prompt: str, **kwargs
    ) -> MaybeAwaitable[float]:
        """
        Estimates the cost of sending the given prompt to a LM.
        Arguments:
            prompt (str) a prompt to be sent to LM
        Return:
            float the estimated cost of sending the prompt in USD cents
        """
        pass
