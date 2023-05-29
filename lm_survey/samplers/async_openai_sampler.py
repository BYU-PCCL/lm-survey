import sys

import openai
from openai.error import RateLimitError

import torch
from aiolimiter import AsyncLimiter

from lm_survey.samplers.base_sampler import BaseSampler
from lm_survey.samplers.openai_sampler import CHAT_MODELS

# Constants for throttling
OPENAI_RPM = 3000


class AsyncOpenAiSampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "async-gpt3" in self.model_name:
            # engine is all text after 'async-gpt3-'
            self.engine = "-".join(self.model_name.split("-")[2:])
        else:
            self.engine = self.model_name

        self._async_limiter = AsyncLimiter(OPENAI_RPM)

        print(f"Using async {self.engine} engine.")

        if openai.api_key is None:
            raise ValueError("OpenAI API key must be set")

        self.using_chat_model = self.engine in CHAT_MODELS

    async def rank_completions(self, prompt, completions):
        # 100 is the maximum number of log probs we can get.
        top_log_probs, response = await self.send_prompt(prompt, n_probs=100)

        log_probs = torch.tensor(
            [
                top_log_probs.get(completion, -torch.inf)
                for completion in completions
            ]
        )

        normalized_log_probs = torch.nn.functional.log_softmax(log_probs, dim=0)

        completion_log_probs = {
            completion: normalized_log_prob.item()
            for completion, normalized_log_prob in zip(
                completions, normalized_log_probs
            )
        }

        return completion_log_probs, response

    async def _throttled_completion(self, prompt, logprobs=None, **kwargs):
        while True:
            # We do this inside of the loop so that retries respect the rate limit too.
            async with self._async_limiter:
                try:
                    if self.using_chat_model:
                        return await openai.ChatCompletion.acreate(
                            model=self.engine,
                            messages=[{"role": "system", "content": prompt}],
                            **kwargs
                        )
                    return await openai.Completion.acreate(
                        model=self.engine, logprobs=logprobs, **kwargs
                    )
                except RateLimitError:
                    # TODO: This is not a good way to do logging; we should actually use
                    # the logging module or something similar.
                    if self.logger:
                        self.logger.exception("Rate limited, retrying...")
                    print("Rate limited, retrying...", file=sys.stderr)
                    continue

    async def send_prompt(self, prompt, n_probs=100, **kwargs):
        try:
            response = await self._throttled_completion(
                prompt=prompt,
                max_tokens=1,
                logprobs=n_probs,
                temperature=0,
                **kwargs,
            )
            if self.using_chat_model:
                token_response = response["choices"][0]["message"]["content"]  # type: ignore
                # To get exact matching with existing completion model code, we need to
                # add a space to the beginning of the token.
                sorted_logprobs = {f" {token_response}": 1}
            else:
                logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]  # type: ignore
                # sort dictionary by values
                sorted_logprobs = dict(
                    sorted(logprobs.items(), key=lambda x: x[1], reverse=True)
                )
            return sorted_logprobs, response
        except Exception as e:
            print(e)
            if self.logger:
                self.logger.exception(e)
            return {}, None

    async def sample_several(self, prompt, temperature=0, n_tokens=10):
        response = await self._throttled_completion(
            prompt=prompt,
            max_tokens=n_tokens,
            temperature=temperature,
        )
        return response["choices"][0]["text"], response  # type: ignore

    def estimate_prompt_cost(self, prompt: str, **kwargs) -> float:
        raise NotImplementedError
