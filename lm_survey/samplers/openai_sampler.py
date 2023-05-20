import typing

import tiktoken
import torch

from lm_survey.samplers.base_sampler import BaseSampler
import openai

# model cost in cents per 1000 tokens (https://openai.com/pricing)
# tuple is (prompt cost per token, completion cost per token)
OPENAI_TOKEN_COSTS = {
    # chat
    "gpt-4": (3, 6),
    "gpt-3.5-turbo": (0.2, 0.2),
    # text
    "text-davinci-003": (2, 2),
    "text-davinci-002": (2, 2),
    "text-davinci-001": (2, 2),
    "text-curie-001": (0.2, 0.2),
    "text-babbage-001": (0.05, 0.05),
    "text-ada-001": (0.04, 0.04),
    "davinci": (2, 2),
    "curie": (0.2, 0.2),
    "babbage": (0.05, 0.05),
    "ada": (0.04, 0.04),
}


class OpenAiSampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "gpt3" in self.model_name:
            # engine is all text after 'gpt3-'
            self.engine = "-".join(self.model_name.split("-")[1:])
        else:
            self.engine = self.model_name

        print(f"Using {self.engine} engine.")

        if openai.api_key is None:
            raise ValueError("OpenAI API key must be set")

        self.tokenizer = None

    def rank_completions(self, prompt, completions):
        # 100 is the maximum number of log probs we can get.
        top_log_probs = self.send_prompt(prompt, n_probs=100)

        log_probs = torch.tensor(
            [top_log_probs.get(completion, -torch.inf) for completion in completions]
        )

        normalized_log_probs = torch.nn.functional.log_softmax(log_probs, dim=0)

        return {
            completion: normalized_log_prob.item()
            for completion, normalized_log_prob in zip(
                completions, normalized_log_probs
            )
        }

    def send_prompt(self, prompt, n_probs=100, **kwargs):
        try:
            response = openai.Completion.create(
                engine=self.engine,
                prompt=prompt,
                max_tokens=1,
                logprobs=n_probs,
                temperature=0,
                **kwargs,
            )
            logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]  # type: ignore
            # sort dictionary by values
            sorted_logprobs = dict(
                sorted(logprobs.items(), key=lambda x: x[1], reverse=True)
            )
            return sorted_logprobs
        except Exception as e:
            print(e)
            return {}

    def sample_several(self, prompt, temperature=0, n_tokens=10):
        response = openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            max_tokens=n_tokens,
            temperature=temperature,
        )
        return response["choices"][0]["text"]  # type: ignore

    def _setup_tokenizer(self):
        if not self.tokenizer:
            self.tokenizer = tiktoken.encoding_for_model(self.engine)

    def estimate_prompt_cost(self, prompt: str):
        self._setup_tokenizer()
        # +1 for single token completion
        prompt_token_count = len(self.tokenizer.encode(prompt))
        prompt_cost, completion_cost = OPENAI_TOKEN_COSTS[self.engine]
        return ((prompt_cost * prompt_token_count) + (completion_cost)) / 1000

    def batch_estimate_prompt_cost(
        self, prompts: typing.List[str]
    ) -> typing.List[float]:
        self._setup_tokenizer()
        # +1 for single token completion
        token_counts = [
            len(encoded) + 1 for encoded in self.tokenizer.encode_batch(prompts)
        ]
        prompt_cost, completion_cost = OPENAI_TOKEN_COSTS[self.engine]
        return [
            ((prompt_cost * token_count) + completion_cost) / 1000
            for token_count in token_counts
        ]


if __name__ == "__main__":
    sampler = OpenAiSampler("gpt3-ada")
    text = sampler.rank_completions(
        prompt="What is the capital of France?\nThe capital of France is",
        completions=[" Paris", " London", " Berlin"],
    )
    print(text)
