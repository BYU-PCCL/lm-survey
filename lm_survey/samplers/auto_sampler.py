from lm_survey.samplers.hf_sampler import HfSampler
from lm_survey.samplers.openai_sampler import OpenAiSampler
from lm_survey.samplers.async_openai_sampler import AsyncOpenAiSampler
from lm_survey.samplers.base_sampler import BaseSampler


class AutoSampler(BaseSampler):
    def __init__(self, model_name: str, *args, **kwargs):
        if model_name.startswith("gpt3") or model_name.startswith("gpt4"):
            self.sampler = OpenAiSampler(model_name, *args, **kwargs)
        elif model_name.startswith("async-gpt3") or model_name.startswith("async-gpt4"):
            self.sampler = AsyncOpenAiSampler(model_name, *args, **kwargs)
        else:
            self.sampler = HfSampler(model_name, *args, **kwargs)

    def rank_completions(self, prompt, completions):
        return self.sampler.rank_completions(prompt, completions)

    def send_prompt(self, prompt, n_probs):
        return self.sampler.send_prompt(prompt, n_probs)

    def sample_several(self, prompt, temperature=0, n_tokens=10):
        return self.sampler.sample_several(prompt, temperature, n_tokens)

    def estimate_prompt_cost(self, prompt: str) -> float:
        return self.sampler.estimate_prompt_cost(prompt)

    def __getattr__(self, attr):
        return getattr(self.sampler, attr)


if __name__ == "__main__":
    sampler = AutoSampler("gpt3-ada")
    text, response = sampler.get_best_next_token(
        prompt="What is the capital of France?\nThe capital of France is",
    )
    print(text)
