import torch
from lm_survey.samplers.base_sampler import BaseSampler
import openai


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


if __name__ == "__main__":
    sampler = OpenAiSampler("gpt3-ada")
    text = sampler.rank_completions(
        prompt="What is the capital of France?\nThe capital of France is",
        completions=[" Paris", " London", " Berlin"],
    )
    print(text)
