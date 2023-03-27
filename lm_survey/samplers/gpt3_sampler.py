from samplers.base_sampler import BaseSampler
import openai


class GPT3Sampler(BaseSampler):
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
    sampler = GPT3Sampler("gpt3-davinci")
    text = sampler.get_best_next_token(
        prompt="What is the capital of France?\nThe capital of France is",
    )
    print(text)
