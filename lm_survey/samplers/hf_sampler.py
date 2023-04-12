import typing
from lm_survey.samplers.base_sampler import BaseSampler

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class HfSampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print(f"Loading {self.model_name}...")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, device_map="balanced"
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.state_dict_path is not None:
            state_dict = torch.load(self.state_dict_path, map_location=self.device)
            self.model.load_state_dict(state_dict)

        print(f"Using {torch.cuda.device_count()} GPUs.")

    def rank_completions(
        self, prompts: typing.List[str], completions: typing.List[typing.List[str]]
    ) -> typing.List[typing.Dict[str, float]]:
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = [
            logits[attention_mask.bool()][-1].to("cpu")
            for logits, attention_mask in zip(outputs.logits, inputs.attention_mask)
        ]

        return [
            self._get_rankings(sample_completions, sample_logits)
            for sample_completions, sample_logits in zip(completions, logits)
        ]

    def _get_rankings(self, completions: typing.List[str], logits: torch.Tensor):
        completion_ids = self.tokenizer(
            completions,
            return_tensors="pt",
        ).input_ids

        completion_logits = torch.gather(logits, 0, completion_ids[:, -1])
        completion_log_probs = torch.nn.functional.log_softmax(completion_logits, dim=0)

        return {
            completion: log_prob.item()
            for completion, log_prob in zip(completions, completion_log_probs)
        }

    def send_prompt(
        self, prompt: str, n_probs: int, **kwargs
    ) -> typing.Dict[str, float]:
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)

        # get logits for final word (the prediction) from model output
        logits = output.logits[-1][-1].to("cpu")

        # get 'n_probs' predicted tokens associated with the above logits
        tokens = torch.argsort(logits, descending=True)[:n_probs]

        # decode tokens into text
        preds = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
        # Sometimes symbols don't come out great in ascii encoding
        preds = [p.encode("ascii", "ignore").decode("ascii") for p in preds]

        # calculate real probabilities associated with each prediction
        log_probs = torch.nn.functional.log_softmax(logits, dim=0)
        log_probs, _ = torch.sort(log_probs, descending=True)
        log_probs = log_probs[:n_probs]

        # create dictionary and map prediction word to log prob
        self.pred_dict = {}
        for pred, log_prob in zip(preds, log_probs):
            self.pred_dict[pred] = log_prob.item()

        return self.pred_dict

    def sample_several(self, prompt, temperature=0, n_tokens=10):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        ).to("cpu")
        preds = self.tokenizer.batch_decode(
            tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        return preds[0][len(prompt) + 1 :]


if __name__ == "__main__":
    sampler = HfSampler(model_name="/mnt/pccfs2/backed_up/models/llama/hf/llama-7b-hf")

    completions_dict = sampler.rank_completions(
        prompts=[
            "What is the capital of France?\n\nA) Paris\nB) London\nC) Berlin\nD) Rome\n\nAnswer:",
            "What is the capital of Germany?\n\nA) Paris\nB) Berlin\nC) Rome\n\nAnswer:",
        ],
        completions=[["A", "B", "C", "D"], ["A", "B", "C"]],
    )

    print(completions_dict)
