from samplers.base_sampler import BaseSampler

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel  # type: ignore


class GPT2Sampler(BaseSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        Supported models: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2'
        """
        print(f"Loading {self.model_name}...")

        self.model = GPT2LMHeadModel.from_pretrained(
            self.model_name, device_map="balanced"
        )
        self.model.eval()  # type: ignore

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.model_name, config=self.config_path
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.state_dict_path is not None:
            state_dict = torch.load(self.state_dict_path, map_location=self.device)
            self.model.load_state_dict(state_dict)  # type: ignore

        print(f"Using {torch.cuda.device_count()} GPUs.")

    def send_prompt(self, prompt, n_probs, **kwargs):
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output = self.model(**inputs)  # type: ignore

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
        tokens = self.model.generate(  # type: ignore
            **inputs,
            max_new_tokens=n_tokens,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        ).to(  # type: ignore
            "cpu"
        )
        preds = self.tokenizer.batch_decode(
            tokens, clean_up_tokenization_spaces=True, skip_special_tokens=True
        )
        return preds[0][len(prompt) + 1 :]


if __name__ == "__main__":
    sampler = GPT2Sampler("gpt2")
    text = sampler.get_best_next_token(
        prompt="What is the capital of France?\nThe capital of France is",
    )
    print(text)