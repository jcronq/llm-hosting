import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

from typing import List

def create_model(model_dir):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        load_in_4bit=True,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return pipe


def generate(pipe, text, **kwargs):
    results = pipe(
        text,
        return_full_text=False,
        **kwargs,
    )
    return results[0]["generated_text"]

class Prompt:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def __call__(self, text, **kwargs):
        match(self.model_name):
            case "mistral-7b-instruct":
                return f"<s>[INST]{text}[/INST]"
            case _:
                raise ValueError(f"Unknown model name {self.model_name}")


# def generate(text, max_tokens: int=100, temperature=0.7, top_k=None, top_p=None):
#     sampling_args = {max_tokens: max_tokens, temperature: temperature}
#     if top_k is not None:
#         sampling_args["top_k"] = top_k
#     if top_p is not None:
#         sampling_args["top_p"] = top_p
#     sampling_params = SamplingParams(**sampling_args)
#     results = model.generate(text, sampling_params=sampling_params)
#     breakpoint()
#     for result in results:
#         yield result.outputs[0].text

