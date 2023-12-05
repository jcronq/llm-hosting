import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

def create_model_4bit(model_dir):
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

def create_model_8bit(model_dir):
    bnb = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        load_in_8bit=True,
        quantization_config=bnb,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    return pipe

common_kwarg_map = {
    "max_tokens": "max_new_tokens",
    "min_tokens": "min_new_tokens",
}

def normalize_kwargs(kwargs):
    for k, v in common_kwarg_map.items():
        if k in kwargs:
            kwargs[common_kwarg_map[v]] = kwargs.pop(k)
    return kwargs

def generate(pipe, text, **kwargs):
    kwargs = normalize_kwargs(kwargs)
    results = pipe(
        text,
        return_full_text=False,
        **kwargs,
    )
    return results[0]["generated_text"]
