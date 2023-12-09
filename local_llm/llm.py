import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)

def load_model_inference(model_dir: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb,
    )
    tokenizer.pad_token_id = model.config.eos_token_id

    if "llama" in model_dir:
        model.to_bettertransformer()
    return model, tokenizer

def create_pipeline(model, tokenizer, batch_size: int=1):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    return pipe


def create_model_4bit(model_dir, batch_size: int=1):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb,
    )
    tokenizer.pad_token_id = model.config.eos_token_id

    if "llama" in model_dir:
        model.to_bettertransformer()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

    return pipe

def create_model_8bit(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        load_in_8bit=True,
        device_map="auto",
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
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
    is_batched = False
    if isinstance(text, list):
        is_batched = True
    kwargs = normalize_kwargs(kwargs)
    results = pipe(
        text,
        return_full_text=False,
        **kwargs,
    )
    if is_batched:
        return [result[0]["generated_text"] for result in results]
    return results[0]["generated_text"]
