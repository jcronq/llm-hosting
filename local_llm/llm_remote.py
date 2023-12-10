import json
import requests

def generate(endpoint: str, prompt: str, **kwargs):
    if "max_new_tokens" in kwargs:
        kwargs["max_tokens"] = kwargs.pop("max_new_tokens")
    if "min_new_tokens" in kwargs:
        kwargs.pop("min_new_tokens")
        kwargs["ignore_eos"] = True
    response = requests.post(f"{endpoint}/generate", data=json.dumps({"prompt": prompt, **kwargs}))
    return response.json()["text"][0][len(prompt):]