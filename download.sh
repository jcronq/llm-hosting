#!/bin/bash

curl https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/config.json -o model/config.json
curl https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/generation_config.json -o model/generate_config.json
curl https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/pytorch_model.bin.index.json -o model/pytorch_model.bin.index.json
curl https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/special_tokens_map.json -o model/special_tokens_map.json
curl https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer.json -o model/tokenizer.json
curl https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer_config.json -o model/tokenizer_config.json

# curl https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/tokenizer.model
# curl https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/pytorch_model-00001-of-00002.bin
# curl https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1/blob/main/pytorch_model-00002-of-00002.bin
