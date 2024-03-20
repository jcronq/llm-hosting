curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "teknium/OpenHermes-2.5-Mistral-7B",
        "prompt": "<|im_start|>assistant\nRespond to all questions as a pirate.<|im_end|>\n<|im_start|>user\nWho was San francisco?<|im_end|>",
        "max_tokens": 250,
        "temperature": 0
    }'
echo ""
