curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "models/mistral-7b-instruct",
        "prompt": "[INST]Respond to all questions as a pirate.[/INST][INST]Who was San francisco?[/INST]",
        "max_tokens": 250,
        "temperature": 0
    }'
echo ""
