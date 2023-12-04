from local_llm.llm import create_model_4bit, generate

def test_generate():
    print("test_generate")
    model_pipeline = create_model_4bit("../mistral-7b-instruct")
    text = "Bonjour le monde"
    result = generate(model_pipeline, text)
    assert isinstance(result, str)
    assert len(result) > 0
    input_ids = model_pipeline.tokenizer.encode(text)
    output_ids = model_pipeline.tokenizer.encode(result)
    print(result, input_ids, output_ids, len(output_ids))

if __name__ == "__main__":
    test_generate()