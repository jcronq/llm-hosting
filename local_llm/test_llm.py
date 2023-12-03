from local_llm.llm import create_model, generate

def test_generate():
    print("test_generate")
    pipe = create_model("../mistral-7b-instruct")
    text = "Bonjour le monde"
    result = generate(pipe, text)
    assert isinstance(result, str)
    assert len(result) > 0
    input_ids = pipe.tokenizer.encode(text)
    output_ids = pipe.tokenizer.encode(result)
    print(result, input_ids, output_ids, len(output_ids))

if __name__ == "__main__":
    test_generate()