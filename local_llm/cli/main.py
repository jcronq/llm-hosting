import fire
from pathlib import Path
import functools

import torch

import local_llm.llm as llm
from local_llm.profiling import test_throughput, profile_llm_generation, test_batching, save_test_results
from local_llm.cli.interactive import interact
from local_llm.prompting import Prompt

def get_model_dir(model_name):
    return f"models/{model_name}"

def run_interact(model_name="mistral-7b-instruct", model_loader=llm.create_model_4bit):
    model_dir = get_model_dir(model_name)
    model_pipeline = model_loader(model_dir)
    # Bind model_pipeline to generate
    generate = functools.partial(llm.generate, model_pipeline)
    # Get the Prmopt builder for the given model
    prompt_builder = Prompt(Path(model_dir).stem)
    interact(generate, prompt_builder)

def serve_api(model_name="mistral-7b-instruct", model_loader=llm.create_model_4bit):
    model_dir = get_model_dir(model_name)
    import uvicorn
    from local_llm.rest_api.openai.main import app, init_model, TIMEOUT_KEEP_ALIVE
    init_model(model_dir)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )

def run_throughput_test(which_test="input", model_name="mistral-7b-instruct", model_loader=llm.create_model_4bit):
    model_dir = get_model_dir(model_name)
    model_pipeline = model_loader(model_dir)
    if which_test == "input":
        input_lengths = range(100, 4000, 300)
        output_lengths = [1]
        results_file = "input_throughput.csv"
    elif which_test == "output":
        input_lengths = [100]
        output_lengths = range(100, 4000, 300)
        results_file = "output_throughput.csv"
    generate = functools.partial(llm.generate, model_pipeline)
    test_throughput(generate, model_pipeline.tokenizer, input_lengths, output_lengths, results_file)

def run_batching_test(max_batch_size: int, model_name="mistral-7b-instruct"):
    output_length = 1
    model_dir = get_model_dir(model_name)
    model, tokenizer = llm.load_model_inference(model_dir)
    results = [("batch_size", "input_length", "output_length", "runtime")]
    try:
        for input_length  in range(10, 100+1, 10):
            try:
                for batch_size in range(10, max_batch_size+1, 10):
                    model_pipeline = llm.create_pipeline(model, tokenizer, batch_size)
                    generate = functools.partial(llm.generate, model_pipeline)
                    results.append(test_batching(generate, model_pipeline.tokenizer, input_length, output_length, batch_size))
            except torch.cuda.OutOfMemoryError:
                print("OOM: Proceeding with next experiment.")
                results.append((batch_size, input_length, output_length, -1))
    finally:
        save_test_results(results, "batching_throughput.csv")
    

def run_profile_test(model_name="mistral-7b-instruct", input_length_tests=None, output_length=10, model_loader=llm.create_model_4bit):
    model_dir = get_model_dir(model_name)
    model_pipeline = model_loader(model_dir)
    if input_length_tests is None:
        input_length_tests = [100, 3000]
    for input_length in input_length_tests:
        generate = functools.partial(llm.generate, model_pipeline)
        profile_llm_generation(generate, model_pipeline.tokenizer, input_length, output_length)

def cli():
    fire.Fire({
        "profile-batching": run_batching_test,
        "profile-resources": run_profile_test,
        "profile-throughput": run_throughput_test, 
        "interact": run_interact,
        "serve": serve_api,
    })

if __name__ == "__main__":
    run_throughput_test(model_dir="mistral-7b-instruct", model_loader=llm.create_model_4bit)