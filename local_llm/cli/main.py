import fire
from pathlib import Path
import functools

import torch

import local_llm.llm as llm
from local_llm.profiling import test_throughput, profile_llm_generation
from local_llm.cli.interactive import interact
from local_llm.prompting import Prompt

def run_interact(model_dir="models/mistral-7b-instruct", model_loader=llm.create_model_4bit):
    model_pipeline = model_loader(model_dir)
    # Bind model_pipeline to generate
    generate = functools.partial(llm.generate, model_pipeline)
    # Get the Prmopt builder for the given model
    prompt_builder = Prompt(Path(model_dir).stem)
    interact(generate, prompt_builder)

def run_throughput_test(which_test="input", model_dir="models/mistral-7b-instruct", model_loader=llm.create_model_4bit):
    model_pipeline = model_loader(model_dir)
    if which_test == "input":
        input_lengths = range(100, 4000, 300)
        output_lengths = [1]
        results_file = "input_throughput.csv"
    elif which_test == "output":
        input_lengths = [100]
        output_lengths = range(100, 4000, 300)
        results_file = "output_throughput.csv"
    test_throughput(model_pipeline, input_lengths, output_lengths, results_file)

def run_profile_test(model_dir="models/mistral-7b-instruct", input_length_tests=None, output_length=10, model_loader=llm.create_model_4bit):
    model_pipeline = model_loader(model_dir)
    if input_length_tests is None:
        input_length_tests = [100, 3000]
    for input_length in input_length_tests:
        profile_llm_generation(model_pipeline, input_length, output_length)

def cli():
    fire.Fire({
        "profile-resources": run_profile_test,
        "profile-throughput": run_throughput_test, 
        "interact": run_interact,
    })

if __name__ == "__main__":
    run_throughput_test(model_dir="models/mistral-7b-instruct", model_loader=llm.create_model_4bit)