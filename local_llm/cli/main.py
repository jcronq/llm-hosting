import fire
from pathlib import Path
import functools

import torch

import local_llm.llm as llm
import local_llm.llm_remote as llm_remote
from local_llm.profiling import test_throughput, profile_llm_generation, test_batching, save_test_results
from local_llm.cli.interactive import interact
from local_llm.prompting import Prompt


def get_model_dir(model_name):
    return f"models/{model_name}"


def get_model_and_tokenizer(model_name):
    if model_name.startswith("http://"):
        generate = functools.partial(llm_remote.generate, model_name)
        tokenizer = llm.load_tokenizer("models/mistral-7b-instruct")
    else:
        model_dir = get_model_dir(model_name)
        model_pipeline = llm.create_model_4bit(model_dir)
        generate = functools.partial(llm.generate, model_pipeline)
        tokenizer = model_pipeline.tokenizer
    return generate, tokenizer


def run_interact(model_dir="mistral-7b-instruct", model_name="mistral-7b-instruct"):
    generate, _ = get_model_and_tokenizer(model_dir)
    # model_dir = get_model_dir(model_name)
    # model_pipeline = model_loader(model_dir)
    # # Bind model_pipeline to generate
    # generate = functools.partial(llm.generate, model_pipeline)
    # Get the Prmopt builder for the given model
    prompt_builder = Prompt(Path(model_name).stem)
    interact(generate, prompt_builder)


def serve_api(model_name="mistral-7b-instruct"):
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


def run_throughput_test(which_test="input", model_name="mistral-7b-instruct"):
    generate, tokenizer = get_model_and_tokenizer(model_name)
    if which_test == "input":
        input_lengths = range(100, 4000, 300)
        output_lengths = [1]
        results_file = "input_throughput.csv"
    elif which_test == "output":
        input_lengths = [100]
        output_lengths = range(100, 4000, 300)
        results_file = "output_throughput.csv"
    test_throughput(generate, tokenizer, input_lengths, output_lengths, results_file)


def run_profiler(model_name="mistral-7b-instruct"):
    generate, tokenizer = get_model_and_tokenizer(model_name)
    model, tokenizer = llm.load_model_inference(f"models/{model_name}")
    output_length = 1
    max_batch_size = 50
    batch_step_size = 2
    max_input_length = 2000
    input_step_size = 200
    results = [("batch_size", "input_length", "output_length", "runtime")]
    try:
        for input_length in range(input_step_size, max_input_length + 1, input_step_size):
            try:
                last_throughput = 0
                for batch_size in range(batch_step_size, max_batch_size + 1, batch_step_size):
                    model_pipeline = llm.create_pipeline(model, tokenizer, batch_size)
                    generate = functools.partial(llm.generate, model_pipeline)
                    result = test_batching(generate, tokenizer, input_length, output_length, batch_size)
                    input_token_throughput = input_length * batch_size / result[3]
                    output_token_throughput = batch_size / 0.057
                    print(f"Input token throughput: {input_token_throughput} tokens/sec")
                    print(f"Output token throughput: {output_token_throughput} tokens/sec")
                    delta_throughput = input_token_throughput - last_throughput
                    if delta_throughput < -500:
                        print(f"Throughput dropped by {delta_throughput} tokens/sec")
                        break
                    results.append(result)
                    last_throughput = input_token_throughput
            except torch.cuda.OutOfMemoryError:
                print("OOM: Proceeding with next experiment.")
                results.append((batch_size, input_length, output_length, -1))
            except RuntimeError as e:
                print(e)
                results.append((batch_size, input_length, output_length, -2))
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print(e)
                results.append((batch_size, input_length, output_length, -3))
    finally:

        columns = ["batch_size"]
        columns += [f"{idx}" for idx in range(input_step_size, max_input_length + 1, input_step_size)]
        new_format = tuple(columns)
        new_results = [new_format]
        tmp = {}
        for result in results[1:]:

            # input_length
            obj = tmp.setdefault(result[0], {})
            obj[result[1]] = result[3]
        for batch_size, result in tmp.items():

            tmp_result = [batch_size]
            for input_length in range(input_step_size, max_input_length + 1, input_step_size):

                tmp_result += [result.get(input_length, -4)]
            new_results.append(tuple(tmp_result))

        save_test_results(new_results, "batching_throughput.csv")


def run_batching_test(model_name="mistral-7b-instruct"):
    model, tokenizer = llm.load_model_inference(f"models/{model_name}")
    output_length = 1
    max_batch_size = 50
    batch_step_size = 2
    max_input_length = 2000
    input_step_size = 200
    results = [("batch_size", "input_length", "output_length", "runtime")]
    try:
        for input_length in range(input_step_size, max_input_length + 1, input_step_size):
            try:
                last_throughput = 0
                for batch_size in range(batch_step_size, max_batch_size + 1, batch_step_size):
                    model_pipeline = llm.create_pipeline(model, tokenizer, batch_size)
                    generate = functools.partial(llm.generate, model_pipeline)
                    result = test_batching(generate, tokenizer, input_length, output_length, batch_size)
                    input_token_throughput = input_length * batch_size / result[3]
                    output_token_throughput = batch_size / 0.057
                    print(f"Input token throughput: {input_token_throughput} tokens/sec")
                    print(f"Output token throughput: {output_token_throughput} tokens/sec")
                    delta_throughput = input_token_throughput - last_throughput
                    if delta_throughput < -500:
                        print(f"Throughput dropped by {delta_throughput} tokens/sec")
                        break
                    results.append(result)
                    last_throughput = input_token_throughput
            except torch.cuda.OutOfMemoryError:
                print("OOM: Proceeding with next experiment.")
                results.append((batch_size, input_length, output_length, -1))
            except RuntimeError as e:
                print(e)
                results.append((batch_size, input_length, output_length, -2))
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                print(e)
                results.append((batch_size, input_length, output_length, -3))
    finally:

        columns = ["batch_size"]
        columns += [f"{idx}" for idx in range(input_step_size, max_input_length + 1, input_step_size)]
        new_format = tuple(columns)
        new_results = [new_format]
        tmp = {}
        for result in results[1:]:

            # input_length
            obj = tmp.setdefault(result[0], {})
            obj[result[1]] = result[3]
        for batch_size, result in tmp.items():

            tmp_result = [batch_size]
            for input_length in range(input_step_size, max_input_length + 1, input_step_size):

                tmp_result += [result.get(input_length, -4)]
            new_results.append(tuple(tmp_result))

        save_test_results(new_results, "batching_throughput.csv")


def run_profile_test(model_name="mistral-7b-instruct", input_length_tests=None, output_length=10):
    generate, tokenizer = get_model_and_tokenizer(model_name)
    if input_length_tests is None:
        input_length_tests = [100, 3000]
    for input_length in input_length_tests:
        profile_llm_generation(generate, tokenizer, input_length, output_length)


def scratch(model_name="http://localhost:8000"):
    generate = functools.partial(llm_remote.generate, model_name)
    result = generate("Hi", max_tokens=20)
    print(result)


def cli():
    fire.Fire(
        {
            "profile-batching": run_batching_test,
            "profile-resources": run_profile_test,
            "profile-throughput": run_throughput_test,
            "interact": run_interact,
            "serve": serve_api,
            "scratch": scratch,
        }
    )


if __name__ == "__main__":
    run_throughput_test(model_dir="mistral-7b-instruct", model_loader=llm.create_model_4bit)
