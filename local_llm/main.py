import fire
from pathlib import Path

from local_llm.llm import create_model, generate, Prompt
from local_llm.profiling import test_throughput, profile_llm_generation

def interact(model_name, model_pipe):
    model_prompt = Prompt(model_name)
    max_tokens = 100
    temperature = 0.7
    while True:
        text = input("Enter text: ")
        if text.startswith("/"):
            match text[1:]:
                case "exit":
                    return
                case "quit":
                    return
                case "q":
                    return
                case "stop":
                    return
                case "temperature":
                    try:
                        temperature = float(text.split(" ")[1])
                    except ValueError:
                        print("Error: Invalid value passed to temperature.")
                        continue
                case "max_tokens":
                    try:
                        max_tokens = int(text.split(" ")[1])
                    except ValueError:
                        print("Error: Invalid value passed to max_tokens.")
                        continue
                case "break":
                    breakpoint()
                case _:
                    print("Unknown command")
            continue
        result = generate(
            model_pipe,
            model_prompt(text),
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            num_return_sequences=1,
            do_sample=True,
        ).replace("\\n", "\n")
        print(result)

def run_interact(model_dir="models/mistral-7b-instruct"):
    model_pipe = create_model(model_dir)
    interact(Path(model_dir).stem, model_pipe)

def run_throughput_test(which_test="input", model_dir="models/mistral-7b-instruct"):
    model_pipe = create_model(model_dir)
    if which_test == "input":
        input_lengths = range(100, 3000, 300)
        output_lengths = [1]
        results_file = "input_throughput.csv"
    elif which_test == "output":
        input_lengths = [100]
        output_lengths = range(100, 3000, 300)
        results_file = "output_throughput.csv"
    test_throughput(model_pipe, input_lengths, output_lengths, results_file)

def run_profile_test(model_dir="models/mistral-7b-instruct", input_length_tests=None, output_length=10):
    model_pipe = create_model(model_dir)
    if input_length_tests is None:
        input_length_tests = [100, 3000]
    for input_length in input_length_tests:
        profile_llm_generation(model_pipe, input_length, output_length)

def cli():
    fire.Fire({
        "profile-resources": run_profile_test,
        "profile-throughput": run_throughput_test, 
        "interact": run_interact,
    })

if __name__ == "__main__":
    run_test("models/mistral-7b-instruct")