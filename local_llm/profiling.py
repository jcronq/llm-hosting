import time
import functools
import csv

from torch.profiler import profile, record_function, ProfilerActivity

from local_llm.llm import create_model, generate
from local_llm.resources.lorem_ipsum import lorem_ipsum


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() - start
        return result, end
    return wrapper

def warm_llm(model_pipe):
    text = lorem_ipsum[:100]
    generate(model_pipe, text, max_new_tokens=10)


@functools.cache
def get_lorem_ids(tokenizer):
    lorem_ids = tokenizer.encode(lorem_ipsum)
    return lorem_ids

def get_lorem_text(tokenizer, length):
    lorem_ids = get_lorem_ids(tokenizer)
    input_ids = lorem_ids[:length]
    return tokenizer.decode(input_ids, skip_special_tokens=True)

def profile_llm_generation(model_pipe, input_length, output_length):
    timed_generate =  timeit(functools.partial(generate, model_pipe))

    text = get_lorem_text(model_pipe.tokenizer, input_length)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            timed_generate(text, max_new_tokens=output_length, min_new_tokens=output_length)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace(f"trace_{input_length}_input.json")

def test_throughput(model_pipe, input_lengths, output_lengths, output_file=None):
    timed_generate =  timeit(functools.partial(generate, model_pipe))

    warm_llm(model_pipe)
    results = [("input_length", "output_length", "runtime")]
    try:
        for input_length in input_lengths:
            for output_length in output_lengths:
                text = get_lorem_text(model_pipe.tokenizer, input_length)
                _, runtime = timed_generate(text, max_new_tokens=output_length, min_new_tokens=output_length)
                result = (input_length, output_length, runtime)
                results.append(result)
                print(result)
    finally:
        if output_file is not None:
            save_test_results(results, output_file)

def save_test_results(results, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)