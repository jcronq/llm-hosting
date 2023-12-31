import time
import functools
import csv
import json

import torch
from torch.profiler import profile, record_function, ProfilerActivity

from local_llm.resources.lorem_ipsum import lorem_ipsum
from local_llm.utils import human_readable_memory, normalize_device_id


def timeit(func):
    """Wraps and returns a function that will time the given function."""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time() - start
        return result, end
    return wrapper

def warm_llm(generate):
    """ 
    The LLM tend to be really slow if this is the first request they receive.  
    So this function just acts as an initial query so subsequent profiling runs 
    show only inference runtimes
    """

    text = lorem_ipsum[:100]
    generate(text, max_new_tokens=10)


@functools.cache
def get_lorem_ids(tokenizer):
    """Returns the tokenized lorem ipsum text"""
    lorem_ids = tokenizer.encode(lorem_ipsum)
    return lorem_ids

def get_lorem_text(tokenizer, length):
    """Returns the text from lorem ipsum, that contains exactly length tokens"""
    lorem_ids = get_lorem_ids(tokenizer)
    input_ids = lorem_ids[:length]
    return tokenizer.decode(input_ids, skip_special_tokens=True)

def profile_llm_generation(generate, tokenizer, input_length, output_length):
    """Runs the torch profiler on an llm generation call"""
    timed_generate =  timeit(generate)

    text = get_lorem_text(tokenizer, input_length)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with record_function("model_inference"):
            timed_generate(text, max_new_tokens=output_length, min_new_tokens=output_length)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    prof.export_chrome_trace(f"trace_{input_length}_input.json")

def test_throughput(generate, tokenizer, input_lengths, output_lengths, output_file=None):
    """Tests an model for throughput latencies.  Creates a csv report."""
    timed_generate =  timeit(generate)

    warm_llm(generate)
    results = [("input_length", "output_length", "runtime")]
    try:
        for input_length in input_lengths:
            for output_length in output_lengths:
                text = get_lorem_text(tokenizer, input_length)
                _, runtime = timed_generate(text, max_new_tokens=output_length, min_new_tokens=output_length)
                result = (input_length, output_length, runtime)
                results.append(result)
                print(result)
    finally:
        if output_file is not None:
            save_test_results(results, output_file)

def test_batching(generate, tokenizer, input_length, output_length, batch_size):
    """Tests an model for throughput latencies.  Creates a csv report."""
    timed_generate =  timeit(generate)
    warm_llm(generate)
    text = get_lorem_text(tokenizer, input_length)
    batch = [text] * batch_size
    value, runtime = timed_generate(batch, max_new_tokens=output_length, min_new_tokens=output_length)
    result = (batch_size, input_length, output_length, runtime)
    print(result)
    return result

def save_test_results(results, filename):
    with open(filename, "w") as f:
        writer = csv.writer(f)
        writer.writerows(results)

def get_cuda_memory():
    device_profile = {"total": {"total": 0, "reserved": 0, "allocated": 0, "free": 0}}
    for i in range(torch.cuda.device_count()):
        memory_profile = {
            "total": torch.cuda.get_device_properties(i).total_memory,
            "reserved": torch.cuda.memory_reserved(i),
            "allocated": torch.cuda.memory_allocated(i),
        }
        memory_profile["free"] = memory_profile["total"] - memory_profile["reserved"] - memory_profile["allocated"]
        for k, v in memory_profile.items():
            device_profile["total"][k] += v
        device_profile[i] = memory_profile
    return device_profile