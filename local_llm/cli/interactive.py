from local_llm.profiling import get_cuda_memory
from local_llm.utils import sprint_byte_object


attribute_type_map = {
    "temperature": float,
    "max_tokens": int,
}

class GotoBreakpoint(Exception):
    pass

class QuitException(Exception):
    pass

def interactive_function(user_input, kwargs):
    
    match user_input.split(" "):
        case ["exit" | "quit" | "q" | "stop"]:
            raise QuitException()
        case [""]:
            pass
        case ["free" | "memory" | "mem" | "cuda" | "gpu" | "print_memory" | "print_mem" | "print_cuda" | "print_gpu" | "print_memory_usage" | "print_mem_usage" | "print_cuda_usage" | "print_gpu_usage" | "print_memory_profile" | "print_mem_profile" | "print_cuda_profile" | "print_gpu_profile" | "print_memory_info" | "print_mem_info" | "print_cuda_info" | "print_gpu_info"]:
            print(sprint_byte_object(get_cuda_memory()))
        case ["break"]:
            raise GotoBreakpoint()
        case ["print" | "p" | "args" | "kwargs" | "print_args" | "print_kwargs" | "printargs" | "printkwargs"]:
            print(kwargs)
        case ["help"]:
            print("Available commands:")
            print("  /help")
            print("  /set <attribute> <value>")
            print("  /break")
            print("  /exit")
            print("  /quit")
            print("  /q")
            print("  /stop")
        case ["set", attribute, value_str]:
            if attribute in attribute_type_map:
                try:
                    value = attribute_type_map[attribute](value_str) # Cast the value to it's corresponding type
                    kwargs[attribute] = value
                except ValueError:
                    print(f"Error: Invalid value passed to {attribute}.\n  Expected {attribute_type_map[attribute]}.\n  Received {value_str}")
            else:
                print(f"Error: Unknown attribute {attribute}")
        case ["break"]:
            raise GotoBreakpoint()
        case _:
            print(f"Error: Unknown command '{user_input}'")
    return kwargs



def interact(generate, prompt_builder):
    kwargs = {
        "max_tokens": 100,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 1,
        "do_sample": True,
    }
    while True:
        text = input("Enter text: ")
        if text.startswith("/"):
            try:
                kwargs = interactive_function(text[1:], kwargs)
                continue
            except GotoBreakpoint:
                breakpoint()
            except QuitException:
                break
        result = generate(prompt_builder(text), **kwargs).replace("\\n", "\n")
        print(result)
