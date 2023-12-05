
def nop_function_parser(_):
    # Do not parse the input for functions
    pass

def quit_function_parser(input_str):
    if input_str in {'quit', 'quit()', ':quit', '/quit', ':q', '/q'}:
        raise SystemExit()

def is_yes_no_parser(input_str):
    return input_str.lower() in ["yes", "y", "no", "n"]

parser_map = {
    'nop': nop_function_parser,
    'quit': quit_function_parser,
    'is_yes_no': is_yes_no_parser,
}