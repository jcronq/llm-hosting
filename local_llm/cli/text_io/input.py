from termcolor import colored
from local_llm.cli.text_io.config import TextIOConfig
from local_llm.cli.text_io.parsers import is_yes_no_parser
from local_llm.cli.text_io.output import print_game_block


def get_input(prompt: str, end=""):
    print(prompt, end=end)
    input_str = input().strip()
    return input_str

def get_yes_no(yes_no_prompt, io_config: TextIOConfig):
    """Takes a function parser, because whenever we enter an input loop, give an option to quit"""
    input_str = get_input(f"{yes_no_prompt} {io_config.input_indicator} ").lower()
    while not is_yes_no_parser(input_str):
        io_config.input_function_parser(input_str)
        print_game_block(io_config.yes_no_fail_msg)
        input_str = get_input()
    return "yes" if input_str == "yes" or input_str == "y" else "no"

def get_str(prompt: str, prompt_end=":", default=None):
    if default is not None:
        prompt += colored(f" ({default}){prompt_end} ", "blue")
    else:
        prompt += f"{prompt_end} "
    input_str = get_input(start_char = prompt)
    if input_str == 'quit' or input_str == ':q':
        quit()
    if default is not None and len(input_str) == 0:
        input_str = default
    confirmed = False
    # while not confirmed:
    while len(input_str) <= 0:
        print_game_block("No input was entered.")
        input_str = get_input()
        if input_str == 'quit' or input_str == ':q':
            quit()
        if default is not None and len(input_str) == 0:
            input_str = default
        # printGameBlock(inStr)
        # confirmed = getYesNo("Is this OK?\n> ")
        # if not confirmed:
        #     inStr = ""
    return input_str

def prompt_input(prompt, options):
    print(prompt)

    legal_keys = [option['key'] for option in options] + ['quit']

    opt_text = '<br>'.join(format_html_block("  ".join([
        f"#{option['key']}#: *{option['text']}*"
        for option in options
    ])))
    print(opt_text.replace('<br>', '\n'))

    cmd = ['']
    while cmd[0] not in legal_keys:
        try:
            cmd = input("> ").strip().split(' ')
            if cmd[0] == 'quit':
                return ['quit']
            elif cmd[0] not in legal_keys:
                print(f"Please select option from {legal_keys}")
                cmd = ['']
        except:
            cmd = ['']

    return cmd

def get_option(prompt, options, default_selection = -1, numeral_color='yellow', block_size = 40):
    print('')
    print(prompt)
    print('')

    for opt_num, line in enumerate(options):
        print(colored(f'{opt_num+1})', numeral_color), end='')
        option = wrap_text(line, block_size, offset = 5)
        print(option[4:])

    selection = -1
    while selection < 0:
        try:
            user_input = input("> ").strip()
            if user_input == 'quit' or user_input == ':q':
                return 'quit'
            elif user_input == '' and default_selection >= 0:
                return default_selection
            elif user_input.isnumeric():
                selection = int(user_input)-1
            else:
                selection = -1
        except:
            selection = -1
        if selection < 0 or selection > len(options)-1:
            print(f"Please select a number between 1-{len(options)}")
            if default_selection >= 0:
                print(f"You can also press enter to confirm the default selection: {default_selection}")
        else:
            return selection
