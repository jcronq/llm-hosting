from typing import Callable, Any
from termcolor import colored, cprint
import time
import re
from local_llm.cli.text_io.config import TextIOConfig
from local_llm.cli.text_io.utils import remove_newline
from local_llm.cli.text_io.parsers import nop_function_parser
from local_llm.cli.text_io.colorizer import Colorizer
from local_llm.cli.text_io.text_block import TextBlock
from . import PrinterBase


class ConsolePrinter(PrinterBase):
    @staticmethod
    def c_print(color_name, txt):
        cprint(txt, color_name, end="")

    @staticmethod
    def util_print(colorizer: Colorizer, msg):
        cprint(msg, colorizer.color_from_text_type('color_util'))
    
    @staticmethod
    def print(text_block: TextBlock):
        text_lines = text_block.wrapped_text()
        for text_line in text_lines:
            for rich_text in text_line.rich_text_list:
                ConsolePrinter.c_print(rich_text.color, rich_text.text)
            print()


# def colorize(txt, noMatchColor, colored_words):
#     indexes = []
#     for colored_word in colored_words:
#         searchWord = colored_word['word']
#         color = colored_word['color']
#         reg_ex = re.compile(r'(([ .,!?\n\t]|^){1}'+searchWord+'([ .,!?\n\t]|$){1})')
#         res = reg_ex.findall(txt)
#         index = -1
#         if len(res) > 0:
#             pattern = res[0][0]
#             index = txt.find(pattern)
#             word_len = len(pattern)
#         if index >= 0:
#             indexes.append({'end': index + word_len, 'start': index, 'color': color})
#     indexes.sort(key=lambda i: i['start'])
#     colorized_words = []
#     for i, index in enumerate(indexes):
#         if i == 0 and index['start'] != 0:
#             colorized_words.append(colored(txt[0:index['start']],noMatchColor))
#         colorized_words.append(colored(txt[index['start']:index['end']], index['color']))
#         if i == len(indexes) - 1 and index['end'] != len(txt):
#             colorized_words.append(colored(txt[index['end']:], noMatchColor))

#     return ''.join(colorized_words)
