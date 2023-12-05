from typing import Optional

class Colorizer:
    color_codes = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
        "black": "\033[98m",
        "bold": "\033[1m",
        "underline": "\033[4m",
        "end": "\033[0m",
    }

    text_type_to_color_map = {
        "primary": "white",
        "secondary": "yellow",
        "success": "green",
        "danger": "red",
        "warning": "yellow",
        "info": "cyan",
        "narrator": "green",
        "env": "blue",
        "item": "magenta",
        "location": "red",
        "util": "red",
    }

    ansi_map = {
        'black': '30',
        'red': '31',
        'green': '32',
        'yellow': '33',
        'blue': '34',
        'magenta': '35',
        'cyan': '36',
        'white': '37',
        'reset': '0'
    }

    def __init__(self):
        self._ansi_overrides = {}
        self._text_type_to_color_overrides = {}
        self._color_codes_overrides = {}

    def to_ansi(self, txt, color_name):
        ansi_color = self._ansi_overrides.get(color_name, self.ansi_map[color_name])
        return f'\u001b[{ansi_color}m{txt}\u001b[0m'

    def color_from_text_type(self, text_type):
        return self._text_to_color_overrides.get(text_type, self.text_to_color_map[text_type])

    def set_text_type_to_color(self, text_type, color):
        self._text_to_color_overrides[text_type] = color
