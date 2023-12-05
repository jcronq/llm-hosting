from typing import Optional
import yaml

from local_llm.cli.text_io.parsers import parser_map

class TextIOConfig:
    def __init__(self, conf_file: Optional[str]=None):
        if conf_file is not None:
            with open(conf_file) as _f:
                conf = yaml.safe_load(_f)
                self._conf_values = conf
        else:
            self._conf_values = {}

    @property
    def yes_no_fail_msg(self):
        return self._conf_values.get("yes_no_fail_msg", "InvalidEntry: Expecting either 'yes' or 'no'.")

    @property
    def input_indicator(self):
        return self._conf_values.get("input_prompt", ">")

    @property
    def input_function_parser(self):
        return parser_map[self._conf_values.get("function_parser", "nop")]