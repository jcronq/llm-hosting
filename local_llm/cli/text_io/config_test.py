from .config import TextIOConfig
from .parsers import nop_function_parser

def test_default_config():
    io_config = TextIOConfig()
    assert io_config.input_indicator == ">"
    assert io_config.yes_no_fail_msg == "InvalidEntry: Expecting either 'yes' or 'no'."
    assert io_config.input_function_parser.__name__ ==  nop_function_parser.__name__