import pytest
import local_llm.cli.text_io.parsers as parsers


def test_nop_function_parser():
    # Do not parse the input for functions
    assert parsers.nop_function_parser("Hello") is None

def test_quit_function_parser():
    assert parsers.quit_function_parser("bye") is None
    with pytest.raises(SystemExit):
        assert parsers.quit_function_parser("quit()")
    with pytest.raises(SystemExit):
        assert parsers.quit_function_parser("quit")
    with pytest.raises(SystemExit):
        assert parsers.quit_function_parser(":quit")
    with pytest.raises(SystemExit):
        assert parsers.quit_function_parser("/quit")
    with pytest.raises(SystemExit):
        assert parsers.quit_function_parser(":q")
    with pytest.raises(SystemExit):
        assert parsers.quit_function_parser("/q")

def test_is_yes_no_parser():
    assert parsers.is_yes_no_parser("yes") is True
    assert parsers.is_yes_no_parser("YeS") is True
    assert parsers.is_yes_no_parser("YES") is True
    assert parsers.is_yes_no_parser("Y") is True
    assert parsers.is_yes_no_parser("y") is True

    assert parsers.is_yes_no_parser("no") is True
    assert parsers.is_yes_no_parser("No") is True
    assert parsers.is_yes_no_parser("nO") is True
    assert parsers.is_yes_no_parser("NO") is True
    assert parsers.is_yes_no_parser("n") is True
    assert parsers.is_yes_no_parser("N") is True

    assert parsers.is_yes_no_parser("neS") is False
    assert parsers.is_yes_no_parser("yo") is False
    assert parsers.is_yes_no_parser("ye") is False
    assert parsers.is_yes_no_parser("es") is False
    assert parsers.is_yes_no_parser("o") is False
    assert parsers.is_yes_no_parser("s") is False
    assert parsers.is_yes_no_parser("ys") is False
    assert parsers.is_yes_no_parser("e") is False
