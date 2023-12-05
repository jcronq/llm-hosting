import pytest

from input import get_input, get_yes_no, get_str
from local_llm.cli.text_io.config import TextIOConfig


def test_get_str(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr('builtins.input', lambda: "ChatBot")
    user_input = get_str("What is your name? ")
    assert user_input == "ChatBot"

def test_get_input(monkeypatch: pytest.MonkeyPatch):
    """Test get_input function."""

    monkeypatch.setattr('builtins.input', lambda: "ChatBot")
    user_input = get_input("What is your name? ")
    assert user_input == "ChatBot"

    monkeypatch.setattr('builtins.input', lambda: " ChatBot ")
    user_input = get_input("What is your name? ")
    assert user_input == "ChatBot"

    monkeypatch.setattr('builtins.input', lambda: "quit()")
    user_input = get_input("What is your name? ")
    assert user_input == "quit()"

def test_get_yes_no(monkeypatch: pytest.MonkeyPatch):
    """Test get_yes_no function."""
    config = TextIOConfig(None)


    monkeypatch.setattr('builtins.input', lambda: "Yes")
    yes_no = get_yes_no("Is your name Mark? ", config)
    assert yes_no == "yes"
    monkeypatch.setattr('builtins.input', lambda: "   YeS ")
    yes_no = get_yes_no("Is your name Mark? ", config)
    assert yes_no == "yes"
    monkeypatch.setattr('builtins.input', lambda: "   Y ")
    yes_no = get_yes_no("Is your name Mark? ", config)
    assert yes_no == "yes"
    monkeypatch.setattr('builtins.input', lambda: "   y ")
    yes_no = get_yes_no("Is your name Mark? ", config)
    assert yes_no == "yes"

    monkeypatch.setattr('builtins.input', lambda: "   nO ")
    yes_no = get_yes_no("Is your name Mark? ", config)
    assert yes_no == "no"
    monkeypatch.setattr('builtins.input', lambda: "   N ")
    yes_no = get_yes_no("Is your name Mark? ", config)
    assert yes_no == "no"
    monkeypatch.setattr('builtins.input', lambda: "No")
    yes_no = get_yes_no("Is your name Mark? ", config)
    assert yes_no == "no"
    monkeypatch.setattr('builtins.input', lambda: "n")
    yes_no = get_yes_no("Is your name Mark? ", config)
    assert yes_no == "no"

def test_get_str(monkeypatch: pytest.MonkeyPatch):
    """Test get_str function."""

    monkeypatch.setattr('builtins.input', lambda: "n")
    yes_no = get_yes_no("Is your name Mark? ")
    assert yes_no == "no"