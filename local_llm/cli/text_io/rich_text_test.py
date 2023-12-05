from .rich_text import RichText

def test_rich_text_word_gen():
    expected_words = [
        "Hello,",
        " ",
        "world!",
        "\n",
        "This",
        " ",
        "is",
        " ",
        "a",
        " ",
        "test."
    ]
    rt = RichText(text="Hello, world!  \nThis is a test.")

    words = list(rt.words())
    assert words == expected_words

def test_rich_text_word_gen_start_spacing():
    expected_words = [
        "  ",
        "Hello,",
        " ",
        "world!",
        "\n",
        "This",
        " ",
        "is",
        " ",
        "a",
        " ",
        "test."
    ]
    rt = RichText(text="  Hello, world!  \nThis is a test.")

    words = list(rt.words())
    assert words == expected_words

def test_rich_text_word_gen_newline_spacing():
    expected_words = [
        "  ",
        "Hello,",
        " ",
        "world!",
        "\n",
        "  ",
        "This",
        " ",
        "is",
        " ",
        "a",
        " ",
        "test."
    ]
    rt = RichText(text="  Hello, world!  \n  This is a test.")

    words = list(rt.words())
    assert words == expected_words

def test_rich_text_concatenation():
    rt1 = RichText(text="Hello, ")
    rt2 = RichText(text="world!")
    rt3 = RichText(text="This is a test.")
    rt1 += rt2
    rt1 += "  \n"
    rt1 += rt3
    assert rt1.text == "Hello, world!  \nThis is a test."

def test_clone_settings():
    rt1 = RichText(text="Hello, ")
    rt2 = rt1.clone_settings()
    assert rt1 is not rt2
    assert rt1.text == "Hello, "
    assert rt2.text == ""

    # No shared variables.
    rt2.text = "world"
    rt2.color = "red"
    rt2.background_color = "blue"
    rt2.bold = True
    rt2.underline = True
    assert rt1.text != rt2.text
    assert rt1.color != rt2.color
    assert rt1.background_color != rt2.background_color
    assert rt1.bold != rt2.bold
    assert rt1.underline != rt2.underline
    