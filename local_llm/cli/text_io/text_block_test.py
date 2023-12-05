import pytest
from .rich_text import RichText
from .text_block import TextBlock, TextLine

def test_text_line_append():
    rt1 = RichText(text="Hello, ")
    rt2 = RichText(text="world!")
    rt3 = RichText(text="This is a test.")
    tl = TextLine([rt1])
    tl.append(rt2)
    tl += rt3
    assert tl.rich_text_list == [rt1, rt2, rt3]

def test_text_line_prepend():
    rt1 = RichText(text="Hello, ")
    rt2 = RichText(text="world!")
    rt3 = RichText(text="This is a test.")
    tl = TextLine([rt1])
    tl.prepend(rt2)
    tl.prepend(rt3)
    assert tl.rich_text_list == [rt3, rt2, rt1]

def test_text_line_len():
    rt1 = RichText(text="Hello, ")
    rt2 = RichText(text="world!")
    rt3 = RichText(text="This is a test.")
    tl = TextLine([rt1, rt2, rt3])
    assert len(tl) == 28

def test_error_richtext():
    with pytest.raises(AssertionError):
        TextLine(RichText(text="Hello, "))

def test_wrap_text():
    rt = RichText(text="Hello, world!")
    tb = TextBlock(width=9, rich_text_list=[rt])
    expected_wrapped_text = [
        TextLine([RichText(text="Hello, ")]),
        TextLine([RichText(text="world!")])
    ]
    result = list(tb.wrapped_text())
    assert result == expected_wrapped_text

def test_wrap_text_shortest_word_match():
    rt = RichText(text="Hello, world!")
    tb = TextBlock(width=6, rich_text_list=[rt])
    expected_wrapped_text = [TextLine([RichText(text="Hello,")]), TextLine([RichText(text="world!")])]
    result = list(tb.wrapped_text())
    assert result == expected_wrapped_text

    rt = RichText(text="Hello  \nworld!")
    tb = TextBlock(width=6, rich_text_list=[rt])
    expected_wrapped_text = [TextLine([RichText(text="Hello")]), TextLine([RichText(text="world!")])]
    result = list(tb.wrapped_text())
    assert result == expected_wrapped_text

def test_wrap_text_mid_whitespace_newline():
    rt = RichText(text="Hello \n world")
    tb = TextBlock(width=6, rich_text_list=[rt])
    expected_wrapped_text = [TextLine([RichText(text="Hello")]), TextLine([RichText(text=" world")])]
    result = list(tb.wrapped_text())
    assert result == expected_wrapped_text

    rt = RichText(text="Hello \n world!")
    tb = TextBlock(width=6, rich_text_list=[rt])
    expected_wrapped_text = [TextLine([RichText(text="Hello")]), TextLine([RichText(text=" ")]), TextLine([RichText(text="world!")])]
    result = list(tb.wrapped_text())
    assert result == expected_wrapped_text

def test_mid_word_break():
    rt = RichText(text="replicate")
    tb = TextBlock(width=5, rich_text_list=[rt])
    expected_wrapped_text = [TextLine([RichText(text="repli")]), TextLine([RichText(text="cate")])]
    result = list(tb.wrapped_text())
    assert result == expected_wrapped_text

def test_multi_mid_word_break():
    rt = RichText(text="replicate")
    tb = TextBlock(width=2, rich_text_list=[rt])
    expected_wrapped_text = [
        TextLine([RichText(text="re")]),
        TextLine([RichText(text="pl")]),
        TextLine([RichText(text="ic")]),
        TextLine([RichText(text="at")]),
        TextLine([RichText(text="e")]),
    ]
    result = list(tb.wrapped_text())
    assert result == expected_wrapped_text

def test_multi_mid_word_break_with_space():
    rt = RichText(text="repl icate")
    tb = TextBlock(width=2, rich_text_list=[rt])
    expected_wrapped_text = [
        TextLine([RichText(text="re")]),
        TextLine([RichText(text="pl")]),
        TextLine([RichText(text=" ")]),
        TextLine([RichText(text="ic")]),
        TextLine([RichText(text="at")]),
        TextLine([RichText(text="e")]),
    ]
    result = list(tb.wrapped_text())
    assert result == expected_wrapped_text

def test_multi_mid_word_start_with_space():
    rt = RichText(text=" repl icate")
    tb = TextBlock(width=2, rich_text_list=[rt])
    expected_wrapped_text = [
        TextLine([RichText(text=" ")]),
        TextLine([RichText(text="re")]),
        TextLine([RichText(text="pl")]),
        TextLine([RichText(text=" ")]),
        TextLine([RichText(text="ic")]),
        TextLine([RichText(text="at")]),
        TextLine([RichText(text="e")]),
    ]
    result = list(tb.wrapped_text())
    assert result == expected_wrapped_text

def test_multi_mid_word_start_with_space_edge_case_0():
    rt = RichText(text=" rep licate")
    tb = TextBlock(width=3, rich_text_list=[rt])
    expected_wrapped_text = [
        TextLine([RichText(text=" ")]),
        TextLine([RichText(text="rep")]),
        TextLine([RichText(text=" ")]),
        TextLine([RichText(text="lic")]),
        TextLine([RichText(text="ate")]),
    ]
    result = list(tb.wrapped_text())
    assert result == expected_wrapped_text

def test_border():
    text_width = 5
    border_width = 1
    padding_width = 1

    rt = RichText(text="Hello")
    tb = TextBlock(
        width=text_width,
        rich_text_list=[rt],
        border_thickness=border_width,
        padding_thickness=padding_width,
        border_color="red",
        padding_color="white",
        border_char="*",
        padding_char=" "
    )
    border = [RichText(text="*", color="red"), RichText(text="*", color="red")] 
    open_padding = [RichText(text="*", color="red"), RichText(text=" ", color="white")] 
    close_padding = open_padding[::-1]
    expected = [
        TextLine([*border, RichText(text="*****", color="red"), *border]),
        TextLine([*open_padding, RichText(text="     ", color="white"), *close_padding]),
        TextLine([*open_padding, RichText(text="Hello"), *close_padding]),
        TextLine([*open_padding, RichText(text="     ", color="white"), *close_padding]),
        TextLine([*border, RichText(text="*****", color="red"), *border]),
    ]
    result = list(tb.wrapped_text())
    assert result == expected
    for line in result:
        assert len(line) == text_width + border_width * 2 + padding_width * 2

def test_even_line_width_case():
    text_width = 40
    border_width = 1
    padding_width = 1
    text_str = "I cannot provide a specific time without additional context. Time varies depending on the location and the context, such as whether it is daytime or nighttime, whether it is standard time or daylight saving time, etc. Could you please provide more information so I can provide an accurate answer?"
    text = RichText(text=text_str)
    tb = TextBlock(
        width=text_width,
        rich_text_list=[text],
        border_thickness=border_width,
        padding_thickness=padding_width,
        border_color="red",
        padding_color="white",
        border_char="*",
        padding_char=" "
    )
    expected = [
        TextLine([RichText(text="I cannot provide a specific time without")]),
        TextLine([RichText(text="additional context. Time varies ")]),
        TextLine([RichText(text="depending on the location and the ")]),
        TextLine([RichText(text="context, such as whether it is daytime ")]),
        TextLine([RichText(text="or nighttime, whether it is standard ")]),
        TextLine([RichText(text="time or daylight saving time, etc. Could")]),
        TextLine([RichText(text="you please provide more information so I")]),
        TextLine([RichText(text="can provide an accurate answer?")])
    ]
    result = list(tb.wrapped_text())

    for line in result:
        assert len(line) == text_width + border_width * 2 + padding_width * 2

    for idx, line in enumerate(result[2:-2]):
        text_of_interest = line.rich_text_list[2]
        assert text_of_interest == expected[idx].rich_text_list[0]
    
