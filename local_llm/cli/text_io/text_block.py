from typing import List
from .rich_text import RichText
from .printers import PrinterBase

class TextLine:
    def __init__(self, rich_text_list: List[RichText]):
        assert isinstance(rich_text_list, list)
        self.rich_text_list = rich_text_list

    def __len__(self):
        return sum([len(rich_text) for rich_text in self.text])

    def print(printer: PrinterBase):
        printer.print()

    def append(self, rich_text: RichText):
        self.rich_text_list.append(rich_text)

    def prepend(self, rich_text: RichText):
        self.rich_text_list = [rich_text, *self.rich_text_list]

    def count(self):
        return len(self.rich_text_list)

    def __len__(self):
        return sum([len(rich_text) for rich_text in self.rich_text_list])

    def __add__(self, other: RichText):
        if isinstance(other, RichText):
            self.rich_text_list.append(other)
        else:
            raise TypeError(f"Cannot add {type(other)} to TextLine")
        return self
    
    def __repr__(self):
        return f"TextLine([{','.join([rich_text.__repr__() for rich_text in self.rich_text_list])}])"

    def __str__(self):
        return "".join([f'"{rich_text}"' for rich_text in self.rich_text_list])
    
    def __eq__(self, other: "TextLine"):
        if isinstance(other, TextLine):
            return self.rich_text_list == other.rich_text_list
        else:
            return False

class TextBlock:
    def __init__(
        self,
        width: int,
        rich_text_list: List[RichText],
        padding_char: str = " ",
        border_char: str = "*",
        padding_thickness: int = 0,
        border_thickness: int = 0,
        border_color: str = "red",
        padding_color: str = "white",
    ):
        # Text blocks have a max width, the height however is variable. Height will expand to fit the text.
        self.width = width
        self.rich_text_list = rich_text_list
        self.padding_char = padding_char
        self.border_char = border_char
        self.padding_thickness = padding_thickness
        self.border_thickness = border_thickness
        self.border_color = border_color
        self.padding_color = padding_color

    def __len__(self):
        return sum([len(rich_text) for rich_text in self.text])

    def format_text_line(self, line: TextLine, pad_with_border=False):
        if pad_with_border:
            padding = self.border_char
            padding_color = self.border_color
        else:
            padding = self.padding_char
            padding_color = self.padding_color
        if self.border_thickness > 0:
            line.prepend(RichText(text=padding * self.padding_thickness, color=padding_color))
            if len(line) <= self.width:
                rich_text = line.rich_text_list[-1].clone_settings()
                rich_text.append(" " * (self.width - len(line) +1))
                line.append(rich_text)
            line.append(RichText(text=padding * self.padding_thickness, color=padding_color))
        if self.border_thickness > 0:
            line.prepend(RichText(text=self.border_char * self.border_thickness, color=self.border_color))
            line.append(RichText(text=self.border_char * self.border_thickness, color=self.border_color))
        return line

    @property
    def horizontal_border(self):
        text = RichText(text=self.border_char * self.width, color=self.border_color)
        text_line = TextLine([text])
        border = self.format_text_line(text_line, pad_with_border=True)
        return border

    @property
    def horizontal_padding(self):
        text = RichText(text=self.padding_char * self.width, color=self.padding_color)
        text_line = TextLine([text])
        padding = self.format_text_line(text_line)
        return padding

    
    def wrapped_text(self):
        if self.border_thickness > 0:
            yield self.horizontal_border
        if self.padding_thickness > 0:
            yield self.horizontal_padding
        cur_line = TextLine([])
        for rich_text in self.rich_text_list:

            new_rich_text = rich_text.clone_settings()
            for word in rich_text.words():

                if len(cur_line) + len(new_rich_text) + len(word) > self.width:

                    if len(new_rich_text) > 0:

                        cur_line += new_rich_text
                        yield self.format_text_line(cur_line)
                    if len(word) >= self.width: # If the word is longer than the width

                        #Split up the word into multiple lines
                        new_rich_text = rich_text.clone_settings()
                        for i in range(0, len(word), self.width):

                            new_rich_text += word[i:i+self.width]
                            if len(new_rich_text) == self.width:

                                cur_line = TextLine([new_rich_text])
                                yield self.format_text_line(cur_line)
                                cur_line = TextLine([])
                                new_rich_text = rich_text.clone_settings()
                    else:

                        new_rich_text = rich_text.clone_settings()
                        if not RichText.is_whitespace(word):

                            new_rich_text += word
                        cur_line = TextLine([])
                elif RichText.is_whitespace(word) and any([char == "\n" for char in word]):

                    cur_line += new_rich_text
                    yield self.format_text_line(cur_line)
                    new_rich_text = rich_text.clone_settings()
                    cur_line = TextLine([])
                else:

                    new_rich_text += word
            cur_line += new_rich_text
        if len(cur_line) > 0:

            yield self.format_text_line(cur_line)
        
        if self.padding_thickness > 0:
            yield self.horizontal_padding
        if self.border_thickness > 0:
            yield self.horizontal_border


    # def wrapped_text(self) -> list[RichText]:
    #     for rich_text in self.rich_text_list:
    #         lines = []
    #         line = " "*offset

    #         for word in rich_text.words:
    #             if(len(line) + 1 + len(word) > blockSize+offset):
    #                 spacer = " " * ((blockSize+offset)-len(line))
    #                 line = line + spacer
    #                 lines.append(line)
    #                 line = (" " * offset) + word
    #             else:
    #                 if(len(line) != offset):
    #                     line = line + " "
    #                 line = line + word

    #         spacer = " " * ((blockSize+offset)-len(line))
    #         line = line + spacer
    #         lines.append(line)
    #         return "\n".join(lines)

# def add_colors_from_blocks(txt, searchChar, noMatchColor, matchColor):
#     charIndex = txt.find(searchChar)
#     eolIndex = txt.find("\n")
#     if(eolIndex < 0):
#         eolIndex = len(txt)
#     lines = []
#     inBlock = False
#     while charIndex >= 0 or eolIndex != len(txt):
#         index = min(charIndex, eolIndex)
#         if(index < 0):
#             index = eolIndex
#         line = txt[0:index]
#         if inBlock:
#             line = colored(line, matchColor)
#         else:
#             line = colored(line, noMatchColor)

#         if(txt[index] == '\n'):
#             addtext="\n"
#         else:
#             addtext=" "
#             inBlock = not inBlock
#         lines.append(line+addtext)
#         txt = txt[index+1:len(txt)]
#         charIndex = txt.find(searchChar)
#         eolIndex = txt.find("\n")
#         if(eolIndex < 0):
#             eolIndex = len(txt)
#     if(len(txt) > 0):
#         lines.append(colored(txt, noMatchColor))
#     return "".join(lines)

# def wrap_in_block(txt, blockSize):
#     blockLine = "_" * blockSize
#     tailBlockLine = "=" * blockSize
#     blockPrefix = "| "
#     blockPostfix = " |"
#     blockFixSize = len(blockPostfix) + len(blockPrefix)

#     colorBlock = add_colors_from_blocks(txt, "*", color_map['color_nar'], color_map['color_env'])
#     blockedColorBlock = colored(blockLine + "\n| ", 'white') + colorBlock.replace("\n", colored(" |\n| ", 'white')) + colored(" |\n", 'white') + tailBlockLine
#     return blockedColorBlock

# def faux_type_print(txt, wordsPerMinute):
#     for ch in txt:
#         print(ch, end = "")
#         time.sleep((1/wordsPerMinute)/60)
#     print("")

# def wrap_text_block(txt, blockSize = 40, offset = 0):
#     if isinstance(txt, str):
#         linesRaw = txt.splitlines()
#     elif isinstance(txt, list):
#         linesRaw = txt
#     else:
#         return "ERROR!!! input is neither a string or array"

#     wrappedLines = []
#     for rawLine in linesRaw:
#         split_lines = rawLine.splitlines()
#         for split_line in split_lines:
#             wrappedLines.append(wrap_text(split_line, blockSize - 4, offset))
#     wrappedText = "\n".join(wrappedLines)

#     return wrappedText

# def print_game_block(txt, blockSize=60, offset=0):
#     new_txt = []
#     if isinstance(txt, list):
#         for i, line in enumerate(txt):
#             new_txt.append(remove_newline(line).replace("<br>", " \n"))
#             if i != len(txt) -1:
#                 new_txt.append(' ')
#     else:
#         new_txt.append(remove_newline(txt))
#     wrappedText = wrap_text_block(new_txt, blockSize, offset)
#     block = wrap_in_block(wrappedText, blockSize)
#     faux_type_print(block, 60)

# def replace_with_colors(line):
#     line = replace_with_color(line, '*', color_map['color_nar'])
#     line = replace_with_color(line, '@', color_map['color_item'])
#     line = replace_with_color(line, '^', color_map['color_loc'])
#     line = replace_with_color(line, '#', color_map['color_env'])
#     return line

# def replace_with_color(line, splitOn, color):
#     splits = line.split(splitOn)
#     result_list = []
#     in_color_blk = False
#     for index, split in enumerate(splits):
#         if index == 0:
#             result_list.append(split)
#         elif in_color_blk:
#             result_list.append(split)
#             in_color_blk = False
#         elif not in_color_blk:
#             result_list.append(add_color(split, color))
#             in_color_blk = True

#     return ''.join(result_list)

# def format_html_block(msg_block):
#     new_txt = []
#     if isinstance(msg_block, list):
#         for i, line in enumerate(msg_block):
#             normal_line = remove_newline(line)
#             colored_line = replace_with_colors(normal_line)
#             new_txt.append(colored_line)
#             if i != len(msg_block) -1:
#                 new_txt.append(' ')
#     else:
#         colored_line = replace_with_colors(
#             remove_newline(msg_block)
#         )
#         new_txt.append(colored_line)
#     return new_txt