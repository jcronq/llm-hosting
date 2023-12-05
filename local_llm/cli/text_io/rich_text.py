from pydantic import BaseModel

whitespace = [" ", "\n", "\t"]
tab_width = 2

class RichText(BaseModel):
    text: str
    color: str = "primary_foreground"
    background_color: str = "primary_background"
    bold: bool = False
    underline: bool = False

    @staticmethod
    def is_whitespace(word: str):
        return all([char in whitespace for char in word])

    def __len__(self):
        return sum([1 if char.isprintable() else 0 for char in self.text])
    
    def append(self, txt: str):
        self.text += txt
        return self
    
    def __add__(self, other: str):
        if isinstance(other, RichText):
            other = other.text
        elif isinstance(other, str):
            pass
        else:
            raise TypeError(f"Cannot add {type(other)} to RichText")
        self.text += other
        return self
    
    def clone_settings(self):
        return RichText(
            text="",
            color=self.color,
            background_color=self.background_color,
            bold=self.bold,
            underline=self.underline,
        )

    def __eq__(self, other: "RichText"):
        if isinstance(other, RichText):
            return (
                self.text == other.text
                and self.color == other.color
                and self.background_color == other.background_color
                and self.bold == other.bold
                and self.underline == other.underline
            )
        else:
            return False

    def __repr__(self):
        return f'RichText("{str(self)}")'
    
    def __str__(self):
        return self.text

    def words(self):
        cur_word = ""
        is_whitespace = False
        for char in self.text:
            if self.is_whitespace(char):
                if is_whitespace or len(cur_word) == 0:
                    if char == '\t':
                        cur_word += " " * tab_width
                    elif char == '\n':
                        yield '\n'
                        cur_word = ""
                        is_whitespace = False
                    else:
                        cur_word += char
                        is_whitespace = True
                else:
                    yield cur_word
                    cur_word = char
                    is_whitespace = True
            else:
                if is_whitespace:
                    yield cur_word
                    cur_word = char
                    is_whitespace = False
                else:
                    cur_word += char
        yield cur_word

    