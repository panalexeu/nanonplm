"""
taken from: https://github.com/panalexeu/nanongram/blob/main/nanongram/tokenizer.py
"""
import re 

from .preproc import START_SEQ, END_SEQ

class BaseTokenizer:
    """
    regexp tokenizer with modified gpt2 expr to work with re
    """
    def __init__(self):
        self.exp = rf"{START_SEQ}| ?{END_SEQ}|'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+" 
                     # special tokens        # clitics               # a-n # non a-n 
        self.pat = re.compile(self.exp)

    def __call__(self, text: str, lower: bool = True) -> list[str]: 
        text = text.strip() 
        if lower: 
            text = text.lower() 

        return re.findall(self.pat, text)
    