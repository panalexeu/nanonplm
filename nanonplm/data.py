import os 
import re 
import json
import zipfile 
import requests 
from pathlib import Path 
from abc import abstractmethod, ABC


class BaseDataLoadStrategy(ABC): 
    @abstractmethod
    def __call__(self) -> str:
        """
        implement preprocessing behaviour here.
        the expected output is a list of strings that
        represent semantically meaningful, self-contained
        units of text.
        """
        pass 

class TgDataLoadStrategy(BaseDataLoadStrategy):
    def __init__(self, data_dir: Path, id_: str): 
        if not os.path.exists(data_dir): 
            os.mkdir(data_dir)

        self.cnv_path = data_dir / 'result.json'
        if not os.path.exists(self.cnv_path): 
            raise IOError(f"no {self.cnv_path} was found. put your tg export under dir: {data_dir} and name it result.json")

        with open(self.cnv_path, 'r') as f: 
            self.cnv = json.load(f)
        self.target = id_

    def __call__(self):
        msgs = [] 
        for msg in self.cnv['messages']:
            
            if msg.get('from_id') == self.target and isinstance(msg.get('text'), str):
                text = msg['text']

                # empty str check 
                if text.strip():
                    text = text.replace('\n', ' ') # TODO think what to do with newlines
                    msgs.append(text)

        msgs_text =  ' '.join(msgs)
        return msgs_text
    
class TinyShaekspereDataLoadStrategy(BaseDataLoadStrategy): 
    def __init__(self, data_dir: Path):
        if not os.path.exists(data_dir): 
            os.mkdir(data_dir)

        self.shaek_path = data_dir / "shaekspere.txt"
        if not os.path.exists(self.shaek_path): 
            data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
            with open(self.shaek_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(data_url).text)

    def __call__(self) -> str:
        with open(self.shaek_path, 'r') as f: 
            return f.read() 

class BrownDataLoadStrategy(BaseDataLoadStrategy): 
    """
    nltk implementation is used as reference. 
    i did not install nltk directly to have less deps, 
    and understand how Brown corpus is handled.
    ref: https://github.com/nltk/nltk/blob/develop/nltk/corpus/__init__.py#L80-L87
    """
    def __init__(self, data_dir: Path): 
        if not os.path.exists(data_dir): 
            os.mkdir(data_dir) 

        self.brown_url = 'https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/brown.zip' 
        self.brown_zpath = data_dir / "brown.zip"
        self.brown_path = data_dir / "brown"
        self.fileids = r'c[a-z]\d\d'
        
        if not os.path.exists(self.brown_path):
            res = requests.get(self.brown_url)
            with open(self.brown_zpath, 'wb') as f: 
                f.write(res.content)
            
            with zipfile.ZipFile(self.brown_zpath) as z: 
                z.extractall(data_dir)
            
            os.remove(self.brown_zpath)
        
    def __call__(self, merge: bool = True):
        tokens = []
        files = [f for f in self.brown_path.iterdir() if re.match(pattern=self.fileids, string=f.stem)]  
        for f in files: 
            text = f.read_text(encoding='ascii') 
            tokens.extend(
                self._get_tokens(text)
            )

        # from paper: "Rare words with frequency ≤ 3 were merged into a single symbol, reducing
        # the vocabulary size to |V | = 16, 383" 
        if merge: 
            self._merge_rare_tokens(tokens)

        # to keep base class interface resulting tokens 
        # are joined with space symbol *for now 
        return ' '.join(tokens) 

    def _get_tokens(self, text: str) -> list[str]: 
        tokens = []
        for s in self._sent_split(text):
            for w in self._word_split(s): 
                w, t = self._str2tuple(w)
                tokens.append(w)
        return tokens 

    def _sent_split(self, text: str): 
        """
        ref: 
        1. https://github.com/nltk/nltk/blob/develop/nltk/corpus/reader/tagged.py#L45
        2. https://github.com/nltk/nltk/blob/develop/nltk/tokenize/regexp.py#L122-L127
        """
        pattern = r'\n' 
        sents = [s for s in re.split(pattern=pattern, string=text) if s]
        return sents 

    def _word_split(self, text: str): 
        """
        ref: 
        1. https://github.com/nltk/nltk/blob/develop/nltk/corpus/reader/tagged.py#L44
        2. https://github.com/nltk/nltk/blob/develop/nltk/tokenize/regexp.py#L156-L169
        3. https://github.com/nltk/nltk/blob/develop/nltk/tokenize/regexp.py#L122-L127
        """
        pattern = r'\s+'
        words = [tok for tok in re.split(pattern=pattern, string=text) if tok]
        return words 

    def _str2tuple(self, s: str, sep='/') -> tuple: 
        """
        ref: 
        1. https://github.com/nltk/nltk/blob/develop/nltk/tag/util.py#L10
        """
        loc = s.rfind(sep)
        if loc >= 0: 
            return (s[:loc], s[loc+len(sep):].upper())
        else: 
            return (s, None)

    def _merge_rare_tokens(self, tokens: list[str], merge_count: int = 3) -> str: 
        count_ = dict()
        for t in tokens:
            count_[t] = count_.get(t, 0) + 1

        merge_token = "<merge>"
        for i, t in enumerate(tokens): 
            if count_[t] <= merge_count: tokens[i] = merge_token
