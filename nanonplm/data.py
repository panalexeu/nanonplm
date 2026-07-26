import os 
import json
import requests 
from abc import abstractmethod, ABC
from pathlib import Path 

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
         