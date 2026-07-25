import torch 
import argparse 
from pathlib import Path

from .tokenizer import BaseTokenizer
from .model import Model, ModelConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="nplm sampling")
    parser.add_argument("ckpt_path", type=str) 
    args = parser.parse_args() 

    ckpt_path = Path(args.ckpt_path) 
    model = Model.from_pretrained(ckpt_path)
    tokenizer = BaseTokenizer()

    input = "<s> hello"
    tokens = tokenizer.__call__(input)
    input_ids = torch.tensor([model.vocab_table[t] for t in tokens])
    out_ids = model.sample(x=input_ids, max_tokens=6)
    vocab_tokens = list(model.vocab_table.keys())
    sampled_tokens = [vocab_tokens[id] for id in out_ids]

    print(input)
    print(tokens)
    print(input_ids)
    print(out_ids)
    print(''.join(sampled_tokens))
