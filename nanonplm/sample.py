import torch 
import argparse 
from pathlib import Path

from .model import Model, ModelConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="nplm sampling")
    parser.add_argument("ckpt_path", type=str) 
    args = parser.parse_args() 

    ckpt_path = Path(args.ckpt_path) 
    model = Model.from_pretrained(ckpt_path)

    ids = model.sample(x=torch.tensor([0, 1]), max_tokens=6)
    vocab_tokens = list(model.vocab_table.keys())
    sampled_tokens = [vocab_tokens[id] for id in ids]
    print(' '.join(sampled_tokens))
