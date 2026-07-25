import torch 
import argparse 
from pathlib import Path

from .model import Model, ModelConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="nplm sampling")
    parser.add_argument("ckpt_path", type=str) 
    args = parser.parse_args() 

    model = Model(config=ModelConfig(
        n=2, 
        embed=32,
        hidden=64,
        vocab=85_575
    ))
    ckpt_path = Path(args.ckpt_path) 
    model.load(ckpt_path)

    res = model.sample(x=torch.tensor([0, 1]), max_tokens=6)
    breakpoint()

