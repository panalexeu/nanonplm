"""
the training loop implementation (and model implementation too) 
uses SGD (no batching) as it was done in the paper. 
"""
import argparse 
from pathlib import Path 

from .model import ModelConfig, Model
from .preproc import TgPreporcStrategy

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="tg ingestion pipeline: cnv preprocessing -> neural prob. language model training")
    parser.add_argument("user_id", type=str)
    parser.add_argument("cnv_path", type=str) 
    args = parser.parse_args() 

    preproc_ = TgPreporcStrategy(
        path=Path(args.cnv_path), 
        id_=args.user_id 
    )
