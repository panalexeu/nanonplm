"""
the training loop implementation (and model implementation too) 
uses SGD (no batching) as it was done in the paper. 
"""
import random
import argparse 
from pathlib import Path 

import torch 
import torch.optim as optim 

from .model import ModelConfig, Model
from .preproc import TgPreporcStrategy
from .tokenizer import BaseTokenizer


def _get_sample(tokens: list[str], n: int, vocab_table: dict) -> tuple[torch.Tensor, torch.Tensor]:
    idx = random.randint(0, len(tokens)-n)
    x, y = tokens[idx:idx+2], tokens[idx+2]
    x_ids = torch.tensor([vocab_table[t] for t in x], dtype=torch.int)
    y_ids =  torch.tensor(vocab_table[y], dtype=torch.long)

    return x_ids, y_ids 

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="tg ingestion pipeline: cnv preprocessing -> neural prob. language model training")
    parser.add_argument("user_id", type=str)
    parser.add_argument("cnv_path", type=str) 
    parser.add_argument("--preproc_export_path", type=str, default='./out.txt')
    args = parser.parse_args() 

    preproc_ = TgPreporcStrategy(
        path=Path(args.cnv_path), 
        id_=args.user_id 
    )
    tokenizer = BaseTokenizer()

    texts = preproc_.__call__()
    res = preproc_.join(texts) # join texts into one string
    tokens = tokenizer.__call__(res)
    vocab = set(tokens) 
    vocab_lookup_table = {v: i for i, v in  enumerate(vocab)}
    
    # model/training config 
    lr = 10e-3
    steps = 1_738_070 # 5 ~epochs over (1042842 / 3 (trigram))
    log_step = 1000
    seed = 42 
    decay = 10e-5
    n = 2 # trigram 
    cfg = ModelConfig(
        n=n, 
        embed=32,
        hidden=64,
        vocab=85_575
    )
    model = Model(config=cfg)
    random.seed(seed)

    # info
    print("--------------------------")
    print(f"total model params: {model._get_num_params()}")
    print(f"vocab size: {len(vocab)}")
    print(f"data tokens: {len(tokens)}")
    print("--------------------------")

    # training loop 
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay)
    for i in range(0, steps): 
        optimizer.zero_grad()
        x_ids, y_ids = _get_sample(tokens, n, vocab_lookup_table)
        _, loss = model.__call__(x_ids, y_ids)
        loss.backward() 
        optimizer.step()

        if i % log_step == 0: 
            print(f"step: {i}, train loss {loss}")
        