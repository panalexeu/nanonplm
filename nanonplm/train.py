"""
the training loop implementation (and model implementation too) 
uses SGD (no batching) as it was done in the paper. 
"""
import time 
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
    log_step = 100
    seed = 42 
    decay = 10e-5
    ema_alpha = 0.01  # avgs loss over ~100 steps 
    n = 2 # trigram 
    device_type = "cuda" # "cpu"
    cfg = ModelConfig(
        n=n, 
        embed=32,
        hidden=64,
        vocab=85_575
    )
    device = torch.device(device_type)
    model = Model(config=cfg).to(device)
    random.seed(seed)

    # info
    print("--------------------------")
    print(f"total model params: {model._get_num_params()}")
    print(f"vocab size: {len(vocab)}")
    print(f"data tokens: {len(tokens)}")
    print("--------------------------")

    # training loop 
    ema = lambda loss, ema_loss: ema_alpha*loss + (1-ema_alpha)*ema_loss 
    ema_loss = None
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=decay)
    t0 = time.perf_counter()
    for i in range(0, steps): 
        optimizer.zero_grad()
        x_ids, y_ids = _get_sample(tokens, n, vocab_lookup_table)
        x_ids, y_ids = x_ids.to(device), y_ids.to(device)
        _, loss = model.__call__(x_ids, y_ids)
        ema_loss = ema(loss.item(), ema_loss if ema_loss else loss.item())
        loss.backward() 
        optimizer.step()

        if i % log_step == 0: 
            t1 = time.perf_counter()
            dt = t1 - t0 
            t0 = t1 

            print(f"step: {i}, train loss {loss.item():.4f}, ema loss (alpha={ema_alpha}) {ema_loss:.4f}, dt {dt:.2f}s")
