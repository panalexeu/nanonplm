"""
the training loop implementation (and model implementation too) 
uses SGD (no batching) as it was done in the paper. 
"""
import os 
import time 
import random
import argparse
from pathlib import Path 
from dataclasses import asdict

import torch 
import torch.optim as optim 

from .model import ModelConfig, Model
from .data import TgDataLoadStrategy, TinyShaekspereDataLoadStrategy
from .tokenizer import BaseTokenizer


def _get_sample(tokens: list[str], n: int, vocab_table: dict) -> tuple[torch.Tensor, torch.Tensor]:
    idx = random.randint(0, len(tokens)-1-n)
    x, y = tokens[idx:idx+n], tokens[idx+n]
    x_ids = torch.tensor([vocab_table[t] for t in x], dtype=torch.int)
    y_ids =  torch.tensor(vocab_table[y], dtype=torch.long)

    return x_ids, y_ids 

def _save_ckpt(model_state_dict: dict, vocab_table: dict, model_cfg: dict, ckpt_path: Path):
    ckpt = {
        'model': model_state_dict,
        'vocab_table': vocab_table, 
        'model_cfg': model_cfg
    }
    torch.save(ckpt, ckpt_path)

if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description="data ingestion pipeline: preprocessing -> neural prob. language model training")
    parser.add_argument("--user_id", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default='./data') 
    parser.add_argument("--preproc_export_path", type=str, default='./out.txt')
    args = parser.parse_args() 

    # preproc_ = TgDataLoadStrategy(
    #     data_dir=Path(args.data_dir), 
    #     id_=args.user_id 
    # )
    preproc_ = TinyShaekspereDataLoadStrategy(data_dir=Path(args.data_dir))
    tokenizer = BaseTokenizer()

    text = preproc_.__call__()
    tokens = tokenizer.__call__(text)
    vocab = set(tokens) 
    vocab_lookup_table = {v: i for i, v in  enumerate(vocab)}
    
    # model/training config 
    lr = 1e-3
    steps = 1_000_000 # 10 ~epochs over (297_832 / 3 (trigram))
    log_step = 1_000
    ckpt_save_step = 100_000
    ckpt_dir = Path('./out')
    seed = 42 
    decay = 1e-5
    ema_alpha = 0.001  # avgs loss over ~1000 steps 
    n = 2 # bigram 
    device_type = "cuda" # "cpu"
    cfg = ModelConfig(
        n=n, 
        embed=32,
        hidden=64,
        vocab=len(vocab)
    )
    device = torch.device(device_type)
    model = Model(config=cfg).to(device)
    # # compilation makes dt actually slower 
    # torch.set_float32_matmul_precision('high')
    # model = torch.compile(model)
    random.seed(seed)
    torch.manual_seed(seed)

    # info
    print("--------------------------")
    print(f"total model params: {model._get_num_params()}")
    print(f"vocab size: {len(vocab)}")
    print(f"data tokens: {len(tokens)}")
    print("--------------------------")
    input("press Enter to start training: ")

    if not ckpt_dir.exists(): 
        os.mkdir(ckpt_dir)

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

            print(f"step: {i}, train loss {loss.item():.4f}, ema loss (alpha={ema_alpha}): {ema_loss:.4f}, dt: {dt:.2f}s")

        if (i % ckpt_save_step == 0 and i != 0) or (i == steps-1): 
            ckpt_path = ckpt_dir / Path(f'ckpt{i}.pt')
            _save_ckpt(
                model_state_dict=model.state_dict(), 
                vocab_table=vocab_lookup_table,
                model_cfg=asdict(cfg),
                ckpt_path=ckpt_path
            )
            print(f"saved ckpt at: {ckpt_path}")
