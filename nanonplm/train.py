"""
the training loop implementation (and model implementation too) 
uses SGD (no batching) as it was done in the paper. 
"""
import os 
import math
import time 
import random
import argparse
from pathlib import Path 
from dataclasses import asdict

import torch 

from .model import ModelConfig, Model
from .data import (
    TgDataLoadStrategy, 
    BrownDataLoadStrategy,
    TinyShaekspereDataLoadStrategy
)
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
    parser.add_argument("--test_cutoff", type=float, default=0.9) 
    args = parser.parse_args() 

    # dataloader = TgDataLoadStrategy(
    #     data_dir=Path(args.data_dir), 
    #     id_=args.user_id 
    # )
    # dataloader = TinyShaekspereDataLoadStrategy(data_dir=Path(args.data_dir))
    dataloader = BrownDataLoadStrategy(data_dir=Path(args.data_dir))
    tokenizer = BaseTokenizer()

    text = dataloader.__call__()
    tokens = tokenizer.__call__(text)
    test_cutoff = int(len(tokens) * args.test_cutoff) + 1 
    train_tokens = tokens[:test_cutoff] 
    test_tokens = tokens[test_cutoff:]
    vocab = set(tokens) 
    vocab_lookup_table = {v: i for i, v in  enumerate(vocab)}
    
    # model/training config 
    # basically repeats config for:MLP9 (table 1), Brown
    lr = 1e-2
    r = 1e-7
    steps = 3_539_420 # 20 ~epochs over (1_061_825 / 6 (6gram) ~176_971)
    log_step = 1_000
    ckpt_save_step = 1_000_000
    ckpt_dir = Path('./out')
    seed = 42 
    decay = 1e-3
    ema_alpha = 0.001  # avgs loss over ~1000 steps 
    n = 5 # 6gram 
    device_type = "cuda" # "cpu"
    cfg = ModelConfig(
        n=n, 
        embed=30,
        hidden=100,
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
    print(f"total tokens: {len(tokens)}")
    print(f"train tokens: {len(train_tokens)}")
    print(f"test tokens: {len(test_tokens)}")
    print("--------------------------")
    input("press Enter to start training: ")

    if not ckpt_dir.exists(): 
        os.mkdir(ckpt_dir)

    # training loop 
    ema = lambda loss, ema_loss: ema_alpha*loss + (1-ema_alpha)*ema_loss 
    ema_loss = None
    optimizer = model.configure_optimizer(lr=lr, weight_decay=decay)
    t0 = time.perf_counter()
    for i in range(0, steps): 
        # lr update
        new_lr = lr/(1 + r*i)
        for g in optimizer.param_groups:
            g['lr'] = new_lr

        # fwd pass 
        optimizer.zero_grad()
        x_ids, y_ids = _get_sample(tokens, n, vocab_lookup_table)
        x_ids, y_ids = x_ids.to(device), y_ids.to(device)
        _, loss = model.__call__(x_ids, y_ids)
        ema_loss = ema(loss.item(), ema_loss if ema_loss else loss.item())
        loss.backward() 
        optimizer.step()

        # logging 
        if i % log_step == 0: 
            t1 = time.perf_counter()
            dt = t1 - t0 
            t0 = t1 

            print(f"step: {i}, train loss {loss.item():.4f}, ema loss (alpha={ema_alpha}): {ema_loss:.4f}, dt: {dt:.2f}s, lr: {new_lr}")

        # save ckpt 
        if (i % ckpt_save_step == 0 and i != 0) or (i == steps-1): 
            ckpt_path = ckpt_dir / Path(f'ckpt{i}.pt')
            _save_ckpt(
                model_state_dict=model.state_dict(), 
                vocab_table=vocab_lookup_table,
                model_cfg=asdict(cfg),
                ckpt_path=ckpt_path
            )
            print(f"saved ckpt at: {ckpt_path}")

    # ppl eval 
    if len(test_tokens) > 0: 
        input("press Enter to evaluate on test set: ")
        losses = []
        model.eval()
        for i in range(0, len(test_tokens)-n-1, n): 
            x = test_tokens[i:i+n]
            y = test_tokens[i+n]
            x_ids = torch.tensor([vocab_lookup_table[t] for t in x], dtype=torch.int)
            y_ids =  torch.tensor(vocab_lookup_table[y], dtype=torch.long)
            x_ids, y_ids = x_ids.to(device), y_ids.to(device)
            _, loss = model.__call__(x_ids, y_ids)
            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        ppl = math.exp(avg_loss)
        print(f"test set ppl.: {ppl:.2f}")
