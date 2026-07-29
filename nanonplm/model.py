from typing import Self
from pathlib import Path

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from dataclasses import dataclass

@dataclass
class ModelConfig(): 
    n: int = 4
    embed: int = 32 
    hidden: int = 128
    vocab: int = 4096 
     
class Model(nn.Module): 
    def __init__(self, config: ModelConfig, vocab_table: dict = dict()):
        super().__init__()
        self.config = config 
        self.embed = nn.Embedding(
            num_embeddings=self.config.vocab, 
            embedding_dim=self.config.embed, 
        )
        self.down_proj = nn.Linear(
            in_features=self.config.embed * self.config.n, 
            out_features=self.config.hidden, 
            bias=True 
        ) 
        self.out_proj = nn.Linear(
            in_features=self.config.hidden, 
            out_features=self.config.vocab, 
            bias=True
        ) 
        self.vocab_table = vocab_table
        self.r_vocab_table = list(vocab_table.keys())

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module): 
        if isinstance(module, nn.Linear): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: 
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_num_params(self,): 
        return sum(p.numel() for p in self.parameters())
        
    def __call__(self, x: torch.Tensor, target: torch.Tensor | None): 
        assert x.size(0) == self.config.n, f"input size(0) should equal == {self.config.n}"

        x = self.embed(x)
        x = x.reshape(-1)
        x = F.tanh(self.down_proj(x)) 
        x = self.out_proj(x) 

        loss = None 
        if target is not None: 
            loss = F.cross_entropy(input=x, target=target)

        return x, loss

    def sample(
        self, 
        x: torch.Tensor,
        max_tokens: int, 
        stop_seq: list[int] = [], 
        t: float = 0.5, 
        stream: bool = True
    ): 
        sampled = []
        for _ in range(max_tokens): 
            y = self.__call__(x, target=None)[0]
            y /= t
            probs = F.softmax(y, dim=-1)
            sample = torch.multinomial(probs, num_samples=1)
            if sample in stop_seq: 
                break

            sampled.append(sample)
            x = torch.tensor(x[1:].tolist() + [sample])

            if stream: 
                print(self.r_vocab_table[sample] , end="")

        return sampled 

    def _load(self, ckpt_path: Path): 
        ckpt = torch.load(ckpt_path)
        self.load_state_dict(ckpt["model"])

    @classmethod
    def from_pretrained(cls, ckpt_path: Path) -> Self:
        ckpt = torch.load(ckpt_path)
        model = cls(
            config=ModelConfig(**ckpt['model_cfg']), 
            vocab_table=ckpt['vocab_table']
        )
        model.load_state_dict(ckpt['model'])

        return model 

    def configure_optimizer(self, lr: float, weight_decay: float): 
        """
        from paper: "...R is a weight decay penalty applied only to the weights 
        of the neural network and to the C matrix, not to the biases." 
        """
        decay, no_decay = [], [] 
        for name, param in self.named_parameters(): 
            if 'bias' in name: 
                no_decay.append(param)
            else: 
                decay.append(param)

        optimizer = torch.optim.SGD(params=[
            {'params': decay, 'weight_decay': weight_decay}, 
            {'params': no_decay, 'weight_decay': 0}
        ], lr=lr)

        return optimizer 
    