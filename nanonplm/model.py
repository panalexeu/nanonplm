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
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config 
        self.embed = nn.Embedding(
            num_embeddings=self.config.vocab, 
            embedding_dim=self.config.embed, 
        )
        self.up_proj = nn.Linear(
            in_features=self.config.embed * self.config.n, 
            out_features=self.config.hidden, 
            bias=True 
        ) 
        self.out_proj = nn.Linear(
            in_features=self.config.hidden, 
            out_features=self.config.vocab, 
            bias=True
        ) 

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
        x = F.tanh(self.up_proj(x)) 
        x = self.out_proj(x) 

        loss = None 
        if target is not None: 
            loss = F.cross_entropy(input=x, target=target)

        return x, loss
