### nanonplm

A PyTorch implementation of the architecture proposed in a 2003 paper (more than two decades old!): ["A Neural Probabilistic Language Model"](https://dl.acm.org/doi/epdf/10.5555/944919.944966).
I believe the shortest and best way to describe this architecture is: a neural improvement on n-gram models. I'd also argue it's the simplest possible *neural* LM to implement (with n-grams being the simplest statistical one).

The current implementation's only dependencies are PyTorch and requests. It defines data loaders for Brown, tiny-shakespeare, and Telegram chat exports (`data.py`), the model architecture (`model.py`), a tokenizer (`tokenizer.py`), and sampling/training scripts (`sample.py`, `train.py`). The overall structure of this repo is inspired by Andrej Karpathy's nanoGPT.

The implementation tries to stay as close to the paper as possible by:
1. Omitting weight decay for biases;
2. Using a training config resembling the one presented in the paper (Table 1, MLP9 config);
3. Using the paper's learning rate update rule;
4. Using true SGD.

**Usage:**

Training:
```bash
python -m nanonplm.train --data_dir ./data --test_cutoff 0.9
```

Sampling:
```bash
python -m nanonplm.sample <ckpt_path>
```