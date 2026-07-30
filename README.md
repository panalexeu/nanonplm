### nanonplm

A PyTorch implementation of the architecture proposed in a 2003 paper (more than two decades old!): ["A Neural Probabilistic Language Model"](https://dl.acm.org/doi/epdf/10.5555/944919.944966).
I believe the shortest and best way to describe this architecture is: a neural improvement on n-gram models. I'd also argue it's the simplest possible *neural* LM to implement (with n-grams being the simplest statistical one).

The current implementation's only dependencies are PyTorch and requests. It defines data loaders for Brown, tiny-shakespeare, and Telegram chat exports (`data.py`), the model architecture (`model.py`), a tokenizer (`tokenizer.py`), and sampling/training scripts (`sample.py`, `train.py`). The overall structure of this repo is inspired by Andrej Karpathy's nanoGPT.

The implementation tries to stay as close to the paper as possible by:
1. Omitting weight decay for biases;
2. Using a training config resembling the one presented in the paper (Table 1, MLP9 config);
3. Using the paper's learning rate update rule;
4. Using true SGD.

**Usage**

Training:
```bash
python -m nanonplm.train --data_dir ./data --test_cutoff 0.9
```

Sampling:
```bash
python -m nanonplm.sample <ckpt_path>
```

**Results**

Training the model (1,847,266 params) with the default training script on tiny-shakespeare, on a test set of the last 29,783 unseen tokens, achieves a ppl value of 136.44 (vocab size: 13,986).
Sampling with the default sampling script (t=0.7) produces the following results, which look quite fun — I'd highlight how the model learned the structure of tiny-shakespeare (role:\ntext):

```text 
to be or not to us
o, if i come,
for the death, mysword,
not northumberland, the heart,
for my lord, and one's will;
to kam at the king as all a world,
this, therefore i do in the king
my very the daughter?

second vincentio:
what is their hands,
which alligator not't: the way for
in the henry, and
as the beetle
in the throne and myt
have, by is your brother,
all lord, good by the across
pray, and, let he turn
my all his ghostly:
i say i will, my king!

norfolk:
o, his great death,
and confess this haththough,
for then shall a sister
old'd to the word, and i would not so
...
```

This model checkpoint is stored under `./out/tiny_shk.pt`.