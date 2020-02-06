import torch
from torch.autograd import Variable
from typing import Tuple, List, Dict

BOS = "<S>"
EOS = "</S>"
UNK = "<UNK>"
PAD = "<PAD>"

O_label = "O"
START = "<START>"
STOP = "<STOP>"

Pair = List[List[str]]
Pairs = List[Pair]

Words = List[List[int]]
States = List[Variable]
Probs = List[float]
Unks = List[List[int]]
