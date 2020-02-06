#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import sys

from tokens import *
from vocabulary import Vocabulary
from accuracy import modify, accuracy
import ner


def test(
  model,
  word_vocab: Vocabulary, 
  char_vocab: Vocabulary, 
  tag_vocab : Vocabulary, 
  sentences : list,
  to_stdout : bool=True,
):
  device = torch.device(model._device)

  """
  # check test_dir whether the type is str of list.
  if type(test_dir) == str:
    sentences = open(test_dir, "r").readlines()
    if test_dir[-4:] == ".iob":
      sentences = [
        [
          delimiter.join(chunk.split(delimiter)[:-1])
          for chunk in sentence
        ]
        for sentence in sentences
      ]

  elif type(test_dir) == list:
    sentences = test_dir
  else:
    raise Exception("test_dir must be str or list, not {}".format(type(test_dir)))
  """

  ret = []
  for words in sentences:
    if len(words) <= 0:
      ret.append([])
      continue

    """
    if type(test_dir) is str:
      ids = [
        word_vocab.w2i[w] 
        if w in word_vocab.w2i.keys() else word_vocab.w2i[UNK]
        for w in sentence #.split()
      ]
    else:
      ids = sentence
    """

    #char_lens = [torch.LongTensor(len(w)).view(1, -1).to(device) for w in words]
    chars = [ torch.LongTensor(char_vocab.toIds(list(w))).view(1, -1).to(device) for w in words]
    
    #word_lens = torch.LongTensor(len(words)).to(device)
    words = torch.LongTensor(word_vocab.toIds(words)).view(1, -1).to(device)
  
    score, path = model.viterbi(words, chars, tag_vocab, None, None)

    path = [tag_vocab.i2t[i] for i in path]

    path = [(i, t) for i, t in enumerate(path)]
    path = [t for i, t in modify(path)]

    if to_stdout:
      print(" ".join(path))

    ret.append(path)

  return ret


if __name__ == "__main__":
  import pickle
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("-m", "--model_path", type=str, default=None, help="model path")
  parser.add_argument("-t", "--test_path", type=str, default=None, help="test path")
  parser.add_argument("-v", "--vocab_path", type=str, default=None,  help="vocabulary path")

  parser.add_argument("--hidden_size", type=int, default=256, help="hidden size")
  parser.add_argument("--delimiter", type=str, default="/", help="delimiter")

  args = parser.parse_args()

  for loc in (args.test_path, args.vocab_path, args.model_path):
    if loc is None:
      raise Exception("{} is None".format(loc))

  word_vocab, char_vocab, tag_vocab = pickle.load(open(args.vocab_path, "rb"))

  model = ner.BiLSTMCRF(
    len(word_vocab.w2i.keys()),
    len(char_vocab.w2i.keys()),
    args.hidden_size,
    len(tag_vocab.t2i.keys()),
    dropout=0.0,
  )

  model.load_state_dict(torch.load(args.model_path, map_location=lambda x, loc: x))
  model._device = "cpu"
  model.eval()

  sentences = [
    [ args.delimiter.join(chunk.split(args.delimiter)[:-1]) for chunk in sentence.split() ]
    for sentence in open(args.test_path).readlines()
  ]

  hyp = test(
    model, 
    word_vocab, 
    char_vocab,
    tag_vocab, 
    sentences,
    to_stdout=False,
  )

  ref = [
    [ chunk.split(args.delimiter)[-1] for chunk in sentence.split() ]
    for sentence in open(args.test_path).readlines()
  ]

  accuracy(ref, hyp)

