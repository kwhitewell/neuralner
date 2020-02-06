#!/usr/bin/env python
# -*- coding:utf-8 -*- 

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import time
import numpy as np

from tokens import *
from vocabulary import Vocabulary, createVocabularies
from util import loadData, getMinibatch
import ner

from test import test
from accuracy import accuracy


np.random.seed(0)
torch.manual_seed(0)


def train(
  model,
  optimizer : optim,
  word_vocab: Vocabulary,
  char_vocab: Vocabulary,
  tag_vocab : Vocabulary,
  args,
):
  epoch_size = args.epoch_size
  batch_size = args.batch_size

  train_words, train_tags = loadData(args.train_path, word_vocab, tag_vocab, args.delimiter)
  dev_words, dev_tags = loadData(args.dev_path, word_vocab, tag_vocab, args.delimiter)

  train_size = len(train_words)
  device = torch.device(model._device)

  print(model)
  model.f1 = -1

  p_all = []
  r_all = []
  f1_all = []

  for epoch in range(1, epoch_size + 1):

    indexes = np.random.permutation(train_size)
    epoch_loss = 0.

    model.train()

    for batch_i in range(0, train_size, batch_size):
      idx = indexes[batch_i:batch_i + batch_size]

      # prepare minibatch
      batch_words, batch_word_lens, batch_word_mask, batch_chars, batch_char_lens, batch_tags = getMinibatch(
        [train_words[i] for i in idx],
        [train_tags[i] for i in idx],
        word_vocab,
        char_vocab,
        tag_vocab,
        device,
      )

      """
      for i, words in enumerate(batch_words.tolist()[:3]):
        print(" ".join(word_vocab.toTokens(words)[:batch_word_lens[i]]))
      cs = [[] for _ in range(3)]
      for j in range(len(batch_chars)):
        for i in range(3):
          cs[i] += ["".join(char_vocab.toTokens(batch_chars[j][i].tolist())[:batch_char_lens[j][i]])]

      for i in range(3):
        print(" ".join(cs[i]))
      """

      f_start = time.time()

      # forward
      loss = model(
        batch_words,
        batch_chars,
        batch_tags,
        tag_vocab,
        batch_word_mask,
        batch_word_lens,
        batch_char_lens,
      )

      f_end = time.time()
      b_start = time.time()

      # backward and update parameters
      optimizer.zero_grad()
      loss.backward()
      if model.clipping is not None and model.clipping > 0:
        nn.utils.clip_grad_norm_(model.parameters(), model.clipping)
      optimizer.step()

      b_end = time.time()

      epoch_loss += loss.tolist()

      print("epoch: {:>3d}, batch: {:>4d}, loss: {:10.4f}, forward: {: 2.2f}, backward: {: 2.2f}".format(
        epoch,
        batch_i // batch_size + 1,
        loss.tolist(),
        f_end - f_start,
        b_end - b_start,
      ))

    print("finished epoch: {}, epoch loss: {}\n".format(
      epoch,
      epoch_loss,
    ))

    model.eval()

    # calc accuracy on dev
    p, r, f1 = calcAccuracy(
      model, 
      word_vocab, 
      char_vocab,
      tag_vocab, 
      dev_words, 
      dev_tags,
    )

    p_all += [p]
    r_all += [r]
    f1_all += [f1]

    # if the optimizer is SGD, 
    #   then scheduling the initial learning rate by
    #   lr = initial_lr / ( 1.0 + 0.05 * epoch_number )
    if "SGD" in optimizer.__str__():
      lr = model.lr / (1.0 + 0.05 * epoch)

      for param_group in optimizer.param_groups:
        param_group["lr"] = lr

      print("set the learning rate of {}\n".format(lr))
      
    # save the model parameters
    if model.f1 < f1 and args.model_path is not None:
      torch.save(
        model.state_dict(),
        args.model_path,
      )
    model.f1 = max(model.f1, f1)

  print("best f1-score: {}".format(model.f1))

  """
  # plot
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  print("p_all: ", p_all)
  print("r_all: ", r_all)
  print("f1_all: ", f1_all)

  plt.plot(range(epoch_size), p_all, "bo", markersize=3.5)
  plt.plot(range(epoch_size), p_all, "b-", markersize=2, label="precision")
  plt.plot(range(epoch_size), r_all, "ro", markersize=3.5)
  plt.plot(range(epoch_size), r_all, "r-", markersize=2, label="recall")
  plt.plot(range(epoch_size), f1_all, "go", markersize=3.5)
  plt.plot(range(epoch_size), f1_all, "g-", markersize=2, label="f1_score")

  plt.xlabel("Epoch")
  plt.legend(loc="lower right")
  plt.savefig("temp.png")
  """


def calcAccuracy(
  model,
  word_vocab   : Vocabulary, 
  char_vocab   : Vocabulary, 
  tag_vocab    : Vocabulary, 
  dev_words    : List[List[int]],
  dev_tags     : List[List[int]],
):
  _device = model._device
  model._device = "cpu"
  model = model.to(torch.device(model._device))

  hyp = test(
    model, 
    word_vocab, 
    char_vocab,
    tag_vocab, 
    dev_words, 
    to_stdout=False,
  )
  for h in hyp[:5]:
    print(" ".join(h))

  """
  ref = [
    tag_vocab.toTagTokens(t)
    for t in dev_tags
  ]
  """
    
  p, r, f1 = accuracy(dev_tags, hyp)
  print()

  model._device = _device
  modle = model.to(torch.device(model._device))

  return p, r, f1 


if __name__ == "__main__":
  import os
  import pickle
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("-g", "--gpu", type=int, default=-1, help="gpu id")

  parser.add_argument("--train_path", type=str, default=None, help="train path")
  parser.add_argument("--dev_path", type=str, default=None, help="dev path")
  parser.add_argument("--model_path", type=str, default=None, help="save path")
  parser.add_argument("--vocab_path", type=str, default=None, help="vocab path")

  parser.add_argument("-o", "--optimizer", type=str, default="Adam", help="optimizer")
  parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
  parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
  parser.add_argument("--weight_decay", type=float, default=1e-6, help="weight decay")
  parser.add_argument("--clipping", type=float, default=1.0, help="clipping")

  parser.add_argument("-b", "--epoch_size", type=int, default=50, help="epoch size")
  parser.add_argument("-e", "--batch_size", type=int, default=10, help="epoch size")

  parser.add_argument("--vocab_size", type=int, default=25000, help="vocab size")
  parser.add_argument("--hidden_size", type=int, default=256, help="hidden size")
  parser.add_argument("--dropout", type=float, default=0.2, help="dropout")

  parser.add_argument("--delimiter", type=str, default="/", help="delimiter")
  args = parser.parse_args()

  if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)

  # raise exception if a file does not exist
  for filename in (args.train_path, args.dev_path, args.vocab_path):
    if filename is None:
      raise Exception("{} is None.".format(filename))

  # create/load vocabularies
  if os.path.exists(args.vocab_path):
    word_vocab, char_vocab, tag_vocab = pickle.load(open(args.vocab_path, "rb"))
    print("loaded {}".format(args.vocab_path))
  else:
    word_vocab, char_vocab, tag_vocab = createVocabularies(
      filename=args.train_path,
      delimiter=args.delimiter,
    )
    print("created {}".format(args.vocab_path))

    with open(args.vocab_path, "wb") as f:
      pickle.dump([word_vocab, char_vocab, tag_vocab], f)

  print("vocab (word) size: ", len(word_vocab.w2i))
  print("vocab (tag)  size: ", len(tag_vocab.t2i))
  print("\t", " ".join(list(tag_vocab.t2i.keys())))
  print()

  # initialize model params
  model = ner.BiLSTMCRF(
    len(word_vocab.w2i.keys()),
    len(char_vocab.w2i.keys()),
    args.hidden_size,
    len(tag_vocab.t2i.keys()),
    args.dropout,
  )

  # change the mode for training.
  model.train()

  model.delimiter = args.delimiter
  model.clipping = args.clipping

  model._device = "cuda:{}".format(args.gpu) if args.gpu >= 0 else "cpu"
  model = model.to(torch.device(model._device))

  if args.optimizer == "SGD":
    optimizer = optim.SGD(
      model.parameters(),
      lr=args.lr,
      weight_decay=args.weight_decay,
      momentum=args.momentum,
    )
  elif args.optimizer == "Adam":
    optimizer = optim.Adam(
      model.parameters(),
      lr=args.lr,
      weight_decay=args.weight_decay,
    )

  # start training.
  train(
    model,
    optimizer,
    word_vocab,
    char_vocab,
    tag_vocab,
    args,
  )

