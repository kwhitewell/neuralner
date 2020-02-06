#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
from torch.autograd import Variable

from tokens import *
from vocabulary import Vocabulary


def loadData(
  filename  : str,
  word_vocab: Vocabulary,
  tag_vocab : Vocabulary,
  delimiter : str="/",
):
  words_list = []
  tags_list  = []

  for line in open(filename, "r").readlines():
    
    words = []
    tags = []
    
    for chunk in line.rstrip().split(" "):
      chunk = chunk.split(delimiter)
      assert len(chunk) == 2

      words += [chunk[0]]
      tags  += [chunk[1]]
      
    words_list += [words]
    tags_list  += [tags]

  return words_list, tags_list


def fill(
  batch: list,
  lens : list,
  pad  : int=-1,
):
  len_max = max(lens)

  return [
    batch[i] + [pad] * (len_max - lens[i])
    for i in range(len(lens))
  ]


def getMask(lens: list):
  len_max = max(lens)
  
  return torch.ByteTensor(
    [
      [0] * lens[i] + [1] * (len_max - lens[i])
      for i in range(len(lens))
    ]
  )


def getMinibatch(
  batch_words: list,
  batch_tags : list,
  word_vocab : Vocabulary,
  char_vocab : Vocabulary,
  tag_vocab  : Vocabulary,
  device     : torch.device,
):
  sorted_pairs = [
    [s, t]
    for s, t in sorted(
      [ [_s, _t] for _s, _t in zip(batch_words, batch_tags) ],
      key=lambda x: len(x[0]), reverse=True,
    )
  ]

  batch_words = [x[0] for x in sorted_pairs]
  batch_word_lens = [len(x) for x in batch_words]
  batch_tags = [x[1] for x in sorted_pairs] 

  max_len_batch = max(batch_word_lens)
  batch_size = len(batch_words)

  _batch_chars = []
  batch_char_lens = []

  for j in range(max_len_batch):
    chars = []
    for i in range(batch_size):
      if j >= len(batch_words[i]):
        chars += [[char_vocab.w2i[EOS]]]
      else:
        chars += [char_vocab.toIds(list(batch_words[i][j]))]

    _batch_chars += [chars]
    batch_char_lens += [[len(c) for c in chars]]

  batch_words = [word_vocab.toIds(s) for s in batch_words]
  batch_tags = [tag_vocab.toTagIds(s) for s in batch_tags]

  batch_chars = []
  for c, l in zip(_batch_chars, batch_char_lens):
    batch_chars += [
      torch.LongTensor(
        fill(c, l, char_vocab.w2i[EOS])
      ).to(device)
    ]

  batch_words = torch.LongTensor(
    fill(
      batch_words,
      batch_word_lens,
      word_vocab.w2i[EOS],
      #word_vocab.w2i[PAD],
    )
  ).to(device)

  batch_tags = torch.LongTensor(
    fill(
      batch_tags,
      batch_word_lens,
      tag_vocab.t2i[STOP],
    )
  ).to(device)

  batch_word_mask = getMask(batch_word_lens).to(device)

  #return batch_words, batch_tags, batch_masks, batch_word_lens
  return batch_words, batch_word_lens, batch_word_mask, batch_chars, batch_char_lens, batch_tags


def getOrder(lens: list):
  lens = torch.LongTensor(lens)
  sorted_lens, order = torch.sort(lens, descending=True)
  return sorted_lens, order


def getOrderOrig(
  order: torch.LongTensor,
):
  pairs = [
    (o1, o2)
    for o1, o2 in zip(order.tolist(), list(range(len(order))))
  ]
  pairs.sort(key=lambda x: x[0])

  order_orig = [o2 for o1, o2 in pairs]
  return torch.LongTensor(order_orig)



