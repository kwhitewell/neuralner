#!/usr/bin/env python
# -*- coding:utf-8 -*-

import copy
import numpy as np

from tokens import *

# used to create word_, label_, and tag_vocab
class Vocabulary:
  def __init__(
    self,
    use_unk: bool=True,
  ):
    self.w2i = {} # word to index
    self.w2c = {} # word to count
    self.n_words = 0
    self.use_unk = use_unk

  def create(
    self,
    words: list,
  ):
    for word in words:
      if not word in self.w2i.keys():
        self.w2i[word] = self.n_words
        self.w2c[word] = 1
        self.n_words += 1
      else:
        self.w2c[word] += 1

  def filter(
    self,
    freq: int,
  ):
    for word in (UNK, BOS, EOS):
      if word in self.w2i.keys():
        del self.w2i[word]
        del self.w2c[word]
        self.n_words -= 1

    self.filterWord(freq)

  def filterWord(
    self,
    freq: int=2,
  ):
    #w2i = copy.deepcopy(self.w2i)
    w2c = copy.deepcopy(self.w2c)

    self.w2i = {BOS: 0, EOS: 1}
    if self.use_unk:
      self.w2i[UNK] = len(self.w2i.keys())

    self.w2c = {}
    self.n_words = len(self.w2i.keys())

    #for word, count in sorted(w2c.items(), key=lambda x: x[1], reverse=True)[:size]:
    for word, count in sorted(w2c.items()):
      if count >= freq:
        self.w2i[word] = self.n_words
        self.w2c[word] = w2c[word]
        self.n_words += 1

  def createi2w(self):
    self.i2w = {
      value: key
      for key, value in self.w2i.items()
    }

  def toIds(
    self,
    words: list,
  ):
    return [
      self.w2i[w]
      if w in self.w2i.keys() else self.w2i[UNK]
      for w in words
    ]

  def toTagIds(
    self,
    words: list,
  ):
    return [
      self.t2i[w]
      for w in words
    ]

  def toTokens(
    self,
    ids: list,
  ):
    return [
      self.i2w[i]
      for i in ids
    ]

  def toTagTokens(
    self,
    ids: list,
  ):
    return [
      self.i2t[i]
      for i in ids
    ]


def createWordVocabulary(
  lines  : list,
  freq   : int=2,
  use_unk: bool=True,
):
  vocab = Vocabulary(use_unk)

  if lines is not None:
    vocab.create(lines)
    vocab.filter(freq)
    vocab.createi2w()

  return vocab


def createCharVocabulary(
  lines  : list,
  freq   : int=0,
  use_unk: bool=True,
):
  vocab = Vocabulary(use_unk)

  for line in lines:
    for w in line:
      for i in range(len(w)):
        c = w[i]

        if c in vocab.w2c.keys():
          vocab.w2c[c] += 1
        else:
          vocab.w2c[c] = 1

  vocab.filter(freq)
  vocab.createi2w()

  return vocab


def createTagVocabulary(tags: list=None):
  tag_vocab = Vocabulary(True)

  tag_vocab.t2i = { START: 0, STOP: 1, "O": 2}
  tag_vocab.n_tags = len(tag_vocab.t2i.keys())

  for tag in tags:
    for fix in ("B", "I"):
      #tag_vocab.t2i["{}-{}".format(fix, tag)] = tag_vocab.n_tags
      tag_vocab.t2i["{}-{}".format(tag, fix)] = tag_vocab.n_tags
      tag_vocab.n_tags += 1

  tag_vocab.i2t = {
    val: key
    for key, val in tag_vocab.t2i.items()
  }

  return tag_vocab

# create word and tag vocabulary
def createVocabularies(
  filename : str,
  delimiter: str="/",
):
  # create vocabulary for words
  words = []
  chars = []
  tags  = []

  for line in open(filename, "r").readlines():
    for chunk in line.rstrip().split():
      chunk = chunk.split(delimiter)
      assert len(chunk) == 2

      words += [chunk[0]]
      if "-" in chunk[1] and not chunk[1].split("-")[0] in tags:
        tags += [chunk[1].split("-")[0]]

      #for w in words:
      #  chars += [w.split()]

  word_vocab = createWordVocabulary(words)
  char_vocab = createCharVocabulary(words)
  tag_vocab  = createTagVocabulary(tags)

  return word_vocab, char_vocab, tag_vocab

