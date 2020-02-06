#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import numpy as np

from tokens import *
from util import getOrder, getOrderOrig
from vocabulary import Vocabulary


class BiLSTMCRF(nn.Module):
  def __init__(
    self,
    vocab_word_size : int,
    vocab_char_size : int,
    hidden_size: int,
    tag_size   : int,
    dropout    : float=0.2,
  ):
    super(BiLSTMCRF, self).__init__()

    self.vocab_word_size = vocab_word_size
    self.vocab_char_size = vocab_char_size
    self.tag_size = tag_size
    self.hidden_char_size = hidden_size // 4
    self.hidden_size = hidden_size 
    self.dropout = dropout

    self.num_layers = 1 
    self.num_directions = 2

    # embedding layer
    self.word_embedding = nn.Embedding(
      num_embeddings=self.vocab_word_size, 
      embedding_dim=self.hidden_size,
    )

    self.char_embedding = nn.Embedding(
      num_embeddings=self.vocab_char_size, 
      embedding_dim=self.hidden_char_size,
    )

    # BiLSTM
    self.word_bilstm = nn.LSTM(
      input_size=self.hidden_char_size * 2 + self.hidden_size,
      hidden_size=self.hidden_size,
      num_layers=self.num_layers,
      bidirectional=True,
      batch_first=True,
    )

    self.char_bilstm = nn.LSTM(
      input_size=self.hidden_char_size,
      hidden_size=self.hidden_char_size,
      num_layers=1,
      bidirectional=True,
      batch_first=True,
    )

    # hidden to tag (size)
    self.h2t = nn.Linear(
      in_features=self.hidden_size * 2,
      out_features=self.tag_size,
    )

    # CRF
    self.transitions = nn.Parameter(
      torch.randn(self.tag_size, self.tag_size),
    )

  # initialize all parameters
  def initParams(
    self,
    _min: float=-0.1,
    _max: float=0.1,
  ):
    for param in self.parameters():
      param.data.uniform_(_min, _max)

  # prepare tensor for LSMT's states and memory cells
  def initState(
    self,
    batch_size    : int,
    hidden_size   : int,
    num_layers    : int,
    num_directions: int=2,
  ):
    return torch.zeros(
      num_layers * num_directions,
      batch_size,
      hidden_size,
    )

  def encode_chars(
    self,
    embedded: torch.FloatTensor,
    lens    : list=None,
  ):
    batch_size = embedded.size(0)
    device = torch.device(self._device)

    if lens is not None:
      lens, order_sorted = getOrder(lens)
      order_orig = getOrderOrig(order_sorted)

      order_sorted = order_sorted.to(device)
      order_orig = order_orig.to(device)

      embedded = embedded.index_select(0, order_sorted)

    # ((1 x 2) x b x h)
    h = self.initState(batch_size, self.hidden_char_size, 1).to(device)
    c = self.initState(batch_size, self.hidden_char_size, 1).to(device)

    if lens is not None:
      embedded = rnn_utils.pack_padded_sequence(embedded, lens, True)
      hs, (h, c) = self.char_bilstm(embedded, (h, c))
      hs = rnn_utils.pad_packed_sequence(hs, True)[0]
    else:
      hs, (h, c) = self.char_bilstm(embedded, (h, c))

    h = torch.cat([h[0], h[1]], dim=-1)

    if lens is not None:
      h = h.index_select(0, order_orig)

    return h

  # encode a sentence into hidden representations
  def encode(
    self,
    embedded: torch.FloatTensor,
    lens     : list=None,
  ):
    batch_size = embedded.size(0)
    device = torch.device(self._device)

    # ((1 x 2) x b x h)
    h = self.initState(batch_size, self.hidden_size, self.num_layers).to(device)
    c = self.initState(batch_size, self.hidden_size, self.num_layers).to(device)

    if lens is not None:
      embedded = rnn_utils.pack_padded_sequence(embedded, lens, True)
      hs, (h, c) = self.word_bilstm(embedded, (h, c))
      hs = rnn_utils.pad_packed_sequence(hs, True)[0]
    else:
      hs, (h, c) = self.word_bilstm(embedded, (h, c))

    logits = self.h2t(hs) # (b x l x h) -> (b x l x t)

    return logits, (h, c)

  def log_sum_exp(
    self,
    vec         : torch.Tensor,
    size_average: bool=True,
  ):
    batch_size = vec.size(0)
    
    score = torch.gather(vec, -1, torch.max(vec, dim=-1)[1].unsqueeze(-1)).squeeze()

    score = score + torch.log(torch.sum(
        torch.exp(vec - score.unsqueeze(-1).expand_as(vec)),
        dim=-1,
      )
    )

    if size_average is True:
      score = torch.sum(score)
      score /= batch_size

    return score
    
  # calculate a score by forward algorithm
  def getForwardScore(
    self,
    logits   : torch.Tensor,
    tag_vocab: Vocabulary,
    mask     : torch.Tensor,
    lens     : list,
  ):
    batch_size = logits.size(0)
    device = torch.device(self._device)

    start = tag_vocab.t2i[START]
    stop  = tag_vocab.t2i[STOP]

    alpha = torch.from_numpy(np.full((batch_size, self.tag_size), -10000.)).float()
    alpha[:, start] = 0

    alpha = alpha.to(device)

    alphas = []

    for i, logit in enumerate(logits.transpose(0, 1)):
      alpha_t = []

      for tag in range(self.tag_size):
        emit_score = logit[:, tag].unsqueeze(-1).expand_as(alpha)
        trans_score = self.transitions[tag].unsqueeze(0).expand_as(alpha)

        _sum = (alpha + emit_score + trans_score).masked_fill(mask[:, i].contiguous().view(-1, 1).expand_as(alpha), 0)
        alpha_t.append(self.log_sum_exp(alpha + emit_score + trans_score, False))

      alpha_t = torch.cat(alpha_t).view(self.tag_size, batch_size).transpose(0, 1)

      alphas.append(alpha_t)
      alpha = alpha_t

    alpha = torch.cat(
      [
        alphas[lens[i] - 1][i]
        for i in range(len(lens))
      ]
    )
    alpha = alpha.view(batch_size, self.tag_size)

    alpha = alpha + self.transitions[stop].unsqueeze(0).expand_as(alpha)
    alpha = self.log_sum_exp(alpha, False)

    return alpha

  # calculate a score with tags
  def getGoldScore(
    self,
    logits   : torch.Tensor,
    tags     : torch.Tensor, # (b x l)
    tag_vocab: Vocabulary,
    mask     : torch.Tensor,
    lens     : list,
  ):
    batch_size = logits.size(0)
    device = torch.device(self._device)

    start = tag_vocab.t2i[START]
    stop  = tag_vocab.t2i[STOP]

    _start = torch.LongTensor(np.full((batch_size, 1), start)).to(device)
    _zeros = (torch.zeros(batch_size, 1)).byte().to(device)

    # (b x l) -> ((l + 1) x b)
    tags = torch.cat([_start, tags], dim=-1).transpose(0, 1)
    mask = torch.cat([_zeros, mask], dim=-1).transpose(0, 1)

    score = 0.

    for i, logit in enumerate(logits.transpose(0, 1)):
      _score = self.transitions[tags[i + 1], tags[i]] + \
               torch.gather(logit, -1, tags[i + 1].unsqueeze(-1)).view(-1)

      score += _score.masked_fill(mask[i], 0)
    
    _stop = torch.LongTensor([stop] * batch_size).to(device)

    _tag = torch.LongTensor(
      [
        tags[lens[i]][i] 
        for i in range(len(lens))
      ],
    )
      
    score += self.transitions[_stop, _tag]

    return score

  # apply viterbi algorithm for one sentence
  def viterbi(
    self,
    words : torch.Tensor,
    chars : torch.Tensor,
    tag_vocab: Vocabulary,
    word_lens: list=None,
    char_lens: list=None,
  ):
    embedded_chars = []
    for c in chars:
      h = self.encode_chars(self.char_embedding(c))
      embedded_chars += [h.unsqueeze(1)]

    embedded_chars = torch.cat(embedded_chars, dim=1)

    embedded_words = self.word_embedding(words)
    embedded = torch.cat([embedded_chars, embedded_words], dim=-1)

    logits, _ = self.encode(embedded)

    back_pointers = []

    init_vars = torch.from_numpy(np.full((1, self.tag_size), -10000.)).float()
    init_vars[0][tag_vocab.t2i[START]] = 0

    fwd_var = init_vars

    for logit in logits.transpose(0, 1):
      p_t = [] # for back pointer at t
      v_t = [] # for viterbi at t

      for tag in range(self.tag_size):
        next_tag_var = fwd_var + self.transitions[tag]
        best_tag_id = torch.max(next_tag_var, dim=-1)[1]

        p_t.append(best_tag_id.data.tolist())
        v_t.append(next_tag_var[0][best_tag_id].view(-1))

      fwd_var = (torch.cat(v_t) + logit).view(1, -1)
      back_pointers.append(p_t)

    end_var = fwd_var + self.transitions[tag_vocab.t2i[STOP]]

    best_tag_id = torch.max(end_var, dim=-1)[1].tolist()[0]
    path_score = end_var[0][best_tag_id].tolist()

    best_path = [best_tag_id]
    for p_t in reversed(back_pointers):
      best_tag_id = p_t[best_tag_id][0]
      best_path.append(best_tag_id)

    best_path = best_path[::-1][1:]

    return path_score, best_path

  # forward one step
  def forward(
    self,
    words    : torch.Tensor,
    chars    : list,
    tags     : torch.Tensor,
    tag_vocab: Vocabulary,
    word_mask: torch.Tensor,
    word_lens: list=None,
    char_lens: list=None,
  ):
    embedded_chars = []
    for c, l in zip(chars, char_lens):
      h = self.encode_chars(self.char_embedding(c), l) 
      embedded_chars += [h.unsqueeze(1)]

    embedded_chars = torch.cat(embedded_chars, dim=1)

    embedded_words = self.word_embedding(words)
    embedded = torch.cat([embedded_chars, embedded_words], dim=-1)

    logits, _ = self.encode(embedded, word_lens)

    f_score = self.getForwardScore(logits, tag_vocab, word_mask, word_lens)
    g_score = self.getGoldScore(logits, tags, tag_vocab, word_mask, word_lens)

    # -log( P(x) / Z ) = -( log(P(x)) - log(Z) )
    loss = -(g_score - f_score)
    loss = torch.mean(loss)

    return loss


