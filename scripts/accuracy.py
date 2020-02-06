#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import sys
from typing import List, Tuple

from tokens import *


def modify(line: List[Tuple[int, str]]):
  for i, chunk in enumerate(line):

    if chunk[1] in (START, STOP):
      line[i] = (i, "O")
      continue

    if "-" in chunk[1]:
      body, tail = chunk[1].split("-")

      if tail == "I":
        if i <= 0 or not "-" in line[i - 1][1]:
          line[i] = (i, "O")
          continue

        _body, _tail = line[i - 1][1].split("-")

        if body != _body:
          line[i] = (i, "O")

  return line


def compress(lines: List[List[str]]):
  ret = []

  for line in lines:
  
    line = [ (i, tag) for i, tag in enumerate(line) ]

    line = modify(line)

    ret_t = []
    for i, t_i in line:
      if t_i == "O":
        ret_t += [(i, i, t_i)]

      elif t_i[-2:] == "-B":

        j = i
        for k, t_k in line[i + 1:]:
          if t_k[-2:] != "-I":
            break
          else:
            j = k

        ret_t += [(i, j, t_i[:-2])]

    ret += [ret_t]

  return ret


def accuracy(
  ref: List[List[str]],
  hyp: List[List[str]],
  delimiter: str="/",
  quiet: bool=False,
):
  ref = compress(ref)
  hyp = compress(hyp) 

  tp = tn = fp = fn = 0
  eps = 1e-9

  for line_r, line_h in zip(ref, hyp):
    for h in line_h:
      if h[-1] == "O":
        if h in line_r:
          tn += 1
        else:
          fn += 1
      else:
        if h in line_r:
          tp += 1
        else:
          fp += 1

  precision = tp / (tp + fp + eps)
  recall = tp / (tp + fn + eps)
  f1_score = 2 * precision * recall / ( precision + recall + eps)

  if not quiet:
    print("precision: ", precision)
    print("recall   : ", recall)
    print("f1_score : ", f1_score)

  return precision, recall, f1_score
        

def main():

  if len(sys.argv) < 3:
    raise Exception("Usage: python accuracy.py reference hypothesis delimiter")

  for i in (1, 2):
    if os.path.exists(sys.argv[i]) is False:
      raise Exception("{} does not exist".format(sys.argv[i]))

  if len(sys.argv) < 4:
    delimiter = "/"
  else:
    delimiter = sys.argv[3]

  ref = compress(
    [
      [x.rstrip().split(delimiter)[-1] for x in line.split()]
      for line in open(sys.argv[1]).readlines()
    ]
  )

  hyp = compress( 
    [
      line.split()
      for line in open(sys.argv[2]).readlines()
    ]
  )

  accuracy(ref, hyp, delimiter)


if __name__ == "__main__":
  main()
