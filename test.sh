#!/bin/bash

mkdir -p models

python ./scripts/test.py \
  --test_path ./data/test.iob \
  --model_path ./models/temp.model \
  --vocab_path ./models/temp.vocab \
  --hidden_size 64
