#!/bin/bash

mkdir -p models

python ./scripts/train.py \
  --train_path ./data/train.iob \
  --dev_path ./data/dev.iob \
  --model_path ./models/temp.model \
  --vocab_path ./models/temp.vocab \
  --vocab_size 1000 \
  --hidden_size 64 \
  --dropout 0.1 \
  --optimizer Adam \
  --lr 2e-4 \
  --epoch_size 1000 \
  --batch_size 10
