#!/bin/bash

python run.py --output_dir=./saved_models \
  --model_type=roberta --tokenizer_name=microsoft/codebert-base --model_name_or_path=microsoft/codebert-base \
  --train_data_file=../CodeXGLUE-Defect-detection/dataset/train.jsonl --eval_data_file=../CodeXGLUE-Defect-detection/dataset/valid.jsonl --test_data_file=../CodeXGLUE-Defect-detection/dataset/test.jsonl \
  --epoch 5 --block_size 400 --train_batch_size 32 --eval_batch_size 64 --learning_rate 2e-5 --max_grad_norm 1.0 \
  --do_train --evaluate_during_training \
  --seed 123456
