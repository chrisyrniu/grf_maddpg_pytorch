#!/bin/bash

python -u test.py \
  academy_3_vs_1_with_keeper \
  models \
  --run_num 1 \
  --n_episodes 200 \
  --episode_length 200 \
  --n_controlled_lagents 3 \
  --n_controlled_ragents 0 \
  --reward_type checkpoints \
  --render \
  | tee test.log

#  --incremental
