#!/bin/bash

python -u main.py \
  academy_3_vs_1_with_keeper \
  maddpg \
  --seed 1 \
  --n_rollout_threads 1 \
  --n_training_threads 6 \
  --n_controlled_lagents 3 \
  --n_controlled_ragents 0 \
  --buffer_length 1000000 \
  --n_episodes 200000 \
  --n_exploration_eps 150000 \
  --episode_length 200 \
  --steps_per_update 100 \
  --batch_size 1024 \
  --init_noise_scale 0.3 \
  --final_noise_scale 0.0 \
  --save_interval 1000 \
  --hidden_dim 64 \
  --lr 0.01 \
  --tau 0.01 \
  --agent_alg MADDPG \
  --adversary_alg MADDPG \
  --reward_type checkpoints \
  | tee train.log

#  --gpu \
#  --render \
