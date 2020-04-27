#!/bin/bash

python -u main.py \
  academy_3_vs_1_with_keeper \
  maddpg \
  --seed -1 \
  --n_rollout_threads 4 \
  --n_training_threads 6 \
  --n_controlled_lagents 3 \
  --n_controlled_ragents 0 \
  --buffer_length 1000000 \
  --n_episodes 200000 \
  --n_exploration_eps 45000 \
  --epoch_size 5 \
  --episode_length 100 \
  --steps_per_update 100 \
  --batch_size 1024 \
  --init_noise_scale 0.3 \
  --final_noise_scale 0.0 \
  --save_interval 500 \
  --hidden_dim 64 \
  --lr 0.005 \
  --tau 0.005 \
  --agent_alg MADDPG \
  --adversary_alg MADDPG \
  --reward_type checkpoints \
  | tee train.log

#  --gpu \
#  --render \
