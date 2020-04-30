# MADDPG in Google Research Football
MADDPG implementation in Google Research Football

## Requirements
* My [fork](https://github.com/chrisyrniu/football) version of Google Research Football
* OpenAI Gym
* OpenAI baselines
* PyTorch
* Tensorboard and [tensorboardX](https://github.com/lanpa/tensorboardX)

## Run Training
`sh run_train.sh`

## Run Testing
`sh run_test.sh`

## Check Training Processing and Results
* Use tensorboardx
* Use plot_script.py:

`python plot_script.py saved/ name Reward`

`python plot_script.py saved/ name Steps-Taken`

## Acknowledgement
This repository is revised from [MADDPG-PyTorch](https://github.com/shariqiqbal2810/maddpg-pytorch)


