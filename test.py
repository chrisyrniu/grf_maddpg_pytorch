import argparse
import torch
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from algorithms.maddpg import MADDPG
from utils.multi_agent_env import MultiAgentEnv


def run(config):
    model_path = (Path('./models') / config.env_id / config.model_name /
                  ('run%i' % config.run_num))
    if config.incremental is not None:
        model_path = model_path / 'incremental' / ('model_ep%i.pt' %
                                                   config.incremental)
    else:
        model_path = model_path / 'model.pt'

    maddpg = MADDPG.init_from_save(model_path)
    env = MultiAgentEnv(config.env_id, config.n_controlled_lagents, config.n_controlled_ragents, config.reward_type, config.render)
    maddpg.prep_rollouts(device='cpu')

    for ep_i in range(config.n_episodes):
        print("Episode %i of %i" % (ep_i + 1, config.n_episodes))
        obs = env.reset()
        for t_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = [Variable(torch.Tensor(obs[i]).view(1, -1),
                                  requires_grad=False)
                         for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_actions = maddpg.step(torch_obs, explore=False)
            # convert actions to numpy arrays
            actions = [ac.data.numpy().flatten() for ac in torch_actions]
            obs, rewards, dones, infos = env.step(actions)
            if dones[0]:
            	break
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of model")
    parser.add_argument("--run_num", default=1, type=int)
    parser.add_argument("--incremental", default=None, type=int,
                        help="Load incremental policy from given episode " +
                             "rather than final policy")
    parser.add_argument("--n_episodes", default=200, type=int)
    parser.add_argument("--episode_length", default=200, type=int)
    parser.add_argument("--render", default=False, type=bool)
    parser.add_argument("--n_controlled_lagents", default=3, type=int)
    parser.add_argument("--n_controlled_ragents", default=0, type=int)
    parser.add_argument("--reward_type",
                        default="scoring", type=str,
                        choices=['scoring', 'checkpoints'])

    config = parser.parse_args()

    run(config)