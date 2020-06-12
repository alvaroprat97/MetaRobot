"""
Launcher for experiments with PEARL
"""
import sys
sys.path.append("")

import os
import pathlib
import numpy as np
import click
import json
import torch
from tensorboardX import SummaryWriter
import datetime

from envs.metaENV import ENV, VisualiserWrapper
from envs.MetaPeg2D import WINDOW_X, WINDOW_Y, ORIGIN, PEG_DEPTH
# from backend.envs.wrappers import NormalizedBoxEnv
from backend.torch.PEARL.policies import TanhGaussianPolicy
from backend.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from backend.torch.PEARL.sac import PEARLSoftActorCritic
from backend.torch.PEARL.agent import PEARLAgent
from backend.launchers.launcher_util import setup_logger
import backend.torch.pytorch_util as ptu
from configs.default import default_config

print("Selecting Default variant for notebook compatiblity ... \n")

def run_policy(agent, task_idx, framework = 'PEARL'):
    env = VisualiserWrapper(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False, visible = True)
    env.show = True
    env.task_idx = task_idx
    env.agent = agent
    env.run_policy(agent)

def experiment(variant):

    log_dir = 'runs/{}_PEARL_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                        "Peg2DRobot")
    writer = SummaryWriter(logdir=log_dir)

    env = ENV()
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[128, 128, 128],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        print(f"Loading existing weights from {variant['path_to_weights']} ... \n")
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))
    else:
        assert variant['modality'] is 'train'
        
    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    if variant['modality'] is 'train':
        # run the algorithm
        if isinstance(writer, SummaryWriter):
            algorithm.writer = writer
        algorithm.train()
    elif variant['modality'] is 'test':
        algorithm.training_mode(False)
        run_policy(agent, task_idx=0, framework='PEARL')

if __name__ == "__main__":
    variant = default_config
    experiment(variant)