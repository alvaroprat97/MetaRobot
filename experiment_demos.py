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
import pickle

from envs.metaENV import ENV, VisualiserWrapper
from envs.MetaPeg2D import WINDOW_X, WINDOW_Y, ORIGIN, PEG_DEPTH
from backend.torch.PEARL.policies import TanhGaussianPolicy
from backend.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, LatentGNNEncoder, NormalAux
from backend.torch.PEARL.sac import PEARLSoftActorCritic
from backend.torch.PEARL.agent import PEARLAgent
from backend.launchers.launcher_util import setup_logger
import backend.torch.pytorch_util as ptu
from configs.default import default_config

def run_policy(agent, task_idx, framework = 'PEARL'):
    env = VisualiserWrapper(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False, visible = True)
    env.show = True
    env.task_idx = task_idx
    env.agent = agent
    env.run_policy(agent)

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

def experiment(variant):

    log_dir = variant['util_params']['run_dir'] + "runs/{}_{}".format(datetime.datetime.now().strftime("%d_%H"),
                                        "PEARL")
    writer = SummaryWriter(logdir=log_dir)

    env = ENV(expert=False)
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

    # Demos
    with open(variant['demo_path'] + '.pickle', 'rb') as handle:
        demos = pickle.load(handle)

    # Encoder
    if not variant['GNN_encoder']:
        context_encoder = encoder_model(
            hidden_sizes=[128, 128, 128], 
            input_size=context_encoder_input_dim,
            output_size=context_encoder_output_dim,
        )
    else:
        context_encoder = LatentGNNEncoder(
                        input_dim= context_encoder_input_dim,
                        output_size = context_encoder_output_dim,
                        **variant['LatentGNN']
        )

    # Decoder
    aux_decoder = None
    if variant['aux_loss']:
        aux_decoder = NormalAux(
            hidden_sizes = variant['aux_params']['hidden'],
            input_size = latent_dim,
            output_size = variant['aux_params']['belief_dim'],
            std = variant['aux_params']['aux_std']
        )
    
    # Actor
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

    # Xplor
    # xplor_qf1 = xplor_qf2 = xplor_vf = xplor_policy = None
    # if variant['decoupled_config']['use']:
    #     xplor_qf1 = FlattenMlp(
    #         hidden_sizes=[net_size, net_size, net_size],
    #         input_size=obs_dim + action_dim + latent_dim,
    #         output_size=1,
    #     )
    #     xplor_qf2 = FlattenMlp(
    #         hidden_sizes=[net_size, net_size, net_size],
    #         input_size=obs_dim + action_dim + latent_dim,
    #         output_size=1,
    #     )
    #     xplor_vf = FlattenMlp(
    #         hidden_sizes=[net_size, net_size, net_size],
    #         input_size=obs_dim + latent_dim,
    #         output_size=1,
    #     )
    #     xplor_policy = TanhGaussianPolicy(
    #         hidden_sizes=[net_size, net_size, net_size],
    #         obs_dim=obs_dim + latent_dim,
    #         latent_dim=latent_dim,
    #         action_dim=action_dim,
    #     )
    # print("TanhGaussianLSTMPolicy() not implemented yet... \n")

    # Dual Agents
    # xplor_agent = None
    # if variant['decoupled_config']['use']:
    #     xplor_agent = PEARLAgent(
    #         xplor_policy,
    #         context_encoder=context_encoder,
    #         aux_decoder=aux_decoder,
    #         aux_params = variant['aux_params'],
    #         latent_dim=latent_dim,  
    #         **variant['algo_params']
    #     )

    agent = PEARLAgent(
        policy,
        context_encoder=context_encoder,
        aux_decoder=aux_decoder,
        aux_params = variant['aux_params'],
        latent_dim=latent_dim,
        **variant['algo_params']
    )
    nets = [agent, qf1, qf2, vf] 
    # if variant['decoupled_config']['use'] is True:
        # nets += [xplor_agent, xplor_qf1, xplor_qf2, xplor_vf]
    nets.append(context_encoder)
    if aux_decoder is not None:
        nets.append(aux_decoder)
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=nets,
        demo_paths=demos,
        decoupled = variant['decoupled_config']['use'],
        **variant['algo_params']
    )
    # print(agent.context_encoder, agent.aux_decoder)
    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        print(f"Loading existing weights from {variant['path_to_weights']} ... \n")
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        aux_decoder.load_state_dict(torch.load(os.path.join(path, 'aux_decoder.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))
    else:
        assert variant['modality'] is 'train'
        
    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    try:
        if ptu.gpu_enabled():
            algorithm.to()
    except:
        TypeError("Cannot set the GPU in this machine")
    else:
        pass
    
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
        raise Exception("This implementation is not complete ... \n")

    writer.close()

def run():
    print("Selecting Default variant for notebook compatiblity ... \n")
    variant = default_config
    experiment(variant)

if __name__ == "__main__":
    run()