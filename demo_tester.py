import sys
sys.path.append("")

import os, shutil
import os.path as osp
import pickle
import json
import numpy as np
import click
import torch

import pyglet
from tensorboardX import SummaryWriter
import datetime

from envs.metaENV import ENV, VisualiserWrapper
from envs.MetaPeg2D import WINDOW_X, WINDOW_Y, ORIGIN, PEG_DEPTH
from backend.torch.PEARL.policies import TanhGaussianPolicy
from backend.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, LatentGNNEncoder, NormalAux
from backend.torch.PEARL.agent import PEARLAgent
from backend.torch.PEARL.sac import PEARLSoftActorCritic

from experiment import deep_update_dict
from backend.torch.PEARL.policies import MakeDeterministic
from backend.samplers.util import rollout, rollout_window
from configs.default import default_config

def run_sim_path(task_idx, trial, path, framework = 'PEARL'):
    env = VisualiserWrapper(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, 
                            resizable = False, visible = False)
    env.show = False
    env.reset_task(task_idx)
    env.set_task_idx(task_idx)
    print(f'Adapting to task {task_idx} on trial {trial}')
    env.run_path(path)
    print('DONE ... \n')
    
def demo_sim_policy(variant, 
               path_to_exp, 
               num_trajs=1,
               demos = None,
               deterministic=False,  
               visible = False,
               continuous = False,
               randomise_task = False,
              ):
    '''
    simulate a trained policy adapting to a new task
    optionally save videos of the trajectories - requires ffmpeg
    :variant: experiment configuration dict
    :path_to_exp: path to exp folder
    :num_trajs: number of trajectories to simulate per task (default 1)
    :deterministic: if the policy is deterministic (default stochastic)
    :save_video: whether to generate and save a video (default False)
    '''
    view = True

    env = ENV() if not view else VisualiserWrapper(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", 
                                                   vsync = False, resizable = False, visible = visible)
    env.show = view
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1    

    eval_tasks=list(tasks[-variant['n_eval_tasks']:])
    print('testing on {} test tasks, {} trajectories each'.format(len(eval_tasks), num_trajs))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim    
    net_size = variant['net_size']

    context_encoder = LatentGNNEncoder(
                    input_dim= context_encoder_input_dim,
                    output_size = context_encoder_output_dim,
                    **variant['LatentGNN']
    )
    if variant['aux_loss']:
        aux_decoder = NormalAux(
            hidden_sizes = variant['aux_params']['hidden'],
            input_size = latent_dim,
            output_size = variant['aux_params']['belief_dim'],
            std = variant['aux_params']['aux_std']
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
        policy,
        context_encoder=context_encoder,
        aux_decoder=aux_decoder,
        aux_params = variant['aux_params'],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # load trained weights (otherwise simulate random policy)
    context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder.pth'), map_location='cpu'))
    policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy.pth'),map_location='cpu'))
    if aux_decoder is not None:
        aux_decoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'aux_decoder.pth'),map_location='cpu'))

    nets = [agent, qf1, qf2, vf] 
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
    algorithm.load_demos()   
    
    env.set_visible(visible = visible)
    env.set_visibles()
        
    max_steps = 300
    
    # loop through tasks collecting rollouts
    all_rets = []
    accum_trajs = True
    use_demos = True
    
    all_paths = []
    for idx in eval_tasks:
        env.set_task_idx(idx)
        env.reset_task(idx)
        if randomise_task:
            env.alter_task()
        env.rollout_counter = 0
        demo_context = algorithm.sample_context(idx, demos = True, batch_size = 64)
        if use_demos:
            agent.demo_clear_z(demo_context)
        else:
            agent.clear_z()
        paths = []
        det = False
        
        if continuous:
            print(f'Adapting to task {idx} on continuous adaptation, extended context True')
            env.view_rollout(agent, accum_context = True, max_steps = max_steps, 
                             continuous_update = True, sparse_rewards = True,
                            deterministic = deterministic)
            env.reset_task(idx)
            env.all_paths.append(env.trajectories)
        else:
            for n in range(num_trajs):
                print(f'Adapting to task {idx} on trial {n}, extended context {accum_trajs}')
                env.view_rollout(agent,
                                accum_context = accum_trajs,
                                max_steps = variant['algo_params']['max_path_length'],
                                sparse_rewards = True,
                                deterministic = det)
                env.reset_task(idx)
                if n >= variant['algo_params']['num_exp_traj_eval'] and deterministic:
                    print("Acting Deterministic")
                    det = True
            env.all_paths.append(env.trajectories)
        
        all_rets.append([sum(p['rewards']) for p in paths])

    return env.all_paths 

def main(config, path, num_trajs = 5, demos = None, deterministic = False, 
         continuous = False, visible = False, randomise_task = False):
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    return demo_sim_policy(variant, path, num_trajs, demos, deterministic, visible, continuous, randomise_task)

if __name__ == "__main__":
    config = None
    num_trajs = 10
    demos_path = "expert/Peg2D/ExpertPeg2DPaths"
    with open(demos_path + '.pickle', 'rb') as handle:
        demos = pickle.load(handle)
#     path = "output/Peg2D/2020_08_08_13_42_38" # Works well down to 2.5 reward average. Not that good at interpreting on-policy transitions
#     all_paths_traj_sparse_over = main(config, path, num_trajs, 
#                      demos = demos,
#                      deterministic = True, 
#                      continuous = False,
#                      visible = True,
#                      randomise_task = True)
    # path = "output/Peg2D/2020_08_08_18_26_12" # Works well down to 4 reward average. Really good at compromising Linfo via mixed auxilliary losses (demo, xplor)
#     path = "output/Peg2D/2020_08_10_12_54_13" # improved meta-test adaptation via encoder context training only
#     all_paths_traj_sparse_xplor = main(config, path, num_trajs, 
#                      demos = demos,
#                      deterministic = True, 
#                      continuous = False,
#                      visible = True,
#                      randomise_task = False)
    path = "output/Peg2D/2020_08_11_16_42_52" # improved meta-test adaptation via encoder context training only. Prior based.
    all_paths_traj_sparse_imperfect = main(config, path, num_trajs, 
                     demos = demos,
                     deterministic = True, 
                     continuous = False,
                     visible = True,
                     randomise_task = True)