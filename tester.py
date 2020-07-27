import sys
sys.path.append("")

import os, shutil
import os.path as osp
import pickle
import json
import numpy as np
import click
import torch

from tensorboardX import SummaryWriter
import datetime

from envs.metaENV import ENV, VisualiserWrapper
from envs.MetaPeg2D import WINDOW_X, WINDOW_Y, ORIGIN, PEG_DEPTH
from backend.torch.PEARL.policies import TanhGaussianPolicy
from backend.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, LatentGNNEncoder, NormalAux
from backend.torch.PEARL.agent import PEARLAgent

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
    
def sim_policy(variant, 
               path_to_exp, 
               num_trajs=1, 
               deterministic=False, 
               sparse_rewards = False, 
               view=False, 
               continuous = False,
               visible = False
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
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

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
    if variant['aux_loss']:
        aux_decoder = NormalAux(
            hidden_sizes = variant['aux_params']['hidden'],
            input_size = latent_dim,
            output_size = variant['aux_params']['belief_dim'],
            std = variant['aux_params']['aux_std']
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
    
    if variant['decoupled_config']['use']:
        xplor_policy = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim + latent_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )
        xplor_agent = PEARLAgent(
            xplor_policy,
            context_encoder=context_encoder,
            aux_decoder=aux_decoder,
            aux_params = variant['aux_params'],
            latent_dim=latent_dim,
            **variant['algo_params']
        )

    # deterministic eval
    if deterministic:
        agent = MakeDeterministic(agent)

    # load trained weights (otherwise simulate random policy)
    context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder.pth')))
    policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy.pth')))
    if variant['decoupled_config']['use']:
        xplor_policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'xplor_policy.pth')))
    if aux_decoder is not None:
        aux_decoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'aux_decoder.pth')))

    if view:
        env.set_visible(visible = visible)
        env.set_visibles()
        
    max_steps = 200
    context_window_length = 20
    
    # loop through tasks collecting rollouts
    all_rets = []
    all_paths = []
    for idx in eval_tasks:
        env.set_task_idx(idx)
        env.reset_task(idx)
        env.rollout_counter = 0
        agent.clear_z()
        if variant['decoupled_config']['use']:
            xplor_agent.clear_z()
        paths = []
        if not continuous:
            for n in range(num_trajs):
                _agent = agent if n >= variant['algo_params']['num_exp_traj_eval'] else xplor_agent
                if view:
                    print(f'Adapting to task {idx} on trial {n}')
                    env.view_rollout(_agent, accum_context = True, max_steps = variant['algo_params']['max_path_length'],
                                    sparse_rewards = sparse_rewards)
                    env.reset_task(idx)
                else:
                    path = rollout(env,_agent, max_path_length=variant['algo_params']['max_path_length'], 
                                   accum_context=True, sparse_rewards = sparse_rewards)
                    if n >= variant['algo_params']['num_exp_traj_eval']:
                        _agent.infer_posterior(xplor_agent.context)
                    paths.append(path)
            env.all_paths.append(env.trajectories)
        else:
            if view:
                env.view_rollout(agent, accum_context = True, max_steps = max_steps, 
                                 continuous_update = continuous, sparse_rewards = sparse_rewards)#,
#                                 context_window_length = variant['algo_params']['context_window_steps'])
                env.reset_task(idx)
                env.all_paths.append(env.trajectories)
            else:
                path = rollout_window(env, agent, max_path_length = max_steps, context_window_length = context_window_length, accum_context= True, animated = view)

        all_rets.append([sum(p['rewards']) for p in paths])
#         all_paths.append(paths)

#         if view:
#             for n in range(num_trajs):
#                 run_sim_path(idx, n, paths[n])
                
    # compute average returns across tasks
    if not view:
        n = min([len(a) for a in all_rets])
        rets = [a[:n] for a in all_rets]
        rets = np.mean(np.stack(rets), axis=0)
        for i, ret in enumerate(rets):
            print('trajectory {}, avg return: {} \n'.format(i, ret))

    return env.all_paths 

def main(config, path, num_trajs = 5, deterministic = False, sparse_rewards = False, 
         view = False, continuous = False, visible = False):
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    return sim_policy(variant, path, num_trajs, deterministic, sparse_rewards, view, continuous, visible = True)

if __name__ == "__main__":
    config = None
    num_trajs = 5
    # NOT DECOUPLED
    # path = "output/Peg2D/2020_07_08_15_51_03"
    # DECOUPLED
    path = "output/Peg2D/2020_07_20_22_34_06"
    all_paths = main(config, path, num_trajs, 
                     deterministic = False, 
                     sparse_rewards = False, 
                     view = True, 
                     continuous= False,
                     visible = True)

# import sys

# sys.path.append("")

# import os, shutil
# import os.path as osp
# import pickle
# import json
# import numpy as np
# import click
# import torch

# from tensorboardX import SummaryWriter
# import datetime

# from envs.metaENV import ENV, VisualiserWrapper
# from envs.MetaPeg2D import WINDOW_X, WINDOW_Y, ORIGIN, PEG_DEPTH
# from backend.torch.PEARL.policies import TanhGaussianPolicy
# from backend.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
# from backend.torch.PEARL.agent import PEARLAgent

# from experiment import deep_update_dict
# from backend.torch.PEARL.policies import MakeDeterministic
# from backend.samplers.util import rollout, rollout_window
# from configs.default import default_config

# def run_sim_path(task_idx, trial, path, framework = 'PEARL'):
#     env = VisualiserWrapper(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, 
#                             resizable = False, visible = True)
#     env.show = True
#     env.reset_task(task_idx)
#     env.set_task_idx(task_idx)
#     print(f'Adapting to task {task_idx} on trial {trial}')
#     env.run_path(path)
#     print('DONE ... \n')
    
# def sim_policy(variant, path_to_exp, num_trajs=1, deterministic=False, view=False):
#     '''
#     simulate a trained policy adapting to a new task
#     optionally save videos of the trajectories - requires ffmpeg
#     :variant: experiment configuration dict
#     :path_to_exp: path to exp folder
#     :num_trajs: number of trajectories to simulate per task (default 1)
#     :deterministic: if the policy is deterministic (default stochastic)
#     :save_video: whether to generate and save a video (default False)
#     '''

#     env = ENV()
#     tasks = env.get_all_task_idx()
#     obs_dim = int(np.prod(env.observation_space.shape))
#     action_dim = int(np.prod(env.action_space.shape))
#     reward_dim = 1    

#     eval_tasks=list(tasks[-variant['n_eval_tasks']:])
#     print('testing on {} test tasks, {} trajectories each'.format(len(eval_tasks), num_trajs))

#     # instantiate networks
#     latent_dim = variant['latent_size']
#     context_encoder = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim

#     net_size = variant['net_size']
#     recurrent = variant['algo_params']['recurrent']
#     encoder_model = RecurrentEncoder if recurrent else MlpEncoder

#     context_encoder = encoder_model(
#         hidden_sizes=[200,200,200],
#         input_size=obs_dim + action_dim + reward_dim,
#         output_size=context_encoder,
#     )
#     policy = TanhGaussianPolicy(
#         hidden_sizes=[net_size, net_size, net_size],
#         obs_dim=obs_dim + latent_dim,
#         latent_dim=latent_dim,
#         action_dim=action_dim,
#     )
#     agent = PEARLAgent(
#         latent_dim,
#         context_encoder,
#         policy,
#         **variant['algo_params']
#     )
#     # deterministic eval
#     if deterministic:
#         agent = MakeDeterministic(agent)

#     # load trained weights (otherwise simulate random policy)
#     context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder.pth')))
#     policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy.pth')))

#     # loop through tasks collecting rollouts
#     all_rets = []
#     all_paths = []

#     continual = False

#     for idx in eval_tasks:
#         env.reset_task(idx)
#         env.set_task_idx(idx)
#         agent.clear_z()
#         paths = []

#         if not continual:
#             for n in range(num_trajs):
#                 path = rollout(env, agent, max_path_length=variant['algo_params']['max_path_length'], accum_context=True, animated = view)
#                 paths.append(path)
#                 if n >= variant['algo_params']['num_exp_traj_eval']:
#                     agent.infer_posterior(agent.context)
#         else:
#             max_steps = 1000
#             context_window_length = 20
#             path = rollout_window(env, agent, max_path_length = max_steps, context_window_length = context_window_length, accum_context= True, animated = view)
            
#         all_rets.append([sum(p['rewards']) for p in paths])
#         all_paths.append(paths)
        
#         if view:
#             for n in range(num_trajs):
#                 run_sim_path(idx, n, paths[n])
                
#     # compute average returns across tasks
#     n = min([len(a) for a in all_rets])
#     rets = [a[:n] for a in all_rets]
#     rets = np.mean(np.stack(rets), axis=0)
#     for i, ret in enumerate(rets):
#         print('trajectory {}, avg return: {} \n'.format(i, ret))

#     return paths 

# def main(config, path, num_trajs = 5, deterministic = False, view = False):
#     variant = default_config
#     if config:
#         with open(osp.join(config)) as f:
#             exp_params = json.load(f)
#         variant = deep_update_dict(exp_params, variant)
#     return sim_policy(variant, path, num_trajs, deterministic, view)

# if __name__ == "__main__":
#     config = None
#     num_trajs = 1
#     path = "output/Peg2D/2020_06_16_18_59_58/"
#     paths = main(config, path, num_trajs, deterministic = True, view = True)


# def run_sim_path(task_idx, paths, framework = 'PEARL'):
#     env = VisualiserWrapper(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False, visible = True)
#     env.show = True
#     env.task_idx = task_idx
#     for p_idx, path in enumerate(paths):
#         print(f'Adapting to task {task_idx} on trial {p_idx}')
#         env.run_path(path)
#         break

# def sim_policy(variant, path_to_exp, num_trajs=1, deterministic=False, save_video=False):
#     '''
#     simulate a trained policy adapting to a new task
#     optionally save videos of the trajectories - requires ffmpeg
#     :variant: experiment configuration dict
#     :path_to_exp: path to exp folder
#     :num_trajs: number of trajectories to simulate per task (default 1)
#     :deterministic: if the policy is deterministic (default stochastic)
#     :save_video: whether to generate and save a video (default False)
#     '''

#     env = ENV()
#     tasks = env.get_all_task_idx()
#     obs_dim = int(np.prod(env.observation_space.shape))
#     action_dim = int(np.prod(env.action_space.shape))
#     reward_dim = 1    

#     eval_tasks=list(tasks[-variant['n_eval_tasks']:])
#     print('testing on {} test tasks, {} trajectories each'.format(len(eval_tasks), num_trajs))

#     # instantiate networks
#     latent_dim = variant['latent_size']
#     context_encoder = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim

#     net_size = variant['net_size']
#     recurrent = variant['algo_params']['recurrent']
#     encoder_model = RecurrentEncoder if recurrent else MlpEncoder

#     context_encoder = encoder_model(
#         hidden_sizes=[156,156,156],
#         input_size=obs_dim + action_dim + reward_dim,
#         output_size=context_encoder,
#     )
#     policy = TanhGaussianPolicy(
#         hidden_sizes=[net_size, net_size, net_size],
#         obs_dim=obs_dim + latent_dim,
#         latent_dim=latent_dim,
#         action_dim=action_dim,
#     )
#     agent = PEARLAgent(
#         latent_dim,
#         context_encoder,
#         policy,
#         **variant['algo_params']
#     )
#     # deterministic eval
#     if deterministic:
#         agent = MakeDeterministic(agent)

#     # load trained weights (otherwise simulate random policy)
#     context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder.pth')))
#     policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy.pth')))

#     # loop through tasks collecting rollouts
#     all_rets = []
#     for idx in eval_tasks:
#         env.reset_task(idx)
#         env.set_task_idx(idx)
#         agent.clear_z()
#         paths = []
#         for n in range(num_trajs):
#             path = rollout(env, agent, max_path_length=variant['algo_params']['max_path_length'], accum_context=True, save_frames=save_video)
#             paths.append(path)
#             if n >= variant['algo_params']['num_exp_traj_eval']:
#                 agent.infer_posterior(agent.context)
#         run_sim_path(idx, paths)
#         all_rets.append([sum(p['rewards']) for p in paths])

#     # compute average returns across tasks
#     n = min([len(a) for a in all_rets])
#     rets = [a[:n] for a in all_rets]
#     rets = np.mean(np.stack(rets), axis=0)
#     for i, ret in enumerate(rets):
#         print('trajectory {}, avg return: {} \n'.format(i, ret))

# def main(config, path, num_trajs = 5, deterministic = False, video = False):
#     variant = default_config
#     if config:
#         with open(osp.join(config)) as f:
#             exp_params = json.load(f)
#         variant = deep_update_dict(exp_params, variant)
#     return sim_policy(variant, path, num_trajs, deterministic, video)

# if __name__ == "__main__":
#     config = None
#     num_trajs = 8
#     path = "output/Peg2D/2020_06_13_19_31_35/"
#     main(config, path, num_trajs, deterministic = True)

#####################
# DEFAULT VERSION   #
#####################
