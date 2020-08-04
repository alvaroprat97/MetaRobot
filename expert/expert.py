"""
Train an expert with additional task belief information. State is augmented with information about the task which is not given during meta-training.
Use dense rewards.

SAC agent acts deterministic during the demonstration. Trained using Haarnoja's 2018 SAC implementation.

We save the policy for the expert in order to generate expert demonstrations. Neeed to check that the demosntrations succeed!
If policy doesn't succeed then user must manually define a trajectory through the MetaPeg2D GUI.

Ideally, at test time we want a person to provide a demosntration of the task, and then we want the agent to infer from that.
"""

# Train a separate SAC agent for each task family: Peg Insertion, Nail removal, Object flip.

import itertools
import datetime
import sys
import numpy as np
import torch
import os
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

sys.path.append("")
sys.path.append("../")

from envs.metaENV import ENV
from baselines.SAC.sac import SAC
from backend.trainers.multi_rl_trainer import MultiTaskRLAlgorithm

variant = dict(
        algorithm="SAC",
        version="normal",
        seed = 1,
        save_model = True,

        algorithm_kwargs=dict(
            train_tasks = [x for x in range(0,40)],
            eval_tasks = [x for x in range(40,50)],
            num_epochs= 50,
            num_eval_steps_per_epoch= 100,
            num_train_loops_per_epoch = 10,
            num_trains_per_train_loop= 250, 
            num_expl_steps_per_train_loop = 100,
            num_steps_before_training = 100,
            max_path_length= 50,
            batch_size = 128,
            num_train_tasks = 8, # Number of tasks to train on
            num_tasks_sample = 8, # Sample for collection
            prioritised_experience = False,
            importance_sampling = False,
        ),

        trainer_kwargs=dict(
            gamma=0.99,
            tau=0.005,
            target_update_interval=1,
            lr= 5e-4, # Was 5e-3
            alpha = 0.2,
            policy = "Gaussian",
            automatic_entropy_tuning=True,
            hidden_size = 64,       
            delayed_policy_steps = 2,
        ),
        env_kwargs = dict(
            vsync = False,
            resizable = False,
            visible = False
            )
)

# Environment
env = ENV(expert = True)

torch.manual_seed(variant['seed'])
np.random.seed(variant['seed'])

# Agent
num_actions = len(env.action_space)
num_inputs = len(env.observation_space)

agent = SAC(num_inputs, num_actions, action_range = None, **variant['trainer_kwargs'])

# Tensorboard
log_dir = 'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                        "Peg2DRobot",
                                        variant['trainer_kwargs']['policy'],
                                       "autotune" if variant['trainer_kwargs']['automatic_entropy_tuning'] else "")
writer = SummaryWriter(logdir=log_dir)

# training
if __name__ == "main":
    RL_trainer = MultiTaskRLAlgorithm(env, agent, **variant['algorithm_kwargs']) 
    RL_trainer.train(writer)


###############################
# TESTING #####################
###############################

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
# from baselines.SAC.sac import SAC
# from envs.MetaPeg2D import WINDOW_X, WINDOW_Y, ORIGIN, PEG_DEPTH
# from backend.torch.PEARL.policies import TanhGaussianPolicy
# from backend.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, LatentGNNEncoder, NormalAux

# from backend.samplers.util import expert_rollout
# from configs.default import default_config
# from expert import variant

    
# def sim_policy(variant, 
#                path_to_exp, 
#                num_demos=1, 
#                view=False, 
#                continuous = False,
#                visible = False
#               ):
    
#     '''
#     simulate a trained policy adapting to a new task
#     optionally save videos of the trajectories - requires ffmpeg
#     :variant: experiment configuration dict
#     :path_to_exp: path to exp folder
#     :num_trajs: number of trajectories to simulate per task (default 1)
#     :deterministic: if the policy is deterministic (default stochastic)
#     :save_video: whether to generate and save a video (default False)
#     '''

#     env = ENV(expert = True) if not view else VisualiserWrapper(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", 
#                                                        vsync = False, resizable = False, visible = visible)

#     num_actions = len(env.action_space)
#     num_inputs = len(env.observation_space)

#     agent = SAC(num_inputs, num_actions, action_range = None, **variant['trainer_kwargs'])
#     agent.load_model(os.path.join(path_to_exp,'actor_ExpertPeg2D_'), os.path.join(path_to_exp, 'critic_ExpertPeg2D_'))

#     if view:
#         env.set_visible(visible = visible)
#         env.set_visibles()
    
#     # loop through tasks collecting rollouts
#     all_rets = []
#     all_paths = []
#     eval_tasks = [x for x in range(40,50)]
#     for idx in eval_tasks:
#         env.set_task_idx(idx)
#         env.reset_task(idx)
#         env.rollout_counter = 0

#         paths = []

#         for n in range(num_demos):
#             if view:
#                 print(f'Adapting to task {idx} on demo {n}')
#                 env.view_expert_rollout(agent, max_steps = variant['algorithm_kwargs']['max_path_length'])
#                 env.reset_task(idx)
#             else:
#                 path = expert_rollout(env,agent, max_path_length=variant['algorithm_kwargs']['max_path_length'],
#                                      animated = visible, eval = True)                
#                 paths.append(path)
                
#         env.all_paths.append(env.trajectories)
            
#         all_rets.append([sum(p['rewards']) for p in paths])

#     if not view:
#         n = min([len(a) for a in all_rets])
#         rets = [a[:n] for a in all_rets]
#         rets = np.mean(np.stack(rets), axis=0)
#         for i, ret in enumerate(rets):
#             print('trajectory {}, avg return: {} \n'.format(i, ret))

#     return env.all_paths 

# def main(config, path, num_demos = 3, view = False, visible = False):
#     return sim_policy(variant, path, num_demos, view, visible = True)

# if __name__ == "__main__":
#     config = None
#     num_demos = 3
#     path = "models/"
#     all_paths = main(config, 
#                      path, 
#                      num_demos, 
#                      view = True, 
#                      visible = True)
#     from envs.MetaPeg2D import denorm_pos, norm_pos
#     from pymunk import Vec2d
#     for path in all_paths:
#         for demo in path:
#             for obs, nobs in zip(demo['observations'], demo['next_observations']):
#                 obs[0], obs[1] = demo['goal_pos'] - denorm_pos(Vec2d(obs[0], obs[1]))
#                 nobs[0], nobs[1] = demo['goal_pos'] - denorm_pos(Vec2d(nobs[0], nobs[1]))
#     demo_paths = dict(
#                 Peg2D = all_paths)
#     import pickle
#     with open('expert/Peg2D/ExpertPeg2DPaths.pickle', 'wb') as handle:
#         pickle.dump(demo_paths, handle, protocol=pickle.HIGHEST_PROTOCOL)