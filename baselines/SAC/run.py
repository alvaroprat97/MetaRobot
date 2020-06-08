import itertools
import argparse
import datetime
import sys
sys.path.insert(0,'../../envs/')
sys.path.insert(0,'../core/')
import os
from utils import *
from global_vars import BATCH_SIZE, DT, SEED
from PegRobot2D import Frontend, WINDOW_X, WINDOW_Y
import numpy as np
import torch
from sac import SAC
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from rl_batch_trainer import BatchRLAlgorithm

save_dir = "models/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
actor_path = save_dir
if not os.path.exists(actor_path):
    os.makedirs(actor_path)
critic_path = save_dir
if not os.path.exists(critic_path):
    os.makedirs(critic_path)

variant = dict(
        algorithm="SAC",
        version="normal",
        seed = 5,
        replay_buffer_size=int(2e5),
        save_model = True,

        algorithm_kwargs=dict(
            num_epochs= 25,
            num_eval_steps_per_epoch= 2500,
            num_train_loops_per_epoch = 5,
            num_trains_per_train_loop= 500, # Was 100
            num_expl_steps_per_train_loop = 2500,
            min_num_steps_before_training = 0, #5000, # Random exploration steps Initially
            max_path_length= 250,
            batch_size = 256,
            prioritised_experience = True,
        ),

        trainer_kwargs=dict(
            gamma=0.99,
            tau=5e-3,
            target_update_interval=1,
            lr=1e-3, # Was 5e-3
            alpha = 0.2,
            policy = "Gaussian",
            automatic_entropy_tuning=True,
            hidden_size = 128,
        ),

        env_args = [
            WINDOW_X,
            WINDOW_Y,
            "Peg 2D Robot"
            ],

        env_kwargs = dict(
            vsync = False,
            resizable = False,
            visible = False
            )
)

# Environment
env = Frontend(*variant['env_args'], **variant['env_kwargs'])
if variant['algorithm'] is "SAC":
    env.denorm_process = False # No need to denorm because in SAC the gaussian policies are already scaled up

torch.manual_seed(variant['seed'])
np.random.seed(variant['seed'])

# Agent
num_actions = env.num_actions
num_inputs = env.num_states
action_range = env.action_range

agent = SAC(num_inputs, num_actions, action_range, **variant['trainer_kwargs'])

# Tensorboard
log_dir = 'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                        "Peg2DRobot",
                                        variant['trainer_kwargs']['policy'],
                                       "autotune" if variant['trainer_kwargs']['automatic_entropy_tuning'] else "")
writer = SummaryWriter(logdir=log_dir)

# Replay Memory
replay_buffer = ReplayBuffer(variant['replay_buffer_size'])

# training
RL_trainer = BatchRLAlgorithm(replay_buffer, **variant['algorithm_kwargs'])
RL_trainer.train(env, agent, writer)

# Save model
if variant['save_model']:
    agent.save_model("Peg2D")

# # Normalised states?

### Test

from sac import SAC
from utils import *
from models import weights_init
import sys
import argparse
sys.path.insert(0,'../../envs/')
from PegRobot2D import Frontend, WINDOW_X, WINDOW_Y

variant = dict(
        algorithm="SAC",
        version="normal",
        seed = 123456,
        replay_buffer_size=int(1e5),
        save_model = True,
        algorithm_kwargs=dict(
            num_epochs=50,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=250,
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=2500, # Random exploration steps Initially
            max_path_length=250,
            batch_size=256,
        ),

        trainer_kwargs=dict(
            gamma=0.99,
            tau=5e-3,
            target_update_interval=1,
            lr=3e-4,
            alpha = 0.2,
            policy = "Gaussian",
            automatic_entropy_tuning=True,
            hidden_size = 256
        ),

        env_args = [
            WINDOW_X,
            WINDOW_Y,
            "Peg 2D Robot"
            ],

        env_kwargs = dict(
            vsync = False,
            resizable = False,
            visible = False
            )
)

# Environment
# env = Frontend(*variant['env_args'], **variant['env_kwargs'])
# if variant['algorithm'] is "SAC":
#     env.denorm_process = False # No need to denorm because in SAC the gaussian policies are already scaled up
#
# torch.manual_seed(variant['seed'])
# np.random.seed(variant['seed'])
#
# def run_policy(agent, env = None, framework = "SAC"):
#     if isinstance(env, Frontend):
#         del(env)
#     env = Frontend(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False, visible = True)
#     env.agent = agent
#     if framework is "SAC":
#         env.denorm_process = False # Necessary for SAC
#     env.run_policy(agent)
#
# if __name__ == "__main__":
#     env = Frontend(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False, visible = False)
#
#     # Agent
#     num_actions = env.num_actions
#     num_inputs = env.num_states
#     action_range = env.action_range
#
#     tst_agent = SAC(num_inputs, num_actions, action_range, **variant['trainer_kwargs'])
#
#     tst_agent.load_model(actor_path="models/actor_Peg2D_",critic_path="models/critic_Peg2D_")
#
#     run_policy(tst_agent, env, "SAC")

# run_policy(tst_agent, env, "SAC")
