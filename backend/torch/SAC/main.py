import itertools
import argparse
import datetime
import sys
sys.path.insert(0,'../../envs/')
import os
from utils import *
from global_vars import BATCH_SIZE, DT, SEED
from PegRobot2D import Frontend, WINDOW_X, WINDOW_Y
import numpy as np
import torch
from sac import SAC
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

save_dir = "models/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
actor_path = save_dir# + "actors/"
if not os.path.exists(actor_path):
    os.makedirs(actor_path)
critic_path = save_dir# + "critics/"
if not os.path.exists(critic_path):
    os.makedirs(critic_path)

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

parser.add_argument('--env-name', default="Peg 2D Robot",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')

parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')

parser.add_argument('--eval', type=bool, default=False,
                    help='Evaluates a policy a policy every 25 episode(s) (default: True)')

parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')

parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')

parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')

parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')

parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')

parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')

parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')

parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')

parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')

parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')

parser.add_argument('--start_steps', type=int, default=0, metavar='N',
                    help='Steps sampling random actions (default: 10000)')

parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')

parser.add_argument('--replay_size', type=int, default=50000, metavar='N',
                    help='size of replay buffer (default: 10000000)')

parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

parser.add_argument('--save_model', type = bool, default = True, metavar = 'G',
                    help='Save model (default: True)')

# args = parser.parse_args()
args, _ = parser.parse_known_args()

# Environment
env = Frontend(WINDOW_X, WINDOW_Y, args.env_name, vsync = False, resizable = False, visible = False)
env.max_episode_steps = 500 # Num episode steps before reset
env.denorm_process = False # No need to denorm because in SAC the gaussian policies are already scaled up
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
num_actions = env.num_actions
num_inputs = env.num_states
action_range = env.action_range

agent = SAC(num_inputs, num_actions, action_range, args)

# Tensorboard
log_dir = 'runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                        args.env_name,
                                        args.policy,
                                       "autotune" if args.automatic_entropy_tuning else "")
writer = SummaryWriter(logdir=log_dir)

# Replay Memory
memory = Memory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

####################
# MAIN training loop
####################

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False

    # Reset the environment
    state = env.reset()

    # Do while the agent has not reached end state / finished episode
    while not done:

        if args.start_steps > total_numsteps:
            action = env.random_action()  # Sample random action for args.start_steps
        else:
            action = agent.get_action(state)  # Sample action from policy

        ####################################################################################################
        # Probably HERE, could pass an argument which updates i.e. 100 steps every 200 collection steps.
        """So we could essentially collect next states, rewards etc. for 200 steps and then, using the buffer,
        update the critic, actor, and entropy predictor for i.e. 100 steps? Would this be closer to what you ment?
        """
        ####################################################################################################
        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates)

                writer.add_scalar('action/action_1', action[0], updates)
                writer.add_scalar('action/action_2', action[1], updates)
                writer.add_scalar('angular/action_2', action[2], updates)
                writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                writer.add_scalar('loss/policy', policy_loss, updates)
                writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                updates += 1

        # Collect next states, rewards
        next_state, reward, done, _ = env.step_func(action, step = episode_steps) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        # Ignore the "done" signal if it comes from hitting the time horizon (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        # Update current step
        state = next_state

    # Usually implemented for benchmarks i.e. 100K / 500K step benchmarks.
    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

    # Evaluation
    if i_episode % 25 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 2
        for _  in range(episodes):
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            while not done:
                action = agent.get_action(state, evaluate=True)

                next_state, reward, done, _ = env.step_func(action, step = episode_steps)
                episode_reward += reward

                state = next_state
                episode_steps += 1

            avg_reward += episode_reward
        avg_reward /= episodes

        writer.add_scalar('avg_reward/test', avg_reward, i_episode)

        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

if args.save_model:
    agent.save_model("Peg2D")
# PER is a good idea for SAC because once we get BIG erwards... the critic is very unstable,
# so values with high TD error are definately more important
# Normalised states?
