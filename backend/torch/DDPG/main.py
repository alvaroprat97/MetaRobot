import sys
sys.path.insert(0,'../../envs/')
import os
save_dir = "loaders/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

from PegRobot2D import Frontend, WINDOW_X, WINDOW_Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *
from torch import manual_seed, cuda
from numpy.random import seed
from global_vars import BATCH_SIZE, DT, SEED, NUM_EPISODES, NUM_STEPS
import time
from torch import save as torch_save

if __name__ == "__main__":

    rewards = []
    avg_rewards = []

    manual_seed(SEED)
    seed(SEED)

    env = Frontend(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False, visible = False)
    agent = DDPGagent(env, dt = DT)
    param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.2,desired_action_stddev=0.3, adaptation_coefficient=1.02)

    t_start = time.time()

    for episode in range(NUM_EPISODES):
        state = env.reset()
        agent.perturb_actor_parameters(param_noise)
        agent.noise.reset()
        episode_reward = 0
        noise_counter = 0

        for step in range(NUM_STEPS):
            action = agent.get_action(state, action_noise = agent.noise.step(), parametric_noise = param_noise)
            new_state, reward, done, _ = env.step_func(action, dt = DT)
            agent.memory.push(state, action, reward, new_state, done)

            if len(agent.memory) > BATCH_SIZE:
                agent.update(BATCH_SIZE)

            noise_counter += 1
            state = new_state
            episode_reward += reward

            if done:
                sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
                break

        if agent.memory.counter-noise_counter > 0:
            noise_data=[agent.memory.buffer[i] for i in range(agent.memory.counter-noise_counter, agent.memory.counter)]
        else:
            noise_data=[agent.memory.buffer[i] for i in range(agent.memory.counter-noise_counter+agent.memory.max_size//2,
                                                             agent.memory.max_size//2)]\
            + [agent.memory.buffer[i] for i in range(0, agent.memory.counter)]

        noise_data=np.array(noise_data)
        noise_s, noise_a, _,_ , _= zip(*noise_data)

        perturbed_actions = noise_a
        unperturbed_actions = agent.get_action(np.array(noise_s), None, None)
        ddpg_dist = ddpg_distance_metric(perturbed_actions, unperturbed_actions)
        param_noise.adapt(ddpg_dist)

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))
        print(f'Average Return {np.mean(rewards[-10:])} on iteration {episode}')

    t_end = time.time()
    print(f'Finished training, took {t_end-t_start} seconds')

    torch_save(agent.actor, f'{save_dir}actor.pt')
    torch_save(agent.actor_perturbed, f'{save_dir}actor_perturbed.pt')
    np.save(f'{save_dir}param_noise.npy',[param_noise.desired_action_stddev,
                               param_noise.adaptation_coefficient,
                               param_noise.current_stddev])
    np.save(f'{save_dir}rewards.npy',[np.array(rewards), np.array(avg_rewards)])
