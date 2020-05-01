import sys
from env import Frontend, WINDOW_X, WINDOW_Y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *

env = Frontend(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False)
env.show = False

agent = DDPGagent(env)
noise = OUNoise(action_range=env.action_range)
batch_size = 128
rewards = []
avg_rewards = []

for episode in range(50):
    state = env.reset()
    noise.reset()
    episode_reward = 0

    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()
