import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os
sys.path.insert(0,'../core/')
from utils import hard_update, soft_update, ReplayBuffer
import numpy as np

from models import Critic, Actor

class TD3Agent:

    def __init__(self,
                 num_inputs,
                 num_actions,
                 action_range,
                 gamma = 0.99,
                 tau = 0.005,
                 replay_buffer_size = 2e5,
                 delayed_policy_steps = 2,
                 noise_std = 0.2,
                 noise_bound = 0.5,
                 critic_lr = 1e-3,
                 actor_lr = 1e-3,
                 noise = "normal",
                 exploration_noise = 0.1,
                 hidden_size = 128,
                 ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        print("Using CPU because GPU in this unit is too old")

        self.obs_dim = num_inputs
        self.action_dim = num_actions
        self.normalised_action = False

        self.action_max = action_range['high']
        self.num_actions = num_actions
        self.action_scale = torch.FloatTensor(
            (action_range['high'] - action_range['low']) / 2.)
        self.action_bias = torch.FloatTensor(
            (action_range['high'] + action_range['low']) / 2.)

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_bound = noise_bound
        self.delay_step = delayed_policy_steps
        self.hidden_size = hidden_size
        self.noise_type = noise # Noise OU or Normal(0,0.1) on action for exploration
        self.exploration_noise = exploration_noise

         # initialize actor and critic networks
        self.critic1 = Critic(self.obs_dim, self.action_dim, self.hidden_size).to(self.device)
        self.critic2 = Critic(self.obs_dim, self.action_dim, self.hidden_size).to(self.device)
            # Targets
        self.critic1_target = Critic(self.obs_dim, self.action_dim, self.hidden_size).to(self.device)
        self.critic2_target = Critic(self.obs_dim, self.action_dim, self.hidden_size).to(self.device)
            # Actors
        self.actor = Actor(self.obs_dim, self.action_dim, self.hidden_size, self.action_max).to(self.device)
        self.actor_target = Actor(self.obs_dim, self.action_dim, self.hidden_size, self.action_max).to(self.device)

        # Copy target network from original critics (hard update)
        hard_update(self.critic1_target, self.critic1)
        hard_update(self.critic2_target, self.critic2)
        hard_update(self.actor_target, self.actor)

        # initialize optimizers
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # ReplayBuffer
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def get_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state) + self.action_bias

        action = action.detach().cpu().numpy()[0]
        # print(action.shape)

        if evaluate:
            return action

        noise = self.get_exploration_noise()

        action = (action + noise).clip(-self.action_max, self.action_max)

        return action

    def update_parameters(self, batch_size, updates, PER = False, importance_sampling = False):

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.replay_buffer.sample(batch_size=batch_size, per=PER)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
            	torch.randn_like(action_batch) * self.noise_std
            ).clamp(-self.noise_bound, self.noise_bound)

            next_actions = (
            	self.actor_target(next_state_batch) + noise
            )

            # Compute the target Q value
            next_Q1 = self.critic1_target.forward(next_state_batch, next_actions)
            next_Q2 = self.critic2_target.forward(next_state_batch, next_actions)
            expected_Q = reward_batch +  mask_batch * self.gamma * torch.min(next_Q1, next_Q2)

        # print(state_batch.shape) # Batch x Num States
        # print(next_state_batch.shape) # Batch x Num States
        # print(action_batch.shape) # Batch x Num Actions
        # print(reward_batch.shape) # Batch x 1
        # print(mask_batch.shape) # Batch x 1

        # critic loss
        curr_Q1 = self.critic1.forward(state_batch, action_batch)
        curr_Q2 = self.critic2.forward(state_batch, action_batch)

        # print(curr_Q1.shape, curr_Q2.shape)

        if PER:
            # Update probabilities based on TD errors, mean on both Q networks
            with torch.no_grad():
                td_errors = 0.5*(torch.abs(curr_Q1 - expected_Q) + torch.abs(curr_Q2 - expected_Q)).detach()
                self.replay_buffer.update_probas(td_errors)

            if importance_sampling:
                # Based on https://arxiv.org/pdf/1906.04009.pdf
                p_idxes = self.replay_buffer.proba_indexes
                weights = (batch_size*self.replay_buffer.deque_probas[p_idxes])**-self.replay_buffer.beta / np.max(self.replay_buffer.deque_probas)

                scaled_weights = torch.from_numpy(weights).unsqueeze(1)*td_errors

                critic1_loss = F.mse_loss(curr_Q1*scaled_weights, expected_Q.detach()*scaled_weights)
                critic2_loss = F.mse_loss(curr_Q2*scaled_weights, expected_Q.detach()*scaled_weights)

        elif not PER or not importance_sampling:

            critic1_loss = F.mse_loss(curr_Q1, expected_Q.detach())
            critic2_loss = F.mse_loss(curr_Q2, expected_Q.detach())

        else:
            raise ValueError("Warning on importance sampler and PER")

        # update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # delyaed update for actor & target networks
        if(updates % self.delay_step == 0):

            # print(f"here {self.actor(state_batch).shape}, {self.critic1(state_batch, self.actor(state_batch)).shape}")

            policy_loss = -self.critic1(state_batch, self.actor(state_batch)).mean()

            # print(policy_loss)

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()

            # target networks
            self.update_targets()
            policy_loss_logs = policy_loss.item()
            self.last_pgl = policy_loss_logs

        return critic1_loss.item(), critic2_loss.item(), self.last_pgl, None , None

    # def action_space_noise(self, action_batch):
    #     noise = torch.normal(torch.zeros(action_batch.size()), self.noise_std).clamp(-self.noise_bound, self.noise_bound).to(self.device)
    #     return noise

    def get_exploration_noise(self):
        noise =  np.random.normal(0, self.action_max * self.exploration_noise, size=self.num_actions)
        return noise

    def update_targets(self):
        soft_update(self.critic1_target, self.critic1, self.tau)
        soft_update(self.critic2_target, self.critic2, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)

    def save_model(self, env_name, suffix="", actor_path=None, critic_1_path=None, critic_2_path = None):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        if actor_path is None:
            actor_path = "models/actor_{}_{}".format(env_name, suffix)
        if critic_1_path is None:
            critic_1_path = "models/critic_1_{}_{}".format(env_name, suffix)
        if critic_2_path is None:
            critic_2_path = "models/critic_2_{}_{}".format(env_name, suffix)
        print('Loading models from {} and {} and {}'.format(actor_path, critic_1_path, critic_2_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic1.state_dict(), critic_1_path)
        torch.save(self.critic2.state_dict(), critic_2_path)

    # Load model parameters
    def load_model(self, actor_path, critic_1_path, critic_2_path):
        print('Loading models from {} and {} and {}'.format(actor_path, critic_1_path, critic_2_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_1_path is not None and critic_2_path is not None:
            self.critic1.load_state_dict(torch.load(critic_1_path))
            self.critic2.load_state_dict(torch.load(critic_2_path))
