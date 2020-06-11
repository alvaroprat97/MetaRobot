# Modified from https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py

import os
import sys
sys.path.insert(0,'../core/')
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils import soft_update, hard_update, ReplayBuffer
from models import GaussianPolicy, QNetwork, DeterministicPolicy, weights_init
import numpy as np


class SAC(object):
    def __init__(self, num_inputs, num_actions, action_range,
                gamma = 0.99,
                tau = 0.01,
                alpha = 0.2,
                policy = "Gaussian",
                target_update_interval = 1,
                automatic_entropy_tuning = True,
                hidden_size = 256,
                lr = 3e-4,
                delayed_policy_steps = 2,
                replay_buffer_size = 1e5,
                ):

        self.num_actions = num_actions

        try:
            assert num_actions is len(action_range['high'])
        except:
            AttributeError("Wrong action space")

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.delayed_policy_steps = delayed_policy_steps

        self.policy_type = policy
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        # self.device = torch.device("cuda" if cuda else "cpu")
        self.device = torch.device("cpu")
        print("Working on CPU, GPU is too old")

        self.critic = QNetwork(num_inputs, num_actions, hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=lr)

        self.critic_target = QNetwork(num_inputs, num_actions, hidden_size).to(self.device)

        hard_update(self.critic_target, self.critic)

        if self.policy_type == "Gaussian":
            sz = torch.zeros(size = [num_actions,1]).shape
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.tensor(sz).to(self.device)).item()
                print(f"Target Entropy {self.target_entropy}")
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

            self.policy = GaussianPolicy(num_inputs, num_actions, hidden_size, action_range).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, num_actions, hidden_size, action_range).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def get_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def update_parameters(self, batch_size, updates, PER = False, importance_sampling = False):
        """Sample a batch from self.replay_buffer (replay buffer) and update all NNs,
            PER allows prioritised experience"""

        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.replay_buffer.sample(batch_size=batch_size, per=PER)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # print(state_batch.shape) # Batch x Num States
        # print(next_state_batch.shape) # Batch x Num States
        # print(action_batch.shape) # Batch x Num Actions
        # print(reward_batch.shape) # Batch x 1
        # print(mask_batch.shape) # Batch x 1

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch +  self.gamma * mask_batch * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step

        # PER fill in weights
        if PER:
            # Update probabilities based on TD errors, mean on both Q networks
            td_errors = 0.5*(torch.abs(next_q_value - qf1) + torch.abs(next_q_value - qf2)).detach()
            self.replay_buffer.update_probas(td_errors)

            if importance_sampling:
                # Based on https://arxiv.org/pdf/1906.04009.pdf
                p_idxes = self.replay_buffer.proba_indexes
                weights = (batch_size*self.replay_buffer.deque_probas[p_idxes])**-self.replay_buffer.beta / np.max(self.replay_buffer.deque_probas)

                scaled_weights = torch.from_numpy(weights).unsqueeze(1)*td_errors

                qf1_loss = F.mse_loss(qf1*scaled_weights, next_q_value.detach()*scaled_weights)
                qf2_loss = F.mse_loss(qf2*scaled_weights, next_q_value.detach()*scaled_weights)

                # TODO Normalise these by mse of qf1 and next_q_value etc...

        elif not PER or not importance_sampling:

            qf1_loss = F.mse_loss(qf1, next_q_value.detach())  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value.detach())  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        else:
            raise ValueError("Warning on importance sampler and PER")

        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = (self.alpha * log_pi - min_qf_pi).mean()

        # POLICY loss: add a delay to relax the variance estimation
        if updates%self.delayed_policy_steps == 0:

            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs

        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        # This is just for the logs in tensorboardX
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()

    # Save model parameters
    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')
        if actor_path is None:
            actor_path = "models/actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.policy.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    # Load model parameters
    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.policy.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
