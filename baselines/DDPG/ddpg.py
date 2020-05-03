# Based on https://github.com/openai/baselines/tree/master/baselines/ddpg
# https://openai.com/blog/better-exploration-with-parameter-noise/

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from models import *
from utils import *

from global_vars import USE_CUDA, FloatTensor, Tensor, LongTensor, ByteTensor

class DDPGagent:
    def __init__(self, env, hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=25000, dt = 1/60):
        # Params
        self.num_states = env.num_states
        self.num_actions = env.num_actions
        self.gamma = gamma
        self.tau = tau
        self.dt = dt
        self.hidden_size = hidden_size
        self.rand_init_nets()

        if USE_CUDA:
            self.actor.cuda()
            self.actor_target.cuda()
            self.critic.cuda()
            self.critic_target.cuda()

        hard_update(target = self.actor_target, source = self.actor)
        # for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            # target_param.data.copy_(param.data)

        hard_update(target = self.critic_target, source = self.critic)
        # for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            # target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion  = nn.MSELoss()
        self.noise = OrnsteinUhlenbeckProcess(act_dim = self.num_actions, dt = self.dt*10)

        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def rand_init_nets(self):
        # Networks
        self.actor = Actor(self.num_states, self.hidden_size, self.num_actions).type(FloatTensor)

        self.actor_target = Actor(self.num_states, self.hidden_size, self.num_actions).type(FloatTensor)
        self.actor_perturbed = Actor(self.num_states, self.hidden_size, self.num_actions).type(FloatTensor)
        self.critic = Critic(self.num_states + self.num_actions, self.hidden_size, self.num_actions).type(FloatTensor)

        models = [self.actor, self.actor_target, self.actor_perturbed, self.critic]

        for model in models:
            model.apply(weights_init)

        self.critic_target = Critic(self.num_states + self.num_actions, self.hidden_size, self.num_actions).type(FloatTensor)
        self.critic_target.load_state_dict(self.critic.state_dict())

    def get_action(self, state, action_noise = None, parametric_noise = None):

        state = Variable(torch.from_numpy(state).float().unsqueeze(0),requires_grad = False).type(FloatTensor)

        # Eval mode
        self.actor.eval()
        self.actor_perturbed.eval()

        if parametric_noise is not None:
            action = self.actor_perturbed.forward(state)
        else:
            action = self.actor.forward(state)

        action = action.detach().numpy()[0]

        self.actor.train()
        if action_noise is not None:
            action = action + action_noise
        return action

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(target = self.actor_perturbed, source = self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            random = torch.randn(param.shape)
            if USE_CUDA:
                random = random.cuda()
            param += random * param_noise.current_stddev

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size) # _ is done booleans
        states = FloatTensor(states)
        actions = FloatTensor(actions)
        rewards = FloatTensor(rewards)
        next_states = FloatTensor(next_states)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()

        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
