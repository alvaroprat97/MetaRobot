import math
import torch
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen = capacity)
        self.position = 0
        self.deque_probas = np.zeros((capacity,))
        self.weighted_epsilon = 1e-6
        self.alpha = 0.6
        self.beta = 0.7
        self.eta = 0.00006

    def push(self, state, action, reward, next_state, done, weighted = False):
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)

        if weighted:
            if len(self.buffer) is self.capacity:
                deque_probas = self.deque_probas[-self.capacity + self.position:self.position]
            else:
                deque_probas = self.deque_probas[:self.position + 1]

            weight = np.max(np.abs(deque_probas)) + self.weighted_epsilon
            self.deque_probas[self.position] = weight

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, per = False):
        """PER sampling, weighting based on some type of error based on critic (TD?)"""

        if per:
            # Update weights with weighting factor alpha
            if len(self.buffer) == self.capacity:
                weights_array = self.deque_probas**self.alpha + self.weighted_epsilon
            else:
                weights_array = (self.deque_probas[:self.position])**self.alpha + self.weighted_epsilon

            buffer_probability = weights_array/np.sum(weights_array)

            if len(self.buffer) == self.capacity:
                self.proba_indexes = np.random.choice(range(-self.capacity + self.position,self.position),
                                                        batch_size,
                                                        p = buffer_probability
                                                        )
            else:
                self.proba_indexes = np.random.choice(range(self.position),
                                                        batch_size,
                                                        p = buffer_probability
                                                        )

            batch = [self.buffer[rand_index] for rand_index in self.proba_indexes]

        else:
            batch = random.sample(self.buffer, batch_size)

        state, action, reward, next_state, done = map(np.stack, zip(*batch))

        return state, action, reward, next_state, done

    def update_probas(self, qf_delta):
        self.deque_probas[self.proba_indexes] = np.abs(np.array(qf_delta.detach().numpy()).flatten()) + self.weighted_epsilon
        return 0

    def __len__(self):
        return len(self.buffer)

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)
