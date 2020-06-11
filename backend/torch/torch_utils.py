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

        # PER parameters
        self.weighted_epsilon = (1/capacity)*1e-2
        self.alpha = 0.6
        self.beta = 0.7

    def push(self, state, action, reward, next_state, done, weighted = False):
        """weighted (bool) is for the prioritised_experience replay option"""

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


class ActionNoise(object):
    def reset(self):
        pass


class OrnsteinUhlenbeckProcess:
    def __init__(self, act_dim, sigma=0.05, theta=0.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = np.zeros(act_dim)
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def step(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


# PARAM noise
class AdaptiveParamNoiseSpec(object):

    def __init__(self, initial_stddev=0.1, desired_action_stddev=0.2, adaptation_coefficient=1.01):
        """
        Note that initial_stddev and current_stddev refer to std of parameter noise,
        but desired_action_stddev refers to (as name notes) desired std in action space
        """
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient

        self.current_stddev = initial_stddev

    def adapt(self, distance):
        if distance > self.desired_action_stddev:
            # Decrease stddev.
            self.current_stddev /= self.adaptation_coefficient
        else:
            # Increase stddev.
            self.current_stddev *= self.adaptation_coefficient

    def get_stats(self):
        stats = {
            'param_noise_stddev': self.current_stddev,
        }
        return stats

    def __repr__(self):
        fmt = 'AdaptiveParamNoiseSpec(initial_stddev={}, desired_action_stddev={}, adaptation_coefficient={})'
        return fmt.format(self.initial_stddev, self.desired_action_stddev, self.adaptation_coefficient)


def ddpg_distance_metric(actions1, actions2):
    """
    Compute "distance" between actions taken by two policies at the same states
    Expects numpy arrays
    """
    diff = actions1-actions2
    mean_diff = np.mean(np.square(diff), axis=0)
    dist = np.sqrt(np.mean(mean_diff))
    return dist

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
