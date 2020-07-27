import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

import sys
sys.path.append("")

import backend.torch.pytorch_util as ptu

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain = 5/3)
        torch.nn.init.constant_(m.bias, 0)

def _product_of_gaussians(mus, sigmas_squared, GNN = True):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = torch.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / torch.sum(torch.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * torch.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared

def _mean_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of mean of gaussians
    '''
    mu = torch.mean(mus, dim=0)
    sigma_squared = torch.mean(sigmas_squared, dim=0)
    return mu, sigma_squared


def _natural_to_canonical(n1, n2):
    ''' convert from natural to canonical gaussian parameters '''
    mu = -0.5 * n1 / n2
    sigma_squared = -0.5 * 1 / n2
    return mu, sigma_squared


def _canonical_to_natural(mu, sigma_squared):
    ''' convert from canonical to natural gaussian parameters '''
    n1 = mu / sigma_squared
    n2 = -0.5 * 1 / sigma_squared
    return n1, n2

class DecoupledAgent(object):
    def __init__(self,
                actor_agent,
                xplor_agent,
                context_encoder,
                aux_decoder,
                ):
        super().__init__()

class PEARLAgent(nn.Module):

    def __init__(self,
                 policy,
                 latent_dim = None,
                 context_encoder = None,
                 aux_decoder = None,
                 aux_params = {},
                 **kwargs
    ):
        super().__init__()

        self.context_encoder = context_encoder
        self.aux_decoder = aux_decoder
        self.policy = policy

        self.set_encoder_params(aux_params, latent_dim)

        # self.policy.apply(weights_init)
        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.sparse_rewards = kwargs['sparse_rewards']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def set_encoder(self, encoder, aux_decoder = None):
        self.context_encoder = encoder
        self.aux_decoder = aux_decoder

    def set_encoder_params(self, aux_params, latent_dim, belief = None):
        self.use_aux =  aux_params['use']
        self.fixed_std = aux_params['fixed_std']
        self.aux_std = aux_params['aux_std']
        self.beta = aux_params['beta']
        self.belief = belief
        self.latent_dim = latent_dim

    def clear_z(self, num_tasks=1):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = ptu.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = ptu.ones(num_tasks, self.latent_dim)
        else:
            var = ptu.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)

    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        o, a, r, no, d, info = inputs
        if self.sparse_rewards:
            r = info['sparse_reward']
        o = ptu.from_numpy(o[None, None, ...])
        a = ptu.from_numpy(a[None, None, ...])
        r = ptu.from_numpy(np.array([r])[None, None, ...])
        no = ptu.from_numpy(no[None, None, ...])

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = torch.distributions.Normal(ptu.zeros(self.latent_dim), ptu.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it
            additional_trans: additional transition for mutual information
        '''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]  
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()

    def _infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it
            just outputs, does not affect class variables
        '''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, self.context_encoder.output_size)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = [_product_of_gaussians(m, s) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]  
            z_means = torch.stack([p[0] for p in z_params])
            z_vars = torch.stack([p[1] for p in z_params])
        # sum rather than product of gaussians structure
        else:
            z_means = torch.mean(params, dim=1)
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(z_means), torch.unbind(z_vars))]
            z = [d.rsample() for d in posteriors]
            z = torch.stack(z)
        else:
            z = z_means
        return z

    def sample_z(self, tmp = False):
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(self.z_means), torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def infer_aux(self):
        # print("deprecated ...")
        raw_post = self.aux_decoder(self.z, return_log_prob = False)[1]
        raw_post = raw_post.view(raw_post.size(0), -1, self.aux_decoder.output_size)
        assert self.aux_decoder.output_size//2 == self.aux_decoder.output_size/2
        if self.fixed_std:
            sigma_squared = torch.ones_like(raw_post)*self.aux_std**2
            aux_posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(raw_post), torch.unbind(sigma_squared))]
        else:
            raise NotImplementedError
            mu = raw_post[..., :self.aux_decoder.output_size//2]
            sigma_squared = F.softplus(raw_post[..., self.aux_decoder.output_size//2:])
            aux_posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))]
        sampled_post = [d.rsample() for d in aux_posteriors]
        self.belief = torch.stack(sampled_post)

    def get_action(self, obs, deterministic=False):
        ''' sample action from the policy, conditioned on the task embedding '''
        z = self.z
        obs = ptu.from_numpy(obs[None])
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_, deterministic=deterministic)

    def set_num_steps_total(self, n):
        self.policy.set_num_steps_total(n)

    def forward(self, obs, context, aux_targets = None):
        ''' given context, get statistics under the current policy of a set of observations '''
        self.infer_posterior(context)
        # self.sample_z()

        task_z = self.z

        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_, reparameterize=True, return_log_prob=True)

        if aux_targets is not None:
            aux_outputs = self.aux_decoder(task_z, task_descriptor = aux_targets, reparameterize = True, return_log_prob = True)
        else:
            aux_outputs = None

        return policy_outputs, task_z, aux_outputs

    def log_diagnostics(self, eval_statistics):
        '''
        adds logging data about encodings to eval_statistics
        '''
        z_mean = np.mean(np.abs(ptu.get_numpy(self.z_means[0])))
        z_sig = np.mean(ptu.get_numpy(self.z_vars[0]))
        eval_statistics['Z mean eval'] = z_mean
        eval_statistics['Z variance eval'] = z_sig

    @property
    def networks(self):
        if isinstance(self.aux_decoder, nn.Module) and isinstance(self.context_encoder, nn.Module):
            return [self.context_encoder, self.policy, self.aux_decoder]
        else:
            if isinstance(self.aux_decoder, nn.Module):
                return [self.policy, self.aux_decoder]
            else:
                return [self.policy]