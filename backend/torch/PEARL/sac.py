import sys
from collections import OrderedDict

from time import time

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from tensorboardX import SummaryWriter
import backend.torch.pytorch_util as ptu
from backend.core.eval_util import create_stats_ordered_dict
from backend.core.meta_rl_demo_algorithm import MetaRLAlgorithm
import itertools

sys.path.append("")

class PEARLSoftActorCritic(MetaRLAlgorithm):
    def __init__(
            self,
            env,
            train_tasks,
            eval_tasks,
            nets,
            demo_paths = None,
            policy_lr=1e-3,
            qf_lr=1e-3,
            vf_lr=1e-3,
            aux_lr=1e-3,
            context_lr=1e-3,
            kl_lambda=1.,
            info_lambda=1,
            bc_lambda=1,
            aux_lambda=1,
            policy_mean_reg_weight=1e-3,
            policy_std_reg_weight=1e-3,
            policy_pre_activation_weight=0.,
            optimizer_class=optim.Adam,
            recurrent=False,
            use_information_bottleneck=True,
            use_next_obs_in_context=False,
            sparse_rewards=False,
            decoupled = False,
            l2_reg = True,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,
            **kwargs
    ):
        if decoupled:
            super().__init__(
                env=env,
                agent=dict(
                    xplor = nets[4],
                    actor = nets[0],
                    ),
                train_tasks=train_tasks,
                eval_tasks=eval_tasks,
                demo_paths = demo_paths,
                **kwargs
            )
        else:
            super().__init__(
                env=env,
                agent= nets[0],
                train_tasks=train_tasks,
                eval_tasks=eval_tasks,
                demo_paths = demo_paths,
                **kwargs
            )

        # decoupled exploration and exploitation policies
        self.decoupled = decoupled

        self.soft_target_tau = soft_target_tau
        self.policy_mean_reg_weight = policy_mean_reg_weight
        self.policy_std_reg_weight = policy_std_reg_weight
        self.policy_pre_activation_weight = policy_pre_activation_weight
        self.plotter = plotter
        self.render_eval_paths = render_eval_paths

        self.recurrent = recurrent
        self.kl_lambda = kl_lambda
        self.info_lambda = info_lambda
        self.bc_lambda = bc_lambda
        self.aux_lambda = aux_lambda 

        self.l2_reg = True

        # Criterions
        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()
        self.vib_criterion = nn.MSELoss()
        self.l2_reg_criterion = nn.MSELoss()

        self.use_information_bottleneck = use_information_bottleneck
        self.sparse_rewards = sparse_rewards
        self.use_next_obs_in_context = use_next_obs_in_context

        # Actor policy
        self.qf1, self.qf2, self.vf = nets[1:4]
        self.target_vf = self.vf.copy()
        if self.decoupled:
            self.policy_optimizer = optimizer_class(
                self.agent['actor'].policy.parameters(),
                lr=policy_lr,
            )
        else:
            self.policy_optimizer = optimizer_class(
                self.agent.policy.parameters(),
                lr=policy_lr,
            )
            self.pretrain_policy_optimizer = optimizer_class(
                self.agent.policy.parameters(), 
                lr=policy_lr, 
                weight_decay=1e-3
                )

        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )
        self.vf_optimizer = optimizer_class(
            self.vf.parameters(),
            lr=vf_lr,
        )

        # xplorer policy
        if self.decoupled:
            self.xplor_qf1, self.xplor_qf2, self.xplor_vf = nets[5:8]
            self.xplor_target_vf = self.xplor_vf.copy() 
            self.xplor_policy_optimizer = optimizer_class(
                    self.agent['xplor'].policy.parameters(),
                    lr=policy_lr,
                    )
            self.xplor_qf1_optimizer = optimizer_class(
                self.xplor_qf1.parameters(),
                lr=qf_lr,
            )
            self.xplor_qf2_optimizer = optimizer_class(
                self.xplor_qf2.parameters(),
                lr=qf_lr,
            )
            self.xplor_vf_optimizer = optimizer_class(
                self.xplor_vf.parameters(),
                lr=vf_lr,
            )

        # Set encoders
        if self.decoupled:
            self.context_encoder = nets[8]
            self.aux_decoder = nets[9]
            self.agent['actor'].set_encoder(self.context_encoder, self.aux_decoder)
            self.agent['xplor'].set_encoder(self.context_encoder, self.aux_decoder)

            # If we decouple, we use SAC to share the encoder amongst the 2 agents
            self.context_optimizer = optimizer_class(
                self.context_encoder.parameters(),
                lr=context_lr,
            )
            self.aux_optimizer = optimizer_class(
                self.aux_decoder.parameters(),
                lr=aux_lr,
            )
        else:
            self.context_optimizer = optimizer_class(
                self.agent.context_encoder.parameters(),
                lr=context_lr,
            )
            self.aux_optimizer = optimizer_class(
                self.agent.aux_decoder.parameters(),
                lr=aux_lr,
            )
        self.context_extension = 0

    ###### Torch stuff #####
    @property
    def networks(self):
        if self.decoupled:
            return (self.agent['actor'].networks + self.agent['xplor'].networks + [self.agent['actor']] + [self.agent['xplor']] + 
                    [self.qf1, self.qf2, self.vf, self.target_vf] + [self.xplor_qf1, self.xplor_qf2, self.xplor_vf, self.xplor_target_vf])
        else:
            return (self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf])

    def training_mode(self, mode):
        for net in self.networks:
            net.train(mode)

    def to(self, device=None):
        if device == None:
            device = ptu.device
            print(f"Operating on {device} \n")
            # print(self.networks)
        for net in self.networks:
            net.to(device)

    ##### Data handling #####
    def unpack_batch(self, batch, sparse_reward=False):
        ''' unpack a batch and return individual elements '''
        o = batch['observations'][None, ...]
        a = batch['actions'][None, ...]
        if sparse_reward:
            r = batch['sparse_rewards'][None, ...]
        else:
            r = batch['rewards'][None, ...]
        no = batch['next_observations'][None, ...]
        t = batch['terminals'][None, ...]
        if 'contexts' in batch.keys():
            ctxt = batch['contexts'][None, ...]
            return [o, a, r, no, t, ctxt]
        return [o, a, r, no, t]

    def sample_sac(self, indices, actor = True, last = None, return_context = False, batch_size = None):
        ''' sample batch of training data from a list of tasks for training the actor-critic 
            last = number which represents how far out you can randomly sample from (to preserve behavioural policy in demos)
        '''
        # this batch consists of transitions sampled randomly from replay buffer
        # rewards are always dense
        batch_size = batch_size if batch_size is not None else self.batch_size
        if self.decoupled:
            # print("SAC.py non implemented recurrent sampling")
            assert self.context_encoder is self.agent['xplor'].context_encoder
            assert self.agent['xplor'].context_encoder is self.agent['actor'].context_encoder
        if actor:
            batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=batch_size, last = last, context = return_context)) for idx in indices]
        else:
            batches = [ptu.np_to_pytorch_batch(self.xplor_replay_buffer.random_batch(idx, batch_size=batch_size, last = last, context = return_context)) for idx in indices]
        unpacked = [self.unpack_batch(batch) for batch in batches]
        # group like elements together
        unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
        unpacked = [torch.cat(x, dim=0) for x in unpacked]
        return unpacked

    def sample_context(self, indices, demos = False, batch_size = None, unpacked = False, entire_traj = False):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
        sequence = True if self.recurrent or entire_traj else False
        buffer = self.demo_buffer if demos else self.enc_replay_buffer  
        batch_size = self.embedding_batch_size if batch_size is None else batch_size
        batches = [ptu.np_to_pytorch_batch(buffer.random_batch(idx, batch_size=batch_size, sequence=sequence)) for idx in indices]
        context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
        # group like elements together
        context = [[x[i] for x in context] for i in range(len(context[0]))]
        context = [torch.cat(x, dim=0) for x in context]
        # full context consists of [obs, act, rewards, next_obs, terms]
        # if dynamics don't change across tasks, don't include next_obs
        # don't include terminals in context
        dim = 0 if unpacked else 2
        if self.use_next_obs_in_context:
            context = torch.cat(context[:-1], dim=dim) if not unpacked else context[:-1]
        else:
            context = torch.cat(context[:-2], dim=dim) if not unpacked else context[:-2]
        return context

    def importance_sampling(self, obs, actions, context):
        # Flatten
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        context = context.view(t * b, -1)
        obs = torch.cat([obs, context.detach()], dim=1)
        with torch.no_grad():
            pi_out = self.agent.policy(obs, reparameterize=True, return_log_prob=True, bc_actions = actions)
            bi_out = self.agent.behaviour_policy(obs, reparameterize=True, return_log_prob=True, bc_actions = actions)
        pi = torch.exp(pi_out[-1])
        bi = torch.exp(bi_out[-1])
        i_w_ = pi/bi
        i_w_ = torch.clamp(i_w_, 0.0, 2.0)
        return i_w_

    def update_behaviour_policy(self):
        self.agent.update_behaviour_policy()

    def _do_pretraining(self, indices):
        """
        pretrain for one loop on indices
        """
        batch_size = self.embedding_batch_size
        demo_context = self.sample_context(indices, batch_size = batch_size, demos = True)
        self.agent.demo_clear_z(demo_context, num_tasks = len(indices))

        mb_size = self.embedding_mini_batch_size
        num_updates = self.embedding_batch_size // (mb_size - self.context_extension)
        assert num_updates == 1

        # Num updates = 1 usually, unless recurrent
        for i in range(num_updates):
            context = demo_context[:, i * mb_size: i * mb_size + mb_size, :]
            self._take_pretrain_step(indices, context)

            # stop backprop
            if self.decoupled:
                for agent in self.agent.values():
                    agent.detach_z()
            else:
                self.agent.detach_z()
        
    ##### Training #####
    def _do_training(self, indices, collected_indices = None):
        """
        collected_indices: those indices used in recent collection. To be used for the behavioural collection during IS.
        """

        mb_size = self.embedding_mini_batch_size + self.context_extension
        num_updates = self.embedding_batch_size // (mb_size - self.context_extension)

        # sample context batch, use a batch size (from config) to determine how much context to sample
        # This context must be representative of that context used to generate the trajectories.
        context_batch = self.sample_context(indices, demos = False)
        demos_context = self.sample_context(indices, demos = True)
            # BEFORE
        # context = torch.cat([context_batch, demos_context], dim = 1)
            # NOW
        context = context_batch

        # zero out context and hidden encoder state
        # self.agent.demo_clear_z(context,num_tasks=len(indices))
        self.agent.clear_z(num_tasks = len(indices))

        # do this in a loop so we can truncate backprop in the recurrent encoder
        for i in range(num_updates):
            # context = context[:, i * mb_size: i * mb_size + mb_size, :]
            self._take_step(indices, collected_indices, context)

            # stop backprop
            if self.decoupled:
                for agent in self.agent.values():
                    agent.detach_z()
            else:
                self.agent.detach_z()

    def _min_q(self, obs, actions, task_z, actor = True):
        if actor:
            q1 = self.qf1(obs, actions, task_z.detach())
            q2 = self.qf2(obs, actions, task_z.detach())
        else:
            q1 = self.xplor_qf1(obs, actions, task_z.detach())
            q2 = self.xplor_qf2(obs, actions, task_z.detach())
        min_q = torch.min(q1, q2)
        return min_q

    def _update_target_network(self):
        ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)
        if self.decoupled:
            ptu.soft_update_from_to(self.xplor_vf, self.xplor_target_vf, self.soft_target_tau)

    def _extended_context(self, context, obs, actions, rewards, next_obs, idx):
        o = obs[:,:,idx]
        a = actions[:,:,idx]
        r = rewards[:,:,idx]
        no = next_obs[:,:,idx]
        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)
        x_ = torch.cat([context, data], dim=1)   
        return x_

    @torch.no_grad()
    def mutual_information(self, 
                            context, 
                            rewards, 
                            obs, 
                            actions, 
                            next_obs, 
                            task_z = None,
                            mbs = 1,
                            ):
        # Rescale
        t, b, _= rewards.size()
        assert b//mbs == b/mbs
        rewards = rewards.view(t, mbs, b//mbs, -1)
        obs = obs.view(t, mbs, b//mbs, -1)
        actions = actions.view(t, mbs, b//mbs, -1)
        next_obs = next_obs.view(t, mbs, b//mbs, -1)

        if task_z is None:
            _, mb, _ = context.size()
            # Increase significance
            context = context[:, :mb//2, :].clone()
            task_z = self.agent['xplor']._infer_posterior(context)
            task_z = [z.repeat(b, 1) for z in task_z]
            task_z = torch.cat(task_z, dim=0)
        x_tot = []

        for idx in range(b//mbs):
            x_ = self._extended_context(context, obs, actions, rewards, next_obs, idx)
            x_tot.append(x_)
        x_tot = torch.cat(x_tot)
        z_ = self.agent['xplor']._infer_posterior(x_tot)
        z_ = torch.cat([m.repeat(mbs,1) for m in z_])

        rewards = rewards.view(t * b, -1)
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # with torch.no_grad():
        target = rewards + self.discount * self.target_vf(next_obs, task_z.detach())
        basis = torch.min(self.qf1(obs, actions, z_), self.qf2(obs, actions, z_))
        r_mi = torch.abs(basis - target)
        return r_mi

    def _take_pretrain_step(self, indices, demo_context):
        num_tasks = len(indices)

        # Observations and Actions generated by the demonstrations since replay only contains demos at this point.
        obs, actions, _, _, _ = self.sample_sac(indices, actor=True)

        aux_targets = self.env._get_targets(indices)
        t, b, _ = obs.size()
        assert t == num_tasks
        targets = [list(torch.tensor(tuple(aux_targets[i])*b, device = ptu.device).split(len(aux_targets[i]))) for i in range(len(indices))]
        targets = list(itertools.chain(*targets))    
        targets = torch.stack(targets)

        policy_outputs, _, aux_outputs = self.agent(obs, demo_context, aux_targets = targets, bc_actions = actions)

        _, policy_mean, policy_log_std, _ = policy_outputs[:4]
        aux_log_pi = aux_outputs[3]
        bc_log_pi = policy_outputs[-1]

        # Context encoder IB
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent['xplor'].compute_kl_div() if self.decoupled else self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # Auxilliary decoder optimisation & gradient flow to encoder
        self.aux_optimizer.zero_grad()
        aux_loss = -self.aux_lambda*torch.mean(aux_log_pi.flatten())
        aux_loss.backward()

        self.context_optimizer.step()
        self.aux_optimizer.step()

        # Q-function update via 1-step returns
        # TODO

        # Optimise the policy using BC
        policy_loss = -torch.mean(bc_log_pi) # LogLoss
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        policy_reg_loss = mean_reg_loss + std_reg_loss # RegRession loss (Tanh)

        policy_loss = policy_loss + policy_reg_loss
        self.pretrain_policy_optimizer.zero_grad()
        policy_loss.backward()
        self.pretrain_policy_optimizer.step()

    def _take_step(self, indices, collected_indices, context):

        num_tasks = len(indices)
        # data is in dimensions (task, batch, feat)

        # B
        obs, actions, rewards, next_obs, terms = self.sample_sac(indices)
        # Collect recent observiations and actions, and the latent space used to generate these. 
        # Used for mutual information matching. Make sure c_latent is from demos.
        c_obs, c_actions, _, c_nobs, c_terms, c_latent_space = self.sample_sac(collected_indices, 
                                                                        actor = True, 
                                                                        last = self.num_extra_rl_steps_posterior, 
                                                                        return_context = True,
                                                                        batch_size = self.embedding_batch_size
                                                                        )
                                      
        # TODO Should just unpack the demo_context for accuracy, reducing mismatch of inference. 
        demo_context = self.sample_context(indices, demos = True, unpacked = False)
        demo_obs, demo_actions, _, _ = self.sample_context(indices, demos= True, unpacked=True) 
        
        aux_targets = self.env._get_targets(indices)
        demo_targets = self.env._get_targets(indices, altered = True)
        # Get auxilliary targets in tensor format
        t, b, _ = obs.size()
        assert t == num_tasks 
        targets = [list(torch.tensor(tuple(aux_targets[i])*b, device = ptu.device).split(len(aux_targets[i]))) for i in range(len(indices))]
        targets = list(itertools.chain(*targets))    
        targets = torch.stack(targets)
        demo_t, demo_b, _ = demo_obs.size()
        assert demo_t == num_tasks
        demo_targets = [list(torch.tensor(tuple(demo_targets[i])*demo_b, device = ptu.device).split(len(aux_targets[i]))) for i in range(len(indices))]
        demo_targets = list(itertools.chain(*demo_targets))    
        demo_targets = torch.stack(demo_targets)

        # run inference in networks
        detached_aux = False 
        # Policy outputs used for RL agent. Task_z is the sampled belief of the task for the given batch (context is from encoder buffer and demos)
        policy_outputs, task_z, aux_outputs = self.agent(obs, context, aux_targets = targets, aux_no_grad = detached_aux)
        # IL inference to retreive log probabilities (supervised learning part) from demonstration transitions
        # Inference on z performed on context from self.agent(obs, context)
        demo_policy_outputs, demo_aux_outputs = self.agent.bc_forward(demo_obs, demo_actions, demo_context, 
                                                                        aux_targets = demo_targets, aux_no_grad = detached_aux)
        new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
        aux_log_pi = aux_outputs[3]
        demo_aux_log_pi = demo_aux_outputs[3]
        bc_log_pi = demo_policy_outputs[-1]
        
        # flattens out the task dimension
        obs = obs.view(t * b, -1)
        actions = actions.view(t * b, -1)
        next_obs = next_obs.view(t * b, -1)

        # Q and V networks
        q1_pred = self.qf1(obs, actions, task_z)
        q2_pred = self.qf2(obs, actions, task_z)
        v_pred = self.vf(obs, task_z.detach())
        # get targets for use in V and Q updates
        with torch.no_grad():
            target_v_values = self.target_vf(next_obs, task_z)

        # KL constraint on z if probabilistic
        self.context_optimizer.zero_grad()
        if self.use_information_bottleneck:
            kl_div = self.agent['xplor'].compute_kl_div() if self.decoupled else self.agent.compute_kl_div()
            kl_loss = self.kl_lambda * kl_div
            kl_loss.backward(retain_graph=True)

        # Mutual Information dependancy loss
        importance_sampling = True
        # c_terms = c_terms.float()
        # Convert terms to int form as encoder can only receive sparse rewards!
        collected_context = torch.cat([c_obs, c_actions, c_terms.float(), c_nobs], dim = 2)
        iS_factor = self.importance_sampling(c_obs, c_actions, c_latent_space) if importance_sampling else 1.0
        m_i_log_prob = self.agent.encoder_log_probs(collected_context, target_latent_space = c_latent_space)
        m_i_log_prob = m_i_log_prob*iS_factor.flatten()
        m_i = torch.mean(m_i_log_prob)
        m_i_loss = -self.info_lambda*m_i 
        # Separatet the graphs to allow adam to choose different step sizes for the different losses (would not be the same if we just add the loss). 
        m_i_loss.backward(retain_graph = True)

        # Auxilliary decoder optimisation & gradient flow to encoder (optional, currently no gradient backprop to encoder model)
        self.aux_optimizer.zero_grad()
        aux_loss = -(self.aux_lambda)*torch.mean(aux_log_pi.flatten()) 
        aux_demo_loss = -self.aux_lambda*torch.mean(demo_aux_log_pi.flatten())
        # Ignoring demo aux losses
        #TODO WE ONLY CHANGED ADDING AUX_LOSS  
        aux_tot_loss = aux_loss + aux_demo_loss# + aux_loss 
        aux_tot_loss.backward(retain_graph = not detached_aux)

        # qf and encoder update (note encoder does not get grads from policy or vf)
        self.qf1_optimizer.zero_grad()
        self.qf2_optimizer.zero_grad()
        rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
        # scale rewards for Bellman update
        rewards_flat = rewards_flat * self.reward_scale
        terms_flat = terms.view(self.batch_size * num_tasks, -1)
        q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
        qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
        qf_loss.backward() 

        # Context and aux has Actor Critic loss, aux loss and KLD loss
        self.context_optimizer.step() # Loss on KLD, Critic, Mutual Information
        self.aux_optimizer.step() # Loss on Aux
        self.qf1_optimizer.step() # Critic Loss
        self.qf2_optimizer.step() # Critic Loss

        # compute min Q on the new actions
        min_q_new_actions = self._min_q(obs, new_actions, task_z, actor=True)
        # vf update
        v_target = min_q_new_actions - log_pi
        vf_loss = self.vf_criterion(v_pred, v_target.detach())
        self.vf_optimizer.zero_grad()
        vf_loss.backward()
        self.vf_optimizer.step()

        self._update_target_network()

        # policy update 
        # n.b. policy update includes dQ/da
        log_policy_target = min_q_new_actions
        policy_loss = (log_pi - log_policy_target).mean()
        mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
        std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
        pre_tanh_value = policy_outputs[-2]
        pre_activation_reg_loss = self.policy_pre_activation_weight * ((pre_tanh_value**2).sum(dim=1).mean())
        policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
        policy_loss = policy_loss + policy_reg_loss

        pol_optimizer = self.pretrain_policy_optimizer if self.l2_reg else self.policy_optimizer
        pol_optimizer.zero_grad()
        # Retain graph to allow both losses to be reduced at better rates
        policy_loss.backward(retain_graph = True)
        # BC trajectory optimization
        demo_policy_loss = -self.bc_lambda*torch.mean(bc_log_pi) 
        # policy_loss = policy_loss + demo_policy_loss
        demo_policy_loss.backward()
        pol_optimizer.step()

        # save some statistics for eval
        if self.eval_statistics is None:
            # eval should set this to None.
            # this way, these statistics are only computed for one batch.
            self.eval_statistics = OrderedDict()
            if self.use_information_bottleneck:
                z_mean = np.mean(np.abs(ptu.get_numpy(self.agent['xplor'].z_means[0]))) if self.decoupled else np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
                z_sig = np.mean(ptu.get_numpy(self.agent['xplor'].z_vars[0])) if self.decoupled else np.mean(ptu.get_numpy(self.agent.z_vars[0]))
                self.eval_statistics['Z mean train'] = z_mean
                self.eval_statistics['Z variance train'] = z_sig
                self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
                self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

            self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
            self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
            self.eval_statistics['Aux Loss'] = np.mean(ptu.get_numpy(aux_loss))
            self.eval_statistics['Demo Aux Loss'] = np.mean(ptu.get_numpy(aux_demo_loss))
            self.eval_statistics['bc_loss'] = np.mean(ptu.get_numpy(demo_policy_loss))
            self.eval_statistics['info_loss'] = np.mean(ptu.get_numpy(m_i_loss))

            if isinstance(self.writer, SummaryWriter):
                self.writer.add_scalar('loss/QF',np.mean(ptu.get_numpy(qf_loss)),self._n_train_steps_total)
                self.writer.add_scalar('loss/VF',np.mean(ptu.get_numpy(vf_loss)),self._n_train_steps_total)
                self.writer.add_scalar('loss/Aux',np.mean(ptu.get_numpy(aux_loss)),self._n_train_steps_total)
                self.writer.add_scalar('loss/Aux',np.mean(ptu.get_numpy(aux_demo_loss)),self._n_train_steps_total)
                self.writer.add_scalar('loss/Policy',np.mean(ptu.get_numpy(policy_loss)),self._n_train_steps_total)
                self.writer.add_scalar('loss/KL',ptu.get_numpy(kl_loss),self._n_train_steps_total)
                self.writer.add_scalar('loss/info', ptu.get_numpy(m_i_loss),self._n_train_steps_total)
                self.writer.add_scalar('loss/demo_bc ', ptu.get_numpy(demo_policy_loss),self._n_train_steps_total)

            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Predictions',
                ptu.get_numpy(q1_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'V Predictions',
                ptu.get_numpy(v_pred),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

    def get_epoch_snapshot(self, epoch):
        # NOTE: overriding parent method which also optionally saves the env
        snapshot = OrderedDict(
            qf1=self.qf1.state_dict(),
            qf2=self.qf2.state_dict(),
            policy=self.agent.policy.state_dict(),
            vf=self.vf.state_dict(),
            target_vf=self.target_vf.state_dict(),
            context_encoder=self.agent.context_encoder.state_dict(),
            aux_decoder = self.agent.aux_decoder.state_dict(),
        )
        return snapshot

# import sys
# from collections import OrderedDict

# from time import time

# import numpy as np
# import torch
# import torch.optim as optim
# from torch import nn as nn

# from tensorboardX import SummaryWriter
# import backend.torch.pytorch_util as ptu
# from backend.core.eval_util import create_stats_ordered_dict
# from backend.core.meta_rl_demo_algorithm import MetaRLAlgorithm
# import itertools

# sys.path.append("")

# class PEARLSoftActorCritic(MetaRLAlgorithm):
#     def __init__(
#             self,
#             env,
#             train_tasks,
#             eval_tasks,
#             nets,
#             demo_paths = None,
#             policy_lr=1e-3,
#             qf_lr=1e-3,
#             vf_lr=1e-3,
#             aux_lr=1e-3,
#             context_lr=1e-3,
#             kl_lambda=1.,
#             policy_mean_reg_weight=1e-3,
#             policy_std_reg_weight=1e-3,
#             policy_pre_activation_weight=0.,
#             optimizer_class=optim.Adam,
#             recurrent=False,
#             use_information_bottleneck=True,
#             use_next_obs_in_context=False,
#             sparse_rewards=False,
#             decoupled = False,
#             l2_reg = True,

#             soft_target_tau=1e-2,
#             plotter=None,
#             render_eval_paths=False,
#             **kwargs
#     ):
#         if decoupled:
#             super().__init__(
#                 env=env,
#                 agent=dict(
#                     xplor = nets[4],
#                     actor = nets[0],
#                     ),
#                 train_tasks=train_tasks,
#                 eval_tasks=eval_tasks,
#                 demo_paths = demo_paths,
#                 **kwargs
#             )
#         else:
#             super().__init__(
#                 env=env,
#                 agent= nets[0],
#                 train_tasks=train_tasks,
#                 eval_tasks=eval_tasks,
#                 demo_paths = demo_paths,
#                 **kwargs
#             )

#         # decoupled exploration and exploitation policies
#         self.decoupled = decoupled

#         self.soft_target_tau = soft_target_tau
#         self.policy_mean_reg_weight = policy_mean_reg_weight
#         self.policy_std_reg_weight = policy_std_reg_weight
#         self.policy_pre_activation_weight = policy_pre_activation_weight
#         self.plotter = plotter
#         self.render_eval_paths = render_eval_paths

#         self.recurrent = recurrent
#         self.kl_lambda = kl_lambda
#         self.l2_reg = True

#         # Criterions
#         self.qf_criterion = nn.MSELoss()
#         self.vf_criterion = nn.MSELoss()
#         self.vib_criterion = nn.MSELoss()
#         self.l2_reg_criterion = nn.MSELoss()

#         self.use_information_bottleneck = use_information_bottleneck
#         self.sparse_rewards = sparse_rewards
#         self.use_next_obs_in_context = use_next_obs_in_context

#         # Actor policy
#         self.qf1, self.qf2, self.vf = nets[1:4]
#         self.target_vf = self.vf.copy()
#         if self.decoupled:
#             self.policy_optimizer = optimizer_class(
#                 self.agent['actor'].policy.parameters(),
#                 lr=policy_lr,
#             )
#         else:
#             self.policy_optimizer = optimizer_class(
#                 self.agent.policy.parameters(),
#                 lr=policy_lr,
#             )
#             self.pretrain_policy_optimizer = optimizer_class(
#                 self.agent.policy.parameters(), 
#                 lr=policy_lr, 
#                 weight_decay=1e-3
#                 )

#         self.qf1_optimizer = optimizer_class(
#             self.qf1.parameters(),
#             lr=qf_lr,
#         )
#         self.qf2_optimizer = optimizer_class(
#             self.qf2.parameters(),
#             lr=qf_lr,
#         )
#         self.vf_optimizer = optimizer_class(
#             self.vf.parameters(),
#             lr=vf_lr,
#         )

#         # xplorer policy
#         if self.decoupled:
#             self.xplor_qf1, self.xplor_qf2, self.xplor_vf = nets[5:8]
#             self.xplor_target_vf = self.xplor_vf.copy() 
#             self.xplor_policy_optimizer = optimizer_class(
#                     self.agent['xplor'].policy.parameters(),
#                     lr=policy_lr,
#                     )
#             self.xplor_qf1_optimizer = optimizer_class(
#                 self.xplor_qf1.parameters(),
#                 lr=qf_lr,
#             )
#             self.xplor_qf2_optimizer = optimizer_class(
#                 self.xplor_qf2.parameters(),
#                 lr=qf_lr,
#             )
#             self.xplor_vf_optimizer = optimizer_class(
#                 self.xplor_vf.parameters(),
#                 lr=vf_lr,
#             )

#         # Set encoders
#         if self.decoupled:
#             self.context_encoder = nets[8]
#             self.aux_decoder = nets[9]
#             self.agent['actor'].set_encoder(self.context_encoder, self.aux_decoder)
#             self.agent['xplor'].set_encoder(self.context_encoder, self.aux_decoder)

#             # If we decouple, we use SAC to share the encoder amongst the 2 agents
#             self.context_optimizer = optimizer_class(
#                 self.context_encoder.parameters(),
#                 lr=context_lr,
#             )
#             self.aux_optimizer = optimizer_class(
#                 self.aux_decoder.parameters(),
#                 lr=aux_lr,
#             )
#         else:
#             self.context_optimizer = optimizer_class(
#                 self.agent.context_encoder.parameters(),
#                 lr=context_lr,
#             )
#             self.aux_optimizer = optimizer_class(
#                 self.agent.aux_decoder.parameters(),
#                 lr=aux_lr,
#             )
#         self.context_extension = 0

#     ###### Torch stuff #####
#     @property
#     def networks(self):
#         if self.decoupled:
#             return (self.agent['actor'].networks + self.agent['xplor'].networks + [self.agent['actor']] + [self.agent['xplor']] + 
#                     [self.qf1, self.qf2, self.vf, self.target_vf] + [self.xplor_qf1, self.xplor_qf2, self.xplor_vf, self.xplor_target_vf])
#         else:
#             return (self.agent.networks + [self.agent] + [self.qf1, self.qf2, self.vf, self.target_vf])

#     def training_mode(self, mode):
#         for net in self.networks:
#             net.train(mode)

#     def to(self, device=None):
#         if device == None:
#             device = ptu.device
#             print(f"Operating on {device} \n")
#             # print(self.networks)
#         for net in self.networks:
#             net.to(device)

#     ##### Data handling #####
#     def unpack_batch(self, batch, sparse_reward=False):
#         ''' unpack a batch and return individual elements '''
#         o = batch['observations'][None, ...]
#         a = batch['actions'][None, ...]
#         if sparse_reward:
#             r = batch['sparse_rewards'][None, ...]
#         else:
#             r = batch['rewards'][None, ...]
#         no = batch['next_observations'][None, ...]
#         t = batch['terminals'][None, ...]
#         if 'contexts' in batch.keys():
#             ctxt = batch['contexts'][None, ...]
#             return [o, a, r, no, t, ctxt]
#         return [o, a, r, no, t]

#     def sample_sac(self, indices, actor = True, last = None, return_context = False, batch_size = None):
#         ''' sample batch of training data from a list of tasks for training the actor-critic 
#             last = number which represents how far out you can randomly sample from (to preserve behavioural policy in demos)
#         '''
#         # this batch consists of transitions sampled randomly from replay buffer
#         # rewards are always dense
#         batch_size = batch_size if batch_size is not None else self.batch_size
#         if self.decoupled:
#             # print("SAC.py non implemented recurrent sampling")
#             assert self.context_encoder is self.agent['xplor'].context_encoder
#             assert self.agent['xplor'].context_encoder is self.agent['actor'].context_encoder
#         if actor:
#             batches = [ptu.np_to_pytorch_batch(self.replay_buffer.random_batch(idx, batch_size=batch_size, last = last, context = return_context)) for idx in indices]
#         else:
#             batches = [ptu.np_to_pytorch_batch(self.xplor_replay_buffer.random_batch(idx, batch_size=batch_size, last = last, context = return_context)) for idx in indices]
#         unpacked = [self.unpack_batch(batch) for batch in batches]
#         # group like elements together
#         unpacked = [[x[i] for x in unpacked] for i in range(len(unpacked[0]))]
#         unpacked = [torch.cat(x, dim=0) for x in unpacked]
#         return unpacked

#     def sample_context(self, indices, batch_size = None, unpacked = False):
#         ''' sample batch of context from a list of tasks from the replay buffer '''
#         # make method work given a single task index
#         if not hasattr(indices, '__iter__'):
#             indices = [indices]
#         batch_size = self.embedding_batch_size if batch_size is None else batch_size
#         batches = [ptu.np_to_pytorch_batch(self.enc_replay_buffer.random_batch(idx, batch_size=batch_size, sequence=self.recurrent)) for idx in indices]
#         context = [self.unpack_batch(batch, sparse_reward=self.sparse_rewards) for batch in batches]
#         # group like elements together
#         context = [[x[i] for x in context] for i in range(len(context[0]))]
#         context = [torch.cat(x, dim=0) for x in context]
#         # full context consists of [obs, act, rewards, next_obs, terms]
#         # if dynamics don't change across tasks, don't include next_obs
#         # don't include terminals in context
#         dim = 0 if unpacked else 2
#         if self.use_next_obs_in_context:
#             context = torch.cat(context[:-1], dim=dim) if not unpacked else context[:-1]
#         else:
#             context = torch.cat(context[:-2], dim=dim) if not unpacked else context[:-2]
#         return context

#     def importance_sampling(self, obs, actions, context):
#         # z = infer from context
#         obs = torch.stack([obs, z])
#         log_pi = self.agent.policy.forward(obs = target_actions, a = actions)
#         log_bi = self.agent.behaviour_policy.forward(obs = target_actions, a = actions)
#         return None

#     def _do_pretraining(self, indices):
#         """
#         pretrain for one loop on indices
#         """
#         batch_size = self.embedding_batch_size
#         demo_context = self.sample_context(indices, batch_size = batch_size)
#         self.agent.demo_clear_z(demo_context, num_tasks = len(indices))

#         mb_size = self.embedding_mini_batch_size
#         num_updates = self.embedding_batch_size // (mb_size - self.context_extension)

#         # Num updates = 1 usually, unless recurrent
#         for i in range(num_updates):
#             context = demo_context[:, i * mb_size: i * mb_size + mb_size, :]
#             self._take_pretrain_step(indices, context)

#             # stop backprop
#             if self.decoupled:
#                 for agent in self.agent.values():
#                     agent.detach_z()
#             else:
#                 self.agent.detach_z()
        
#     ##### Training #####
#     def _do_training(self, indices, collected_indices = None):
#         """
#         collected_indices: those indices used in recent collection. To be used for the behavioural collection during IS.
#         """

#         mb_size = self.embedding_mini_batch_size + self.context_extension
#         num_updates = self.embedding_batch_size // (mb_size - self.context_extension)

#         # sample context batch, use a batch size (from config) to determine how much context to sample
#         context_batch = self.sample_context(indices)
#         # collected_context_batch = None
#         # if collected_indices is not None:
#         #     # Retreive the transitions of the demo contexts used during collection.
#         #     collected_context_batch = self.sample_context(collected_indices) 

#         # zero out context and hidden encoder state
#         if self.decoupled:
#             self.agent['xplor'].demo_clear_z(context_batch,num_tasks=len(indices))
#             self.agent['actor'].demo_clear_z(context_batch,num_tasks=len(indices))
#         else:
#             self.agent.demo_clear_z(context_batch,num_tasks=len(indices))
#             self.agent.update_behaviour_policy()

#         # do this in a loop so we can truncate backprop in the recurrent encoder
#         for i in range(num_updates):
#             context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
#             self._take_step(indices, collected_indices, context)

#             # stop backprop
#             if self.decoupled:
#                 for agent in self.agent.values():
#                     agent.detach_z()
#             else:
#                 self.agent.detach_z()

#     def _min_q(self, obs, actions, task_z, actor = True):
#         if actor:
#             q1 = self.qf1(obs, actions, task_z.detach())
#             q2 = self.qf2(obs, actions, task_z.detach())
#         else:
#             q1 = self.xplor_qf1(obs, actions, task_z.detach())
#             q2 = self.xplor_qf2(obs, actions, task_z.detach())
#         min_q = torch.min(q1, q2)
#         return min_q

#     def _update_target_network(self):
#         ptu.soft_update_from_to(self.vf, self.target_vf, self.soft_target_tau)
#         if self.decoupled:
#             ptu.soft_update_from_to(self.xplor_vf, self.xplor_target_vf, self.soft_target_tau)

#     def _extended_context(self, context, obs, actions, rewards, next_obs, idx):
#         o = obs[:,:,idx]
#         a = actions[:,:,idx]
#         r = rewards[:,:,idx]
#         no = next_obs[:,:,idx]
#         if self.use_next_obs_in_context:
#             data = torch.cat([o, a, r, no], dim=2)#.unsqueeze(1)
#         else:
#             data = torch.cat([o, a, r], dim=2)#.unsequeeze(1)
#         x_ = torch.cat([context, data], dim=1)   
#         return x_

#     @torch.no_grad()
#     def mutual_information(self, 
#                             context, 
#                             rewards, 
#                             obs, 
#                             actions, 
#                             next_obs, 
#                             task_z = None,
#                             mbs = 1,
#                             ):
#         # Rescale
#         t, b, _= rewards.size()
#         assert b//mbs == b/mbs
#         rewards = rewards.view(t, mbs, b//mbs, -1)
#         obs = obs.view(t, mbs, b//mbs, -1)
#         actions = actions.view(t, mbs, b//mbs, -1)
#         next_obs = next_obs.view(t, mbs, b//mbs, -1)

#         if task_z is None:
#             _, mb, _ = context.size()
#             # Increase significance
#             context = context[:, :mb//2, :].clone()
#             task_z = self.agent['xplor']._infer_posterior(context)
#             task_z = [z.repeat(b, 1) for z in task_z]
#             task_z = torch.cat(task_z, dim=0)
#         x_tot = []

#         for idx in range(b//mbs):
#             x_ = self._extended_context(context, obs, actions, rewards, next_obs, idx)
#             x_tot.append(x_)
#         x_tot = torch.cat(x_tot)
#         z_ = self.agent['xplor']._infer_posterior(x_tot)
#         z_ = torch.cat([m.repeat(mbs,1) for m in z_])

#         rewards = rewards.view(t * b, -1)
#         obs = obs.view(t * b, -1)
#         actions = actions.view(t * b, -1)
#         next_obs = next_obs.view(t * b, -1)

#         # with torch.no_grad():
#         target = rewards + self.discount * self.target_vf(next_obs, task_z.detach())
#         basis = torch.min(self.qf1(obs, actions, z_), self.qf2(obs, actions, z_))
#         r_mi = torch.abs(basis - target)
#         return r_mi

#     def _take_pretrain_step(self, indices, demo_context):
#         num_tasks = len(indices)

#         # Observations and Actions generated by the demonstrations since replay only contains demos at this point.
#         obs, actions, _, _, _ = self.sample_sac(indices, actor=True)

#         aux_targets = self.env._get_targets(indices)
#         t, b, _ = obs.size()
#         assert t == num_tasks
#         targets = [list(torch.tensor(tuple(aux_targets[i])*b, device = ptu.device).split(len(aux_targets[i]))) for i in range(len(indices))]
#         targets = list(itertools.chain(*targets))    
#         targets = torch.stack(targets)

#         policy_outputs, _, aux_outputs = self.agent(obs, demo_context, aux_targets = targets, bc_actions = actions)

#         _, policy_mean, policy_log_std, _ = policy_outputs[:4]
#         aux_log_pi = aux_outputs[3]
#         bc_log_pi = policy_outputs[-1]

#         # Context encoder IB
#         self.context_optimizer.zero_grad()
#         if self.use_information_bottleneck:
#             kl_div = self.agent['xplor'].compute_kl_div() if self.decoupled else self.agent.compute_kl_div()
#             kl_loss = self.kl_lambda * kl_div
#             kl_loss.backward(retain_graph=True)

#         # Auxilliary decoder optimisation & gradient flow to encoder
#         self.aux_optimizer.zero_grad()
#         aux_loss = -self.agent.beta*torch.mean(aux_log_pi.flatten())
#         aux_loss.backward()

#         self.context_optimizer.step()
#         self.aux_optimizer.step()

#         # Optimise the policy using BC
#         policy_loss = -torch.mean(bc_log_pi) # LogLoss
#         mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
#         std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
#         policy_reg_loss = mean_reg_loss + std_reg_loss # RegRession loss (Tanh)
#         # TODO all L2 Loss

#         policy_loss = policy_loss + policy_reg_loss
#         self.pretrain_policy_optimizer.zero_grad()
#         policy_loss.backward()
#         self.pretrain_policy_optimizer.step()

#     def _take_step(self, indices, collected_indices, context):

#         # print("context should come from explorer buffer mainly")
#         # if self.decoupled:
#         #     xplor_context = context
#         #     context = context[:,:-self.context_extension,:]

#         num_tasks = len(indices)
#         # data is in dimensions (task, batch, feat)
#         obs, actions, rewards, next_obs, terms = self.sample_sac(indices, actor=True)
#         # Collect recent observiations and actions, and the latent space used to generate these. Used for mutual information matching. 
#         c_obs, c_actions, _, c_nobs, c_terms, c_context = self.sample_sac(collected_indices, 
#                                                                         actor = True, 
#                                                                         last = self.num_steps_prior, 
#                                                                         return_context = True,
#                                                                         batch_size=self.embedding_batch_size
#                                                                         )
                                      
#         # TODO Should just unpack the context for accuracy, reducing mismatch of inference. 
#         # Used for behavioural cloning. TODO, better to sample demo_obs, demo_actons from context.
#         demo_obs, demo_actions, _, _ = self.sample_context(indices, unpacked=True) 
#         if self.decoupled:
#             xplor_obs, xplor_actions, xplor_rewards, xplor_next_obs, xplor_terms = self.sample_sac(indices, actor=False)
        
#         aux_targets = self.env._get_targets(indices)
#         # Get auxilliary targets in tensor format
#         t, b, _ = obs.size()
#         # assert t == num_tasks 
#         # targets = [list(torch.tensor(tuple(aux_targets[i])*b, device = ptu.device).split(len(aux_targets[i]))) for i in range(len(indices))]
#         # targets = list(itertools.chain(*targets))    
#         # targets = torch.stack(targets)
#         demo_t, demo_b, _ = demo_obs.size()
#         assert demo_t == num_tasks
#         demo_targets = [list(torch.tensor(tuple(aux_targets[i])*demo_b, device = ptu.device).split(len(aux_targets[i]))) for i in range(len(indices))]
#         demo_targets = list(itertools.chain(*demo_targets))    
#         demo_targets = torch.stack(demo_targets)

#         # run inference in networks
#         detached_aux = True # Dectivated for now.
#         if self.decoupled:
#             policy_outputs, task_z, aux_outputs = self.agent['actor'](obs, context, aux_targets = targets)
#             xplor_policy_outputs, xplor_task_z, xplor_aux_outputs = self.agent['xplor'](xplor_obs, context, aux_targets = targets)
#         else:
#             # Policy outputs used for RL agent
#             policy_outputs, task_z, aux_outputs = self.agent(obs, context)
#             # IL inference to retreive log probabilities (supervised learning part) from demonstration transitions
#             # Inference on z performed on context from self.agent(obs, context)
#             demo_policy_outputs, demo_aux_outputs = self.agent.bc_forward(demo_obs, demo_actions, aux_targets = demo_targets, aux_no_grad = detached_aux)

#         new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]
#         # aux_log_pi = aux_outputs[3] if hasattr(aux_outputs, len) else None
#         aux_log_pi = demo_aux_outputs[3]
#         bc_log_pi = demo_policy_outputs[-1]

#         if self.decoupled:
#             xplor_new_actions, xplor_policy_mean, xplor_policy_log_std, xplor_log_pi = xplor_policy_outputs[:4]
#             xplor_aux_log_pi = xplor_aux_outputs[3]

#         # flattens out the task dimension
#         obs = obs.view(t * b, -1)
#         actions = actions.view(t * b, -1)
#         next_obs = next_obs.view(t * b, -1)

#         if self.decoupled:
#             # flattens out the task dimension
#             xplor_obs_ = xplor_obs
#             xplor_actions_ = xplor_actions
#             xplor_next_obs_ = xplor_next_obs
#             xplor_obs = xplor_obs.view(t * b, -1)
#             xplor_actions = xplor_actions.view(t * b, -1)
#             xplor_next_obs = xplor_next_obs.view(t * b, -1)

#         # Q and V networks
#         q1_pred = self.qf1(obs, actions, task_z)
#         q2_pred = self.qf2(obs, actions, task_z)
#         v_pred = self.vf(obs, task_z.detach())
#         # get targets for use in V and Q updates
#         with torch.no_grad():
#             target_v_values = self.target_vf(next_obs, task_z)

#         if self.decoupled:
#             # We are not passing gradients to encoeder because xplor_task_z is detached
#             xplor_q1_pred = self.xplor_qf1(xplor_obs, xplor_actions, xplor_task_z.detach())
#             xplor_q2_pred = self.xplor_qf2(xplor_obs, xplor_actions, xplor_task_z.detach())
#             xplor_v_pred = self.xplor_vf(xplor_obs, xplor_task_z.detach())
#             # get targets for use in V and Q updates
#             with torch.no_grad():
#                 xplor_target_v_values = self.xplor_target_vf(xplor_next_obs, xplor_task_z)

#         # KL constraint on z if probabilistic
#         self.context_optimizer.zero_grad()
#         if self.use_information_bottleneck:
#             kl_div = self.agent['xplor'].compute_kl_div() if self.decoupled else self.agent.compute_kl_div()
#             kl_loss = self.kl_lambda * kl_div
#             kl_loss.backward(retain_graph=True)
            
#         # Mutual Information dependancy loss
#         kappa = 1.0
#         importance_sampling = False
#         # c_terms = c_terms.float()
#         # Convert terms to int form as encoder can only receive sparse rewards!
#         enc_context = torch.cat([c_obs, c_actions, c_terms.float(), c_nobs], dim = 2)
#         iS_factor = self.importance_sampling(c_obs, c_actions, c_context) if importance_sampling else 1.0
#         m_i_log_prob = self.agent.encoder_log_probs(enc_context, target_latent_space = c_context).flatten()
#         m_i = torch.mean(iS_factor*m_i_log_prob)
#         m_i_loss = -kappa*m_i
#         # Separatet the graphs to allow adam to choose different step sizes for the different losses (would not be the same if we just add the loss). 
#         m_i_loss.backward(retain_graph=True)

#         # Auxilliary decoder optimisation & gradient flow to encoder (optional, currently no gradient backprop to encoder model)
#         self.aux_optimizer.zero_grad()
#         aux_loss = -self.agent['xplor'].beta*torch.mean(xplor_aux_log_pi.flatten()) if self.decoupled else -self.agent.beta*torch.mean(aux_log_pi.flatten())
#         # aux_loss.backward(retain_graph=True)
#         aux_loss.backward()

#         # qf and encoder update (note encoder does not get grads from policy or vf)
#         self.qf1_optimizer.zero_grad()
#         self.qf2_optimizer.zero_grad()
#         rewards_flat = rewards.view(self.batch_size * num_tasks, -1)
#         # scale rewards for Bellman update
#         rewards_flat = rewards_flat * self.reward_scale
#         terms_flat = terms.view(self.batch_size * num_tasks, -1)
#         q_target = rewards_flat + (1. - terms_flat) * self.discount * target_v_values
#         qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
#         qf_loss.backward() 
#         # effective gradient E-step: Context and aux has Actor Critic loss, aux loss and KLD loss
#         self.context_optimizer.step()
#         self.aux_optimizer.step()

#         if self.decoupled:
#             self.xplor_qf1_optimizer.zero_grad()
#             self.xplor_qf2_optimizer.zero_grad()
#             # Mutual information
#             # with torch.no_grad():
#             xplor_rewards_flat = self.reward_scale * self.mutual_information(context, xplor_rewards, xplor_obs_, xplor_actions_, xplor_next_obs_, task_z=xplor_task_z, mbs=4)
#             # xplor_rewards_flat = xplor_rewards.view(t * b, -1)
#             xplor_terms = xplor_terms.view(self.batch_size * num_tasks, -1)
#             xplor_q_target = xplor_rewards_flat + (1. - xplor_terms) * self.discount * xplor_target_v_values
#             xplor_qf_loss = torch.mean((xplor_q1_pred - xplor_q_target) ** 2) + torch.mean((xplor_q2_pred - xplor_q_target) ** 2)
#             xplor_qf_loss.backward()

#         # effective gradient M-step: 
#         self.qf1_optimizer.step()
#         self.qf2_optimizer.step()
#         if self.decoupled:
#             self.xplor_qf1_optimizer.step()
#             self.xplor_qf2_optimizer.step()

#         # compute min Q on the new actions
#         min_q_new_actions = self._min_q(obs, new_actions, task_z, actor=True)
#         # vf update
#         v_target = min_q_new_actions - log_pi
#         vf_loss = self.vf_criterion(v_pred, v_target.detach())
#         self.vf_optimizer.zero_grad()
#         vf_loss.backward()
#         self.vf_optimizer.step()

#         if self.decoupled:
#             xplor_min_q_new_actions = self._min_q(xplor_obs, xplor_new_actions, xplor_task_z, actor=False)
#             xplor_v_target = xplor_min_q_new_actions - xplor_log_pi
#             xplor_vf_loss = self.vf_criterion(xplor_v_pred, xplor_v_target.detach())
#             self.xplor_vf_optimizer.zero_grad()
#             xplor_vf_loss.backward()
#             self.xplor_vf_optimizer.step()
#         self._update_target_network()

#         # policy update 
#         # n.b. policy update includes dQ/da
#         log_policy_target = min_q_new_actions
#         policy_loss = (log_pi - log_policy_target).mean()
#         mean_reg_loss = self.policy_mean_reg_weight * (policy_mean**2).mean()
#         std_reg_loss = self.policy_std_reg_weight * (policy_log_std**2).mean()
#         pre_tanh_value = policy_outputs[-2]
#         pre_activation_reg_loss = self.policy_pre_activation_weight * ((pre_tanh_value**2).sum(dim=1).mean())
#         policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
#         policy_loss = policy_loss + policy_reg_loss

#         pol_optimizer = self.pretrain_policy_optimizer if self.l2_reg else self.policy_optimizer
#         pol_optimizer.zero_grad()
#         # Retain graph to allow both losses to be reduced at better rates
#         policy_loss.backward(retain_graph = True)
#         # BC trajectory optimization
#         demo_policy_loss = -torch.mean(bc_log_pi) 
#         # policy_loss = policy_loss + demo_policy_loss
#         demo_policy_loss.backward()
#         pol_optimizer.step()

#         if self.decoupled:
#             xplor_log_policy_target = xplor_min_q_new_actions
#             xplor_policy_loss = (xplor_log_pi - xplor_log_policy_target).mean()

#             xplor_mean_reg_loss = self.policy_mean_reg_weight * (xplor_policy_mean**2).mean()
#             xplor_std_reg_loss = self.policy_std_reg_weight * (xplor_policy_log_std**2).mean()
#             xplor_pre_tanh_value = xplor_policy_outputs[-2]
#             xplor_pre_activation_reg_loss = self.policy_pre_activation_weight * ((xplor_pre_tanh_value**2).sum(dim=1).mean())
#             xplor_policy_reg_loss = xplor_mean_reg_loss + xplor_std_reg_loss + xplor_pre_activation_reg_loss
#             xplor_policy_loss = xplor_policy_loss + xplor_policy_reg_loss

#             self.xplor_policy_optimizer.zero_grad()
#             xplor_policy_loss.backward()
#             self.xplor_policy_optimizer.step()

#         # save some statistics for eval
#         if self.eval_statistics is None:
#             # eval should set this to None.
#             # this way, these statistics are only computed for one batch.
#             self.eval_statistics = OrderedDict()
#             if self.use_information_bottleneck:
#                 z_mean = np.mean(np.abs(ptu.get_numpy(self.agent['xplor'].z_means[0]))) if self.decoupled else np.mean(np.abs(ptu.get_numpy(self.agent.z_means[0])))
#                 z_sig = np.mean(ptu.get_numpy(self.agent['xplor'].z_vars[0])) if self.decoupled else np.mean(ptu.get_numpy(self.agent.z_vars[0]))
#                 self.eval_statistics['Z mean train'] = z_mean
#                 self.eval_statistics['Z variance train'] = z_sig
#                 self.eval_statistics['KL Divergence'] = ptu.get_numpy(kl_div)
#                 self.eval_statistics['KL Loss'] = ptu.get_numpy(kl_loss)

#             self.eval_statistics['QF Loss'] = np.mean(ptu.get_numpy(qf_loss))
#             self.eval_statistics['VF Loss'] = np.mean(ptu.get_numpy(vf_loss))
#             self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(policy_loss))
#             self.eval_statistics['Aux Loss'] = np.mean(ptu.get_numpy(aux_loss))
#             self.eval_statistics['bc_loss'] = np.mean(ptu.get_numpy(demo_policy_loss))
#             self.eval_statistics['info_loss'] = np.mean(ptu.get_numpy(m_i_loss))

#             if isinstance(self.writer, SummaryWriter):
#                 self.writer.add_scalar('loss/QF',np.mean(ptu.get_numpy(qf_loss)),self._n_train_steps_total)
#                 self.writer.add_scalar('loss/VF',np.mean(ptu.get_numpy(vf_loss)),self._n_train_steps_total)
#                 self.writer.add_scalar('loss/Aux',np.mean(ptu.get_numpy(aux_loss)),self._n_train_steps_total)
#                 self.writer.add_scalar('loss/Policy',np.mean(ptu.get_numpy(policy_loss)),self._n_train_steps_total)
#                 self.writer.add_scalar('loss/KL',ptu.get_numpy(kl_loss),self._n_train_steps_total)
#                 self.writer.add_scalar('loss/info', ptu.get_numpy(m_i_loss),self._n_train_steps_total)
#                 self.writer.add_scalar('loss/demo_bc ', ptu.get_numpy(demo_policy_loss),self._n_train_steps_total)

#             self.eval_statistics.update(create_stats_ordered_dict(
#                 'Q Predictions',
#                 ptu.get_numpy(q1_pred),
#             ))
#             self.eval_statistics.update(create_stats_ordered_dict(
#                 'V Predictions',
#                 ptu.get_numpy(v_pred),
#             ))
#             self.eval_statistics.update(create_stats_ordered_dict(
#                 'Log Pis',
#                 ptu.get_numpy(log_pi),
#             ))
#             self.eval_statistics.update(create_stats_ordered_dict(
#                 'Policy mu',
#                 ptu.get_numpy(policy_mean),
#             ))
#             self.eval_statistics.update(create_stats_ordered_dict(
#                 'Policy log std',
#                 ptu.get_numpy(policy_log_std),
#             ))

#     def get_epoch_snapshot(self, epoch):
#         # NOTE: overriding parent method which also optionally saves the env
#         if self.decoupled:
#             snapshot = OrderedDict(
#                 qf1=self.qf1.state_dict(),
#                 qf2=self.qf2.state_dict(),
#                 policy=self.agent['actor'].policy.state_dict(),
#                 vf=self.vf.state_dict(),
#                 target_vf=self.target_vf.state_dict(),
#                 xplor_qf1=self.xplor_qf1.state_dict(),
#                 xplor_qf2=self.xplor_qf2.state_dict(),
#                 xplor_policy=self.agent['xplor'].policy.state_dict(),
#                 xplor_vf=self.xplor_vf.state_dict(),
#                 xplor_target_vf=self.xplor_target_vf.state_dict(),
#                 context_encoder=self.context_encoder.state_dict(),
#                 aux_decoder = self.aux_decoder.state_dict(),
#             )
#         else:
#             snapshot = OrderedDict(
#                 qf1=self.qf1.state_dict(),
#                 qf2=self.qf2.state_dict(),
#                 policy=self.agent.policy.state_dict(),
#                 vf=self.vf.state_dict(),
#                 target_vf=self.target_vf.state_dict(),
#                 context_encoder=self.agent.context_encoder.state_dict(),
#                 aux_decoder = self.agent.aux_decoder.state_dict(),
#             )
#         return snapshot
