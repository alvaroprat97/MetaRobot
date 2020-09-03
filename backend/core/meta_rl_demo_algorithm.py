# # Based on PEARL's backend https://github.com/katerakelly/oyster/tree/master/backend

import abc
import os
import sys
import time
from collections import OrderedDict
from tensorboardX import SummaryWriter
import gtimer as gt
import numpy as np
import torch

from backend.core import eval_util, logger
from backend.data_management.env_replay_buffer import MultiTaskReplayBuffer
from backend.data_management.path_builder import PathBuilder
from backend.samplers.in_place import InPlacePathSampler
from backend.torch import pytorch_util as ptu

# Demos
import pickle

sys.path.append("")

class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            demo_paths = None,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            num_steps_per_eval=1000,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            num_pretrain_steps = 0,
            replay_buffer_size=25000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
            imperfect_demo = False,
    ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.env = env
        self.agent = agent
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_pretrain_steps = num_pretrain_steps
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment

        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter
        self.writer = None
        self.write = False
        writer = self.writer
        self.write = True if isinstance(writer, SummaryWriter) else False
        self.imperfect_demo = imperfect_demo
        self.decoupled = True if isinstance(self.agent, dict) else False
        
        # Demos
        self.demos = demo_paths
        print(f"Operating demonstrations from {self.env.task_families} \n")

        # Actor config
        agent_actor = agent if not self.decoupled else agent['actor']
        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent_actor,
            max_path_length=self.max_path_length,
        )
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
                latent_dim=self.agent.latent_dim
            )
        
        self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks + self.eval_tasks,
                latent_dim = self.agent.latent_dim
        )       

        self.demo_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks + self.eval_tasks,
                latent_dim = self.agent.latent_dim
        )   

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []
        self.m_i_rewards = True if self.decoupled else False
        print(f"Acting on mutual information: {self.m_i_rewards}")

    def make_exploration_policy(self, policy):
         return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.choice(self.eval_tasks)
        else:
            idx = np.random.choice(self.train_tasks)
        return idx

    def load_demos(self):
        """
        load the demosntrations self.demos into the encoder replay buffer, for each training task.
        """
        print(f"Operating on a task families {self.env.task_families}")
        
        idx = 0
        for task_family in self.env.task_families:
            print(f"Loading on {task_family}: {len(self.demos[task_family])}")
            for i in range(0,50):
                task_idx = i + idx*50
                demo_paths = self.demos[task_family][i]
                for path in demo_paths:
                    path['context'] = [None]*len(path['terminals'])
                # Add to replay buffer too for pretraining.
                if task_idx in self.train_tasks:
                    for path in demo_paths:
                        path['rewards'] = [np.array(reward) for reward in path['rewards']]
                    self.replay_buffer.add_paths(task_idx, demo_paths)
                # Add to demo buffer (encoder buffer). Sparsify rewards.
                num_demo_trans = sum([len(path['terminals']) for path in demo_paths])
                for path in demo_paths:
                    path['rewards'] = np.array([[int(x) for x in path['terminals']]]).T
                self.demo_buffer.add_paths(task_idx, demo_paths)
                self.demo_buffer.set_ground(task_idx, num_demo_trans)
            idx += 1

    def train(self):
        '''
        meta-training loop
        '''
        # Load demonstrations into the encoder buffer. Set demonstrations as constant. Also into Replay Buffer.
        print("Loading Demos into Replay and Encoder buffer...")
        self.load_demos()

        # BC and Encoder Initialisation
        print("Pretraining on BC and Auxilliary Head..")
        self.pretrain()

        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)

        # Instantiate a path builder which adds samples and then concatenates them as traces
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(range(self.num_iterations), save_itrs=True):
            self._start_epoch(it_)
            # Set nets to training mode
            self.training_mode(True)
            print(f"Starting epoch {it_}")

            # Collect initial pool of data for train and eval before the first epoch.
            if it_ == 0:
                print('collecting initial pool of data for train and eval ...')
                # temp for evaluating
                for idx in self.train_tasks:
                    self.task_idx = idx
                    self.env.set_task_idx(idx)
                    # Collect on actor buffer B
                    if self.imperfect_demo:
                        self.env.alter_task(idx)
                    else:
                        self.env.reset_task(idx)
                    self.collect_data(num_samples = self.num_initial_steps, 
                                    resample_z_rate = 1, 
                                    update_posterior_rate = np.inf, 
                                    add_to_enc_buffer=True,
                                    clean_prior = False,
                                    add_to_demos= False,
                                    post_from_demos= False
                                    )                 

            # Sample data from all training tasks.
            # Save which tasks we collect from, to keep behavioural policy distribution at training time.
            rand_tasks =  []
            for i in range(self.num_tasks_sample):
                # Select a random task, ODD
                idx = np.random.choice(self.train_tasks)
                rand_tasks.append(idx)
                self.task_idx = idx
                self.env.set_task_idx(idx)
                # Reset the task environment & buffers for the randomly selected task
                if self.imperfect_demo:
                    self.env.alter_task(idx)
                else:
                    self.env.reset_task(idx)
                # Clear the encoder for task idx and keep the demonstrations
                self.enc_replay_buffer.task_buffers[idx].clear()

                # collect some trajectories with z ~ prior. We do not update posterior of z. 
                # This is what the encoder buffer contains
                if self.num_steps_prior > 0:
                    # print("Collecting Prior")
                    self.collect_data(self.num_steps_prior, 1, np.inf,
                                        clean_prior = True, add_to_demos = False) #
                # collect some trajectories with z ~ posterior. We update the posterior of z. 
                if self.num_steps_posterior > 0:
                    # print("Collecting Semi-Posterior")
                    self.collect_data(self.num_steps_posterior, 1, np.inf,
                                        clean_prior = False, add_to_demos = True)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior 
                # not used for encoder training
                if self.num_extra_rl_steps_posterior > 0:
                    # print("Collecting Posterior")
                    # Distributional shift fix: using exploratory prior z, save a posteriori transitions to the encoder too
                    # Z initialised with demos and updated per trajectory. 
                    self.collect_data(self.num_extra_rl_steps_posterior, 1, self.update_post_train, 
                                        add_to_enc_buffer=False, clean_prior = False, 
                                        add_to_demos= True) #

            # Use unique randomised tasks to collect data. 
            rand_tasks = np.unique(rand_tasks)

            # Sample train tasks and compute gradient updates on parameters.
            # print("ONLY DOING IL")
            for train_step in range(self.num_train_steps_per_itr):
                if train_step % 250 == 0:
                    print(f"At training step {train_step}")
                if train_step == 0:
                    # Update behaviour policy for importance sampling
                    self.update_behaviour_policy()
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                self._do_training(indices, rand_tasks)
                self._n_train_steps_total += 1

            gt.stamp('train')

            self.training_mode(False)

            # TODO
            # eval on collected ctxt
            self._try_to_eval(it_)
            gt.stamp('eval')

            self._end_epoch()

    def pretrain(self):
        """
        Imitation Learning (Behavioural Cloning) and encoder initialisation phase
        """
        # Initialise Context from Demonstrations. 
        # Already done as part of the Demo Encoder buffer mix. Infer latent context vector from demos.
        for _ in range(self.num_pretrain_steps):
            indices = np.random.choice(self.train_tasks, self.meta_batch)
            self._do_pretraining(indices)
        # Look at SAC training to sample batches first. 

    def demo_criterion(self, path):
        if path['observations'].shape[0] <= (self.demo_buffer.task_buffers[self.task_idx].num_demo_trans() // 
                        self.demo_buffer.task_buffers[self.task_idx].num_demos()):
            return True
        return False

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, 
                            add_to_enc_buffer=True, clean_prior = False, add_to_demos = False, post_from_demos = False):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        :clean_prior -> if we want to explore from the N(0,1) clean prior.
        '''
        # clar z to start from the prior (clean prior)
        batch_size = min(
                        self.enc_replay_buffer.task_buffers[self.task_idx].size(), 
                        self.enc_replay_buffer.task_buffers[self.task_idx].num_demo_trans(),
                        )
        batch_size = max(
                        self.embedding_batch_size,
                        batch_size
                        )
        
        if clean_prior:
            self.agent.clear_z()
        else:
            # TODO sample entire demo sequences without replacements
            context = self.sample_context(self.task_idx, demos = True, batch_size = batch_size)
            self.agent.demo_clear_z(context)

        num_transitions = 0
        # Context to be added if we succeed and the transitions are good.
        while num_transitions < num_samples:
            
            # Obtain paths for the task on self.task_idx
            sampler = self.sampler
            paths, n_samples = sampler.obtain_samples(deterministic = False,
                                                        accum_context = False,
                                                        max_samples=num_samples - num_transitions,
                                                        max_trajs=update_posterior_rate,
                                                        resample=resample_z_rate,
                                                        writer = self.writer,
                                                        iter_counter = self._n_env_steps_total + num_transitions,
                                                    )
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)

            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)

            # Extend Demos
            if add_to_demos:
                for path in paths:
                    if self.demo_criterion(path):
                        # Sparsify rewards for the context encoder
                        path['rewards'] = np.array([[int(x) for x in path['terminals'].flatten()]]).T
                        self.demo_buffer.add_path(self.task_idx, path)
                        print(f"Extending Demos on task {self.task_idx} to {self.demo_buffer.task_buffers[self.task_idx].num_episode_starts()}")

            if update_posterior_rate != np.inf:
                if post_from_demos:
                    num_demos = self.demo_buffer.task_buffers[self.task_idx].num_demos()
                    num_expert_trajs = self.demo_buffer.task_buffers[self.task_idx].num_episode_starts()
                    # When we have added some trajectories other than the demonstrations (demo augmentation), we want policy to account for this.
                    if num_expert_trajs > num_demos:
                        # print("Exploring On Expanded Demo Buffer...")
                        extra_context = self.sample_context(self.task_idx, demos = True)
                        if not clean_prior:
                            extra_context = torch.cat([extra_context, context], dim = 1)
                        self.agent.infer_posterior(extra_context)
                else:
                    extra_context = self.sample_context(self.task_idx, demos = False)
                    if not clean_prior:
                        extra_context = torch.cat([extra_context, context], dim = 1)
                    self.agent.infer_posterior(extra_context)

        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def _try_to_eval(self, epoch):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        agent = self.agent if not self.decoupled else self.agent['xplor']
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=agent,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run):
        self.task_idx = idx
        self.env.set_task_idx(idx)
        # Reset the task environment & buffers for the randomly selected task
        self.env.reset_task(idx)
        demo_context = self.sample_context(self.task_idx, demos = True, batch_size = self.embedding_batch_size)

        # Clear z and all contexts
        self.agent.demo_clear_z(demo_context)

        paths = []
        num_transitions = 0
        num_trajs = 0

        # print("\n EVALUATION ")
        while num_transitions < self.num_steps_per_eval:
            assert self.eval_deterministic is True
            path, num = self.sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.num_steps_per_eval - num_transitions, 
                                                    max_trajs=1, accum_context=True)
            paths += path
            num_transitions += num
            num_trajs += 1

            if num_trajs >= self.num_exp_traj_eval:
                self.agent.infer_posterior(self.agent.context)

        goal = self.env._get_goal(idx = idx)
        for path in paths:
            path['goal'] = goal # goal

        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))

        return paths

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def evaluate(self, epoch):
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        ### sample trajectories from prior for debugging / visualization
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            demo_context = self.sample_context(self.task_idx, demos = True, batch_size = self.embedding_batch_size)

            if self.decoupled:
                self.agent['xplor'].demo_clear_z(demo_context)
                self.agent['actor'].demo_clear_z(demo_context)
            else:
                self.agent.demo_clear_z(demo_context) 

            print("Saving prior trajectories from explorer")
            sampler = self.sampler if not self.decoupled else self.xplor_sampler
            prior_paths, _ = sampler.obtain_samples(deterministic=self.eval_deterministic, max_samples=self.max_path_length * 20,
                                                        accum_context=False,
                                                        resample=1)
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))

        ### train tasks
        # eval on a subset of train tasks for speed
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))

        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        train_returns_tasks = {key:[] for key in self.env.task_families}
        for idx in indices:
            task = self.env.task_descriptor[idx]
            self.task_idx = idx
            self.env.set_task_idx(idx)
            # Reset the task environment & buffers for the randomly selected task
            self.env.reset_task(idx)
            paths = []

            for num_trajs in range(self.num_steps_per_eval // self.max_path_length):
                demo_context = self.sample_context(idx, demos = True)
                self.agent.demo_clear_z(demo_context)
                p, _ = self.sampler.obtain_samples(deterministic=self.eval_deterministic, 
                                                    max_samples=self.max_path_length,
                                                    accum_context=False,
                                                    max_trajs=1,
                                                    resample=np.inf)
                paths += p

            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards
            train_returns_tasks[task].append(eval_util.get_average_returns(paths))
            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        for task in self.env.task_families:
            train_returns_tasks[task] = np.mean(train_returns_tasks[task])
        ### eval train tasks with on-policy data to match eval of test tasks
        train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)

        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)
        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)

        # save the final posterior
        if self.decoupled:
            self.agent['actor'].log_diagnostics(self.eval_statistics)
            self.agent['xplor'].log_diagnostics(self.eval_statistics)
        else:
            self.agent.log_diagnostics(self.eval_statistics)

        if hasattr(self.env, "log_diagnostics"):
            self.env.log_diagnostics(paths, prefix=None)

        avg_train_return = np.mean(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        for task in self.env.task_families:
            self.eval_statistics[f'R_{task}'] = train_returns_tasks[task]
        self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        if self.render_eval_paths:
            self.env.render_paths(paths)

        if self.plotter:
            self.plotter.draw()

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

    @abc.abstractmethod
    def _do_pretraining(self):
        """
        pretraining using the agent and the aux head to precondition the policy and the encoder
        """
        pass

    @abc.abstractmethod
    def update_behaviour_policy(self):
        "Update b from pi every 1st training step"
        pass

    @abc.abstractmethod
    def sample_context(self, indices):
        """
        Sample context from encoder buffers
        :return:
        """
        pass


