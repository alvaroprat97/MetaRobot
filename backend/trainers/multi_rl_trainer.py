from tensorboardX import SummaryWriter

import os
import sys
import time
from collections import OrderedDict
from tensorboardX import SummaryWriter
import gtimer as gt
import numpy as np
# import abc

sys.path.append("../../")
sys.path.append("")

from backend.core import eval_util, logger
from backend.data_management.env_replay_buffer import MultiTaskReplayBuffer
from backend.data_management.path_builder import PathBuilder
from backend.samplers.in_place import ExpertInPlacePathSampler
from backend.torch import pytorch_util as ptu

class MultiTaskRLAlgorithm(object):

    """
    Multi task trainer for the expert. Used only to train expert SAC policies in order to retreive expert demonstrations during meta-RIL.
    """

    def __init__(self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            task_name = None,
            num_epochs= 25,
            num_tasks_sample = 8,
            num_eval_steps_per_epoch= 250,
            num_train_loops_per_epoch = 10,
            num_trains_per_train_loop= 250, 
            num_expl_steps_per_train_loop = 1000,
            num_steps_before_training = 100,
            replay_buffer_size = 200000,
            max_path_length= 50,
            batch_size = 256,
            num_train_tasks = 8,
            prioritised_experience = True,
            importance_sampling = True):

        self.num_epochs = num_epochs
        self.num_train_tasks = num_train_tasks
        self.num_tasks_sample = num_tasks_sample
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop 
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.num_steps_before_training = num_steps_before_training
        self.max_path_length = max_path_length
        self.batch_size = batch_size
        self.replay_buffer_size = replay_buffer_size
        self.per = prioritised_experience
        self.importance_sampling = importance_sampling
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.writer = None
        self.env = env
        self.agent = agent
        self.task_name = task_name
        self.path_type = ["random", "exploration", "eval"]
        self.memory_steps = 0
        self.updates = 0
        self._n_env_steps_total = 0
        self.task_idx = 0

        self.replay_buffer = MultiTaskReplayBuffer(
            self.replay_buffer_size,
            env,
            self.train_tasks,
        )
        self.sampler = ExpertInPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

    def collect_paths(self, num_samples, deterministic=False, eval = False):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples
        '''
        num_transitions = 0
        while num_transitions < num_samples:
            
            paths, n_samples = self.sampler.obtain_samples(deterministic = deterministic,
                                                            max_samples=num_samples - num_transitions,
                                                            writer = self.writer,
                                                            iter_counter = self._n_env_steps_total + num_transitions,
                                                            eval = eval
                                                            )
            num_transitions += n_samples
            if not deterministic:
                for path in paths:
                    path['context'] = [None]*len(path['terminals'])
                self.replay_buffer.add_paths(self.task_idx, paths)

        self._n_env_steps_total += num_transitions
        
    def collect_data(self,
                    path_type
                    ):

        """Collect path rollouts for different path types"""

        assert path_type in self.path_type
        if path_type == "random":
            num_steps = self.num_steps_before_training
            evaluate = False
        elif path_type == "exploration":
            num_steps = self.num_expl_steps_per_train_loop
            evaluate = False
        elif path_type == "eval":
            num_steps = self.num_eval_steps_per_epoch
            evaluate = True
        else:
            print("No Exploration Type Specified, No Collection... \n")
            return None

        self.collect_paths(num_steps, deterministic=evaluate, eval = evaluate)

    def train(self, writer):

        self.writer = writer
        assert isinstance(writer, SummaryWriter)
        self.agent.replay_buffer = self.replay_buffer
        self.agent.batch_size = self.batch_size

        print("Collecting Initial buffer tranisitons... ")
        for idx in self.train_tasks:
            self.task_idx = idx
            self.env.set_task_idx(idx)
            self.env.reset_task(idx)
            self.collect_data("random")

        print(f"Finished Initial Exploration on {self.num_steps_before_training} steps per task \n")

        for epoch in range(self.num_epochs):

            for _ in range(self.num_train_loops_per_epoch):

                for _ in range(self.num_tasks_sample):
                    idx = np.random.randint(len(self.train_tasks))
                    self.task_idx = idx
                    self.env.set_task_idx(idx)
                    self.env.reset_task(idx)
                    self.collect_data("exploration")

                # Perform some training loops
                for _ in range(self.num_trains_per_train_loop):
                    indices = np.random.choice(self.train_tasks, self.num_train_tasks)
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = self.agent.update_parameters(self.updates, indices)

                    if isinstance(self.writer, SummaryWriter):
                        self.writer.add_scalar('loss/critic_1', critic_1_loss, self.updates)
                        self.writer.add_scalar('loss/critic_2', critic_2_loss, self.updates)
                        self.writer.add_scalar('loss/policy', policy_loss, self.updates)
                        self.writer.add_scalar('loss/entropy_loss', ent_loss, self.updates)
                        self.writer.add_scalar('entropy_temprature/alpha', alpha, self.updates)

                    self.updates += 1

            print(f"Finished Epoch {epoch}")

            # On eval, we want to NOT save the transiitons but also do save the rewards 
            # such that we can evaluate determinstically how well it is doing.
            for idx in self.eval_tasks:
                self.task_idx = idx
                self.env.set_task_idx(idx)
                self.env.reset_task(idx)
                self.collect_data("eval")

            self.agent.save_model(self.task_name)