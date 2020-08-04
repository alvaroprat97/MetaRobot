from tensorboardX import SummaryWriter


class BatchRLAlgorithm(object):
    def __init__(
            self,
            batch_size,
            max_path_length,
            num_epochs,
            num_eval_steps_per_epoch,
            num_expl_steps_per_train_loop,
            num_trains_per_train_loop,
            num_train_loops_per_epoch = 1,
            min_num_steps_before_training = 0,
            prioritised_experience = True,
            importance_sampling = True,
            until_done = False,
            force_training_steps = False
    ):

        self.batch_size = batch_size
        self.max_path_length = max_path_length
        self.num_epochs = num_epochs
        self.num_eval_steps_per_epoch = num_eval_steps_per_epoch
        self.num_trains_per_train_loop = num_trains_per_train_loop
        self.num_train_loops_per_epoch = num_train_loops_per_epoch
        self.num_expl_steps_per_train_loop = num_expl_steps_per_train_loop
        self.min_num_steps_before_training = min_num_steps_before_training
        self.path_type = ["random", "exploration", "eval"]
        self.memory_steps = 0
        self.updates = 0
        self.prioritised_experience = prioritised_experience
        self.importance_sampling = importance_sampling
        self.force_training_steps = force_training_steps
        self.until_done = until_done # Exploration stopped when reached goal / ended episode

    def collect_paths(self,
                      env,
                      agent,
                      path_type,
                      writer_tb = None,
                      reset_state = True,
                      until_done = False):

        """Collect path rollouts for different path types"""

        assert path_type in self.path_type

        if path_type == "random":
            num_steps = self.min_num_steps_before_training
            sampler = env.random_action
            evaluate = None
        elif path_type == "exploration":
            num_steps = self.num_expl_steps_per_train_loop
            sampler = agent.get_action
            evaluate = False
        elif path_type == "eval":
            num_steps = self.num_eval_steps_per_epoch
            sampler = agent.get_action
            evaluate = True
        else:
            print("No Exploration Type Specified, No Collection... \n")
            return None

        env._max_episode_steps = self.max_path_length

        if reset_state:
            state = env.reset()

        done = False

        episode_step = 0
        episode_reward = 0
        # curr_mem_len = len(agent.replay_buffer)
        paths = []

        for step in range(num_steps):

            action = sampler() if path_type is "random" else sampler(state, evaluate)

            next_state, reward, done, _ = env.step_func(action, step = episode_step)
            mask = 1 if episode_step == self.max_path_length else float(not done)
            transition = [state, action, reward, next_state, mask]
            # paths.append(transition)

            if path_type is not "eval":
                agent.replay_buffer.push(*transition, weighted = self.prioritised_experience)

                if isinstance(writer_tb, SummaryWriter):
                    writer_tb.add_scalar('action/action_1', action[0], self.memory_steps)
                    writer_tb.add_scalar('action/action_2', action[1], self.memory_steps)
                    writer_tb.add_scalar('angular/action_2', action[2], self.memory_steps)

            episode_reward += reward

            if not done:
                state = next_state
            else:
                state = env.reset()
                if isinstance(writer_tb, SummaryWriter) and path_type is not "random":
                    writer_tb.add_scalar(f"rewards/{path_type}", episode_reward, self.memory_steps)

                if until_done and episode_step + 1 != self.max_path_length and done:
                    print(f"Done Here at step {step + 1}")
                    return step%self.max_path_length

                episode_reward = 0
                episode_step = 0

            episode_step += 1

            if path_type is not "eval":
                self.memory_steps += 1

        return None

    def train(self, env, agent, writer_tb = None):
        if self.min_num_steps_before_training > 0:

            print("Initial Exploration ...")

            init_expl_paths = self.collect_paths(
                              env,
                              agent,
                              path_type = "random",
                              writer_tb = writer_tb,
                              reset_state = True,
                              until_done = False
                              )

            print(f"Finished Initial Exploration on {self.min_num_steps_before_training} steps \n")

        for epoch in range(0, self.num_epochs):

            # print("Evaluating Paths... ")

            _ = self.collect_paths(
                              env,
                              agent,
                              path_type = "eval",
                              writer_tb = writer_tb,
                              reset_state = True,
                              until_done = True
                              )

            for _ in range(self.num_train_loops_per_epoch):

                # print("Exploring Paths...")

                # Extract Exploration paths from policy
                steps = self.collect_paths(
                                  env,
                                  agent,
                                  path_type = "exploration",
                                  writer_tb = writer_tb,
                                  reset_state = True,
                                  until_done = self.until_done
                                  )

                num_trains = steps if isinstance(steps, int) else self.num_trains_per_train_loop

                # Perform some training loops
                for _ in range(num_trains):
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(self.batch_size,
                                                                                                         self.updates,
                                                                                                         PER = self.prioritised_experience,
                                                                                                         importance_sampling = self.importance_sampling,
                                                                                                         )

                    if isinstance(writer_tb, SummaryWriter):
                        writer_tb.add_scalar('loss/critic_1', critic_1_loss, self.updates)
                        writer_tb.add_scalar('loss/critic_2', critic_2_loss, self.updates)
                        writer_tb.add_scalar('loss/policy', policy_loss, self.updates)
                        # writer_tb.add_scalar('loss/entropy_loss', ent_loss, self.updates)
                        # writer_tb.add_scalar('entropy_temprature/alpha', alpha, self.updates)

                    self.updates += 1

            print(f"Finished Epoch {epoch}, replay size {len(agent.replay_buffer)}")
