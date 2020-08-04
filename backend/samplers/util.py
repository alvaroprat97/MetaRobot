import numpy as np
from tensorboardX import SummaryWriter
import torch

def rollout(env, agent, 
            max_path_length=np.inf, 
            accum_context=True, 
            animated=False, 
            save_frames=False, 
            iter_counter = 0,
            writer = None,
            sparse_rewards = False,
            m_i_rewards =  False,
            m_i_params = None,
            ):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0
    path_reward = 0

    if sparse_rewards and m_i_rewards:
        raise ValueError("Cannot operate with sparse and mutual information rewards, choose one")
    if m_i_rewards is True or m_i_params is not None:
        raise NotImplementedError("Unstable use of rewards and encoder")

    # context = m_i_params['context']
    # gamma = m_i_params['gamma']
    # q_funcs = m_i_params['q_targets']
    # v_func = m_i_params['v_target']
    # use_next_obs_in_context = m_i_params['use_n_o']

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        x_i = [o, a, r, next_o, d, env_info]

        # Mutual information Collection (at collection, this becomes unstable)
        # if m_i_rewards:
        #     o_ = torch.FloatTensor([o])
        #     a_ = torch.FloatTensor([a])
        #     next_o_ = torch.FloatTensor([next_o])
        #     r_ = torch.FloatTensor([r]).unsqueeze(0)
        #     z_old = agent.z
        #     if use_next_obs_in_context:
        #         data = torch.cat([o_, a_, r_, next_o_], dim=1)
        #     else:
        #         data = torch.cat([o_, a_, r_], dim=1)
        #     if context is None:
        #         extended_context = data
        #     else:
        #         extended_context = torch.cat([context, data], dim=1)
        #     agent.infer_posterior(extended_context.unsqueeze(0))
        #     z_new = agent.z
        #     with torch.no_grad():
        #         target = r_ + gamma * v_func(next_o_, z_old)
        #         basis = torch.min(q_funcs[0](o_, a_, z_new), q_funcs[1](o_, a_, z_new))
        #         r_mi = torch.abs(basis - target)

        if sparse_rewards:
            if r<1:
                r = 0

        # update the agent's current context
        if animated:
            env.set_schedule(o, a)
            env.render()
        if accum_context:
            agent.update_context(x_i)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        path_reward += r
        o = next_o
        env_infos.append(env_info)
        if d:
            print(f"Reached Goal at path length {path_length}")
            break

    if animated:
        env.render()

    if isinstance(writer, SummaryWriter):
        # print(iter_counter + path_length)
        writer.add_scalar('scalars/PathLength', path_length, iter_counter + path_length)#, iter_counter + path_length)
        writer.add_scalar('scalars/PathReward', path_reward, iter_counter + path_length)#, iter_counter + path_length)
        # writer.add_scalar('action/action_x', a[0], )#, iter_counter + path_length)
        # writer.add_scalar('action/action_y', a[1], )#, iter_counter + path_length)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )

def rollout_window(env, agent, 
            max_path_length=np.inf, 
            context_window_length = 20,
            accum_context=True, 
            animated=False, 
            iter_counter = 0,
            writer = None,
            sparse_rewards = False
            ):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :param save_frames: if True, save video of rollout
    :return:
    """
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0
    path_reward = 0

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        if sparse_rewards:
            if r<1:
                r = 0
        # update the agent's current context
        if animated:
            env.set_schedule(o, a)
            env.render()
        if accum_context:
            agent.update_context([o, a, r, next_o, d, env_info])
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        path_reward += r
        o = next_o
        env_infos.append(env_info)

        if d:
            print(f"Reached Goal at path length {path_length}")
            break

        if path_length%context_window_length is 0:
            print("Updating Posterior Rollout ... \n")
            agent.infer_posterior(agent.context)

    if animated:
        env.render()

    if writer is not None:
        writer.add_scalar('action/action_x', a[0])#, iter_counter + path_length)
        writer.add_scalar('action/action_y', a[1])#, iter_counter + path_length)
        writer.add_scalar('action/action_y', a[1])#, iter_counter + path_length)
        writer.add_scalar('reward/reward', path_reward)#, iter_counter + path_length)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )

def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )


def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]

def expert_rollout(env, agent, 
            max_path_length=np.inf, 
            animated=False, 
            iter_counter = 0,
            writer = None,
            eval = False
            ):

    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    next_o = None
    path_length = 0
    path_reward = 0

    while path_length < max_path_length:
        a = agent.get_action(o, evaluate = eval)
        next_o, r, d, env_info = env.step(a)
        agent_info = 0
        if animated:
            env.set_schedule(o, a)
            env.render()

        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        path_reward += r
        o = next_o
        env_infos.append(env_info)
        if d:
            print(f"Reached Goal at path length {path_length}")
            break
    
    if animated:
        env.render()

    if not eval:
        if isinstance(writer, SummaryWriter):
            # print(iter_counter + path_length)
            writer.add_scalar('scalars/PathLength', path_length, iter_counter + path_length)#, iter_counter + path_length)
            writer.add_scalar('scalars/PathReward', path_reward, iter_counter + path_length)#, iter_counter + path_length)
            writer.add_scalar('action/action_x', a[0], iter_counter + path_length)
            writer.add_scalar('action/action_y', a[1], iter_counter + path_length)
    else:
        if isinstance(writer, SummaryWriter):
            writer.add_scalar('scalars/Eval/PathLength', path_length, iter_counter + path_length)#, iter_counter + path_length)
            writer.add_scalar('scalars/Eval/PathReward', path_reward, iter_counter + path_length)#, iter_counter + path_length)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )