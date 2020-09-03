import os 
import pickle
import numpy as np
import matplotlib.pyplot as plt 
from envs.utils import denorm_pos, norm_pos, denorm_point, ORIGIN
import matplotlib 
import seaborn as sns
import torch

exp_id = "2020_07_08_15_51_03"
expdir = 'output/Peg2D/{}/eval_trajectories/'.format(exp_id) # directory to load data from
epoch = 70
tlow = 40
thigh = 50
gr = 0.1

def load_pkl(task):
    with open(os.path.join(expdir, 'task{}-epoch{}-run0.pkl'.format(task, epoch)), 'rb') as f:
        data = pickle.load(f)
    return data

def load_pkl_prior():
    with open(os.path.join(expdir, 'prior-epoch{}.pkl'.format(epoch)), 'rb') as f:
        data = pickle.load(f)
    return data

def plot_prior_trajectores():
    paths = load_pkl_prior()
    goals = [norm_pos(load_pkl(task)[0]['goal'] - ORIGIN) for task in range(tlow, thigh)]

    plt.figure(figsize=(8,8))
    axes = plt.axes()
    axes.set(aspect='equal')
    plt.axis([-3, 2, -3, 3])
    for g in goals:
        circle = plt.Circle((g[0], g[1]), radius=gr)
        axes.add_artist(circle)
    rewards = 0
    final_rewards = 0
    for traj in paths:
        rewards += sum(traj['rewards'])
        final_rewards += traj['rewards'][-1]
        states = traj['observations']
        plt.plot(states[:-1, 0], states[:-1, 1], '-o')
        plt.plot(states[-1, 0], states[-1, 1], '-x', markersize=20)
        
def plot_trajectories(all_paths = None, num_trajs = 1):

    if all_paths is None:
        all_paths = []
        for task in range(tlow, thigh):
            paths = [t for t in load_pkl(task)]
            all_paths.append(paths)

    # color trajectories in order they were collected
    cmap = matplotlib.cm.get_cmap('binary')
    sample_locs = np.linspace(0, 0.9, num_trajs)
    colors = [cmap(s) for s in sample_locs]

    fig, axes = plt.subplots(5, 2, figsize=(20, 25))
    t = 0
    for j in range(2):
        for i in range(5): 
            axes[i, j].set_xlim([-1, 5])
            axes[i, j].set_ylim([-3, 3])
            for k, paths in enumerate(all_paths):
                alpha = 1 if k == t else 0.2
                try:
                    g = norm_pos(paths[0]['goal'] - ORIGIN)
                except:
                    g = norm_pos(paths[0]['goal_pos']- ORIGIN)
                circle = plt.Circle((g[0], g[1]), radius=gr, alpha=alpha)
                axes[i, j].add_artist(circle)
            indices = list(np.linspace(0, len(all_paths[t]), num_trajs, endpoint=False).astype(np.int))
            counter = 0
            for idx in indices:
                states = np.array(all_paths[t][idx]['observations'])
                axes[i, j].plot(states[:-1, 0], states[:-1, 1], '-', color=colors[counter])
                axes[i, j].plot(states[-1, 0], states[-1, 1], '-x', markersize=20, color=colors[counter])
                axes[i, j].set(aspect='equal')
                counter += 1
            t += 1

def traj_histo(trajectories, param = 'rewards', var_idx = None):
    """
    trajectories: list of trajectory(dict)
    """
    
    n_t = len(trajectories)
    assert n_t > 0
    
    for i, traj in enumerate(trajectories):
        goal_pos = norm_pos(traj['goal_pos'] - ORIGIN)
        variable = traj[param]
        variable = np.array(variable)
        try:
            kde = sns.kdeplot(variable[:,var_idx], shade = True, label = f"traj {i+1}")
        except:
            kde = sns.kdeplot(variable, shade = True, label = f"traj {i+1}")
    if var_idx is not None:
        plt.axvline(goal_pos[var_idx],0,1,label='goal',linestyle='--',c='r')
        plt.legend()
    plt.xlabel(f"{param}")
    plt.ylabel("Kernel Density")
    
def belief_plot(trajectories, plot_y = True):
    """
    trajectories: list of trajectory(dict)
    """
    mus = []
    sigmas = []
    ns = 0
    nprev = 0

    for trajectory in trajectories:
        for sample in trajectory['belief']:
            n = sample[1]
            mu = [sample[2]]*(n - nprev)
            sigma = [sample[3]]*(n - nprev)
            mus += mu
            sigmas += sigma
            ns += (n - nprev)
            nprev = n
            
    goal_x = trajectories[0]['goal_pos'].x
    goal_y = trajectories[0]['goal_pos'].y

    mu_x = torch.stack(mus)[:,0]
    mu_y = torch.stack(mus)[:,1]
    s_x = torch.stack(sigmas)[:,0,0]
    s_y = torch.stack(sigmas)[:,0,1]
    y_lower_bound = denorm_point(mu_y - 2*s_y, is_y = True)
    y_upper_bound = denorm_point(mu_y + 2*s_y, is_y = True)

    x_lower_bound = denorm_point(mu_x - 2*s_x, is_x = True)
    x_upper_bound = denorm_point(mu_x + 2*s_x, is_x = True)

    if plot_y:
        plt.plot(range(len(mu_y)),denorm_point(mu_y, is_y = True), 'b', label = 'belief')
        plt.fill_between(range(len(mu_y)), y_lower_bound, y_upper_bound, facecolor='blue', alpha=0.5,label='2$\sigma$')
        plt.ylim([150,550])
        plt.axhline(goal_y,0,1,label='goal y',linestyle='--',c='k')
        plt.ylabel("Belief of goal y", color = 'k')
        plt.xlabel("Transition step")
        plt.legend()
    else:                                                                
        plt.plot(range(len(mu_x)),denorm_point(mu_x, is_x = True), 'k', label = 'belief')
        plt.fill_between(range(len(mu_x)), x_lower_bound, x_upper_bound, facecolor='black', alpha=0.5,label='2$\sigma$')
        plt.ylim([900,1100])
        plt.axhline(goal_x,0,1,label='goal x',linestyle='--',c='k')
        plt.legend()
        plt.ylabel("Belief of goal x", color = 'k')
        plt.xlabel("Transition step")

    # beliefs = [np.array(t_bel)[:,0] for t_bel in [traj['belief'] for traj in trajectories]]
    # beliefs_x = [sample.x for sample in np.concatenate(beliefs)]
    # beliefs_y = [sample.y for sample in np.concatenate(beliefs)]
    # goal_y = trajectories[0]['goal_pos'].y
    # goal_x = trajectories[0]['goal_pos'].x

    # plt.plot(range(len(beliefs_y[1:])),beliefs_y[1:], 'k')
    # plt.ylim([200,600])
    # plt.axhline(goal_y,0,1,label='goal y',linestyle='--',c='k')
    # plt.ylabel("Belief of goal y", color = 'k')
    # plt.xlabel("Transition step")

    # plt.twinx()
    # plt.plot(range(len(beliefs_x[1:])),beliefs_x[1:], 'b')
    # plt.ylim([960, 1040])
    # plt.axhline(goal_x,0,1,label='goal x',linestyle='--',c='b')
    # plt.ylabel("Belief of goal x", color = 'b')

def latent_plot(trajectories):

    mus = []
    sigmas = []
    ns = 0
    nprev = 0
    zs = []
    z_mus = []
    z_vars = []

    for trajectory in trajectories:
        zs += trajectory['z']
        z_mus += trajectory['z_mu']
        z_vars += trajectory['z_vars']
        
    z = torch.stack(zs)[:,0,:].numpy()
    mu = torch.stack(z_mus)[:,0,:].numpy()
    sigma = torch.sqrt(torch.stack(z_vars))[:,0,:].numpy()

    # z_ = np.sum(z,1)
    # mu_ = np.sum(mu,1)
    # sigma_ = np.sum(sigma,1)
    nz = z.shape[-1]
    plt.figure()
    for i in range(nz):
        plt.plot(mu[:,i], label = f'$\mu_{i}$')
    plt.ylabel("Latent space sample", color = 'k')
    plt.xlabel("Transition step")
    plt.legend()
    plt.figure()
    for i in range(nz):
        plt.plot(sigma[:,i], label = f'$\sigma_{i}$')
    plt.ylabel("Latent space sample", color = 'k')
    plt.xlabel("Transition step")
    plt.legend()
    
#     fig, axes = plt.subplots(nz//2,2, figsize=(20, 25))
#     j = 0
#     i_ = 0
#     for i in range(nz):
#         z_ = z[:,i]
#         mu_ = mu[:,i]
#         sigma_ = sigma[:,i]
#         lower_bound = mu_ - 2*sigma_
#         upper_bound = mu_ + 2*sigma_
#         if i == nz//2:
#             j = 1
#             i_ = 0
#         print(i,j)
#         axes[i_,j].plot(range(len(mu_)),mu_, 'b', label = "$z$")
#         axes[i_,j].fill_between(range(len(mu_)), lower_bound, upper_bound, facecolor='blue', alpha=0.5,label='2$\sigma$')
#         # plt.ylim([0,-12])
# #         fig.ylabel("Sampled latent space", color = 'k')
# #         fig.xlabel("Transition step")
#         axes[i_,j].legend()
#         i_ += 1