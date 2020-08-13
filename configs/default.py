# default PEARL experiment settings
# all experiments should modify these settings only as needed

default_config = dict(
    env_name='Peg2D',
    n_train_tasks=40,
    n_eval_tasks=10,
    modality = 'train', # dep
    test_idx = 0, # dep
    GNN_encoder = True,
    aux_loss = True,
    demo_path =  "/content/drive/My Drive/meta_robot/" + "expert/Peg2D/ExpertPeg2DPaths",  #
    # demo_path = "expert/Peg2D/ExpertPeg2DPaths", 
    latent_size=6, # dimension of the latent context vector
    net_size=64, # Was 32 # number of units per FC layer in each network
    path_to_weights= None, # "/content/drive/My Drive/meta_robot/" + "output/Peg2D/2020_06_18_11_48_25/", # path to pre-trained weights to load into networks
    env_params=dict(
        n_tasks=50, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=False, # shuffle the tasks after creating them
    ),
    algo_params=dict(
        meta_batch= 16, # number of tasks to average the gradient across, also for pretraining
        num_iterations= 500, # number of data sampling / training iterates (epochs = iterations)
        num_pretrain_steps = 250,
        num_initial_steps= 100, # number of transitions collected per task before training
        num_tasks_sample= 8, # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=0, # number of transitions to collect per task with z ~ prior N(0,1)
        num_steps_posterior=200, # number of transitions to collect per task with z ~ posterior from demonstration.
        num_extra_rl_steps_posterior=200, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_train_steps_per_itr=500, # number of meta-gradient steps taken per iteration
        num_evals=3, # number of independent evals
        num_steps_per_eval= 100,  # nuumber of transitions to eval on
        batch_size=128, # number of transitions in the RL batch
        embedding_batch_size=64, # number of transitions in the context batch
        embedding_mini_batch_size=64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=50, # max path length for this environment
        discount=0.99, # RL discount factor
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        aux_lr = 3E-4,
        context_lr=3E-4,
        reward_scale=10.0, # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        sparse_rewards=False, # whether to sparsify rewards as determined in env
        kl_lambda= 0.1, # weight on KL divergence term in encoder loss,
        bc_lambda = 0.4,
        info_lambda = 0.2,      
        aux_lambda = 5.0,
        use_information_bottleneck=True, # False makes latent context deterministic
        use_next_obs_in_context=True, # use next obs if it is useful in distinguishing tasks
        l2_reg = True, # If we want to regularise behavioural cloning during meta-training. Default is True during pre-training. 
        update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
        num_exp_traj_eval=3, # how many exploration trajs to collect before beginning posterior sampling at test time
        recurrent=False, # recurrent or permutation-invariant encoder
        dump_eval_paths=False, # whether to save evaluation trajectories
        imperfect_demo = True,
    ),
    util_params=dict(
        base_log_dir= "/content/drive/My Drive/meta_robot/output",
        run_dir = "/content/drive/My Drive/meta_robot/",
        # base_log_dir = "output",
        # run_dir = "",
        use_gpu=True,
        gpu_id=0,
        debug=False, # debugging triggers printing and writes logs to debug directory
        docker=False, # TODO docker is not yet supported
    ),
    adapt_params=dict(
        continuous = False, # Adapt continuously at test time or not
        context_window_steps = 20, # Window to adapt context online
    ),
    LatentGNN = dict(
        latent_dims = [6, 6],
        num_kernels = 2,
        hidden_dict = dict(
            in_ = [128, 128],
            out_ = [32, 32]
        ),
    ),
    aux_params = dict(
        belief_dim = 2,
        hidden = [32, 16], #[]
        use = True,
        fixed_std = False,
        aux_std = None, # None if we want to learn this
        decay = False, # Option to decay the loss over time, allowing "optimising for what we want, reducing constraints"
    ),
    decoupled_config = dict(
        use = False,
        xplor = dict(
            recurrent = False,
        ),
        actor = dict(
            recurrent = False,
        )
    )
)

# # default PEARL experiment settings
# # all experiments should modify these settings only as needed

# default_config = dict(
#     env_name='Peg2D',
#     n_train_tasks=40,
#     n_eval_tasks=10,
#     modality = 'train', # dep
#     test_idx = 0, # dep
#     GNN_encoder = True,
#     aux_loss = True,
#     demo_path =  "/content/drive/My Drive/meta_robot/" + "expert/Peg2D/ExpertPeg2DPaths",  #
#     # demo_path = "expert/Peg2D/ExpertPeg2DPaths", 
#     latent_size=4, # dimension of the latent context vector
#     net_size=256, # Was 32 # number of units per FC layer in each network
#     path_to_weights= None, #"/content/drive/My Drive/meta_robot/" + "output/Peg2D/2020_08_07_11_57_21/", # path to pre-trained weights to load into networks
#     env_params=dict(
#         n_tasks=50, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
#         randomize_tasks=False, # shuffle the tasks after creating them
#     ),
#     algo_params=dict(
#         meta_batch= 16, # number of tasks to average the gradient across, also for pretraining
#         num_iterations= 100, # number of data sampling / training iterates (epochs = iterations)
#         num_pretrain_steps = 250,
#         num_initial_steps= 100, # number of transitions collected per task before training
#         num_tasks_sample= 16, # number of randomly sampled tasks to collect data for each iteration
#         num_steps_prior=400, # number of transitions to collect per task with z ~ prior
#         num_steps_posterior=0, # number of transitions to collect per task with z ~ posterior
#         num_extra_rl_steps_posterior=200, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
#         num_train_steps_per_itr=1000, # number of meta-gradient steps taken per iteration
#         num_evals=3, # number of independent evals
#         num_steps_per_eval= 100,  # nuumber of transitions to eval on
#         batch_size=128, # number of transitions in the RL batch
#         embedding_batch_size=64, # number of transitions in the context batch
#         embedding_mini_batch_size=64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
#         max_path_length=50, # max path length for this environment
#         discount=0.99, # RL discount factor
#         soft_target_tau=0.005, # for SAC target network update
#         policy_lr=3E-4,
#         qf_lr=3E-4,
#         vf_lr=3E-4,
#         aux_lr = 3E-4,
#         context_lr=3E-4,
#         reward_scale=10.0, # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
#         sparse_rewards=False, # whether to sparsify rewards as determined in env
#         kl_lambda= 0.1, # weight on KL divergence term in encoder loss,
#         bc_lambda = 0.1,
#         info_lambda = 0.2,
#         aux_lambda = 2.0,
#         use_information_bottleneck=True, # False makes latent context deterministic
#         use_next_obs_in_context=True, # use next obs if it is useful in distinguishing tasks
#         l2_reg = True, # If we want to regularise behavioural cloning during meta-training. Default is True during pre-training. 
#         update_post_train=1, # how often to resample the context when collecting data during training (in trajectories)
#         num_exp_traj_eval=3, # how many exploration trajs to collect before beginning posterior sampling at test time
#         recurrent=False, # recurrent or permutation-invariant encoder
#         dump_eval_paths=False, # whether to save evaluation trajectories
#     ),
#     util_params=dict(
#         base_log_dir= "/content/drive/My Drive/meta_robot/output",
#         run_dir = "/content/drive/My Drive/meta_robot/",
#         use_gpu=True,
#         gpu_id=0,
#         debug=False, # debugging triggers printing and writes logs to debug directory
#         docker=False, # TODO docker is not yet supported
#     ),
#     adapt_params=dict(
#         continuous = False, # Adapt continuously at test time or not
#         context_window_steps = 20, # Window to adapt context online
#     ),
#     LatentGNN = dict(
#         latent_dims = [6, 6],
#         num_kernels = 2,
#         hidden_dict = dict(
#             in_ = [64, 64],
#             out_ = [32, 32]
#         ),
#     ),
#     aux_params = dict(
#         belief_dim = 2,
#         hidden = [32, 16], #[]
#         use = True,
#         fixed_std = False,
#         aux_std = None, # None if we want to learn this
#         decay = False, # Option to decay the loss over time, allowing "optimising for what we want, reducing constraints"
#     ),
#     decoupled_config = dict(
#         use = False,
#         xplor = dict(
#             recurrent = False,
#         ),
#         actor = dict(
#             recurrent = False,
#         )
#     )
# )