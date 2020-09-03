# """
# MT/MLT CONFIG
# """

# MTX
MULT = 2
DEBUG = 1
default_config = dict(
    env_name='ablation_info',
    n_train_tasks=40,
    n_eval_tasks=10,
    modality = 'train', # dep
    test_idx = 0, # dep
    GNN_encoder = True,
    aux_loss = True,
    demo_path =  "/content/drive/My Drive/meta_robot/" + "expert/AllMultiDemos2D",  
    # demo_path = "expert/AllMultiDemos2D", 
    latent_size= 2, # dimension of the latent context vector
    net_size=128*MULT, # Was 32 # number of units per FC layer in each network
    path_to_weights= None, # "/content/drive/My Drive/meta_robot/" + "output/Peg2D/2020_06_18_11_48_25/", # path to pre-trained weights to load into networks
    env_params=dict(
        n_tasks=50, # number of distinct tasks in this domain, shoudl equal sum of train and eval tasks
        randomize_tasks=False, # shuffle the tasks after creating them
    ),
    algo_params=dict(
        meta_batch= 16*MULT//DEBUG, # number of tasks to average the gradient across, also for pretraining
        num_iterations= 100, # number of data sampling / training iterates (epochs = iterations)
        num_pretrain_steps = 1000*MULT, #1000//DEBUG,#*4,
        num_initial_steps= 60*2, # number of transitions collected per task before training
        num_tasks_sample= 16*MULT//DEBUG,#*4, # number of randomly sampled tasks to collect data for each iteration
        num_steps_prior=0, #60*4//DEBUG, # number of transitions to collect per task with z ~ prior N(0,1)
        num_steps_posterior= 60*4//DEBUG, # number of transitions to collect per task with z ~ posterior from demonstration.
        num_extra_rl_steps_posterior=60*4//DEBUG, # number of additional transitions to collect per task with z ~ posterior that are only used to train the policy and NOT the encoder
        num_train_steps_per_itr=1000*MULT//DEBUG, # number of meta-gradient steps taken per iteration
        num_evals=1, # number of independent evals
        num_steps_per_eval= 60,  # nuumber of transitions to eval on
        batch_size=256, # number of transitions in the RL batch
        embedding_batch_size=64, # number of transitions in the context batch
        embedding_mini_batch_size=64, # number of context transitions to backprop through (should equal the arg above except in the recurrent encoder case)
        max_path_length=60, # max path length for this environment
        discount=0.99, # RL discount factor
        soft_target_tau=0.005, # for SAC target network update
        policy_lr=3E-4,
        qf_lr=3E-4,
        vf_lr=3E-4,
        aux_lr = 3E-4,
        context_lr=3E-4,
        reward_scale=25, #10.0, # scale rewards before constructing Bellman update, effectively controls weight on the entropy of the policy
        sparse_rewards=False, # whether to sparsify rewards as determined in env
        kl_lambda= 0.1, # weight on KL divergence term in encoder loss,
        bc_lambda = 0.5, #0.5,
        info_lambda = 0.2,      
        aux_lambda = 4.0,
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
        use_gpu = True,
        # base_log_dir = "output",
        # run_dir = "",
        # use_gpu=False,
        
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
            in_ = [128*MULT, 128*MULT],
            out_ = [32*MULT, 32*MULT]
        ),
    ),
    aux_params = dict(
        belief_dim = 2,
        # hidden = [32, 16],
        hidden = [32, 24], #[]
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