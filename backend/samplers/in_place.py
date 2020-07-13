import numpy as np

from backend.samplers.util import rollout
from backend.torch.PEARL.policies import MakeDeterministic


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self,
                        deterministic=False, 
                        max_samples=np.inf, 
                        max_trajs=np.inf, 
                        accum_context=True, 
                        resample=1,
                        writer = None,
                        iter_counter = None
                        ):
        """
        Obtains samples in the environment until we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths = []
        n_steps_total = 0
        n_trajs = 0

        while n_steps_total < max_samples and n_trajs < max_trajs:
            # if iter_counter is not None:
            #     iter_counter += n_steps_total
            # Rollout function is actually the collector
            if isinstance(iter_counter, int) and isinstance(n_steps_total, int):
                path = rollout(
                            self.env, 
                            policy, 
                            max_path_length=self.max_path_length, 
                            accum_context=accum_context, 
                            writer = writer, 
                            iter_counter=iter_counter + n_steps_total,
                            )
            else:
                path = rollout(
                            self.env, 
                            policy, 
                            max_path_length=self.max_path_length, 
                            accum_context=accum_context, 
                            writer = writer, 
                            )
            # save the latent space Z that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            if n_trajs % resample == 0:
                policy.sample_z()
        return paths, n_steps_total

