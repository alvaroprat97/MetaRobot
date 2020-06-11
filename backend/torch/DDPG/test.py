from ddpg import DDPGagent
from utils import *

import sys
sys.path.insert(0,'../../envs/')

from PegRobot2D import Frontend, WINDOW_X, WINDOW_Y
from main import save_dir
from torch import load as torch_load
from numpy import load as np_load

def run_policy(agent, env):
    if isinstance(env, Frontend):
        del(env)
    env = Frontend(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False, visible = True)
    env.run_policy(agent)

if __name__ == "__main__":
    env = Frontend(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False, visible = True)
    myddpg = DDPGagent(env)
    pnl = np_load(f'{save_dir}param_noise.npy')
    myddpg.actor = torch_load(f'{save_dir}actor.pt')
    myddpg.actor_perturbed = torch_load(f'{save_dir}actor_perturbed.pt')

    run_policy(myddpg, env)
