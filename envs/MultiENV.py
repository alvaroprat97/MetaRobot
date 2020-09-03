import sys
sys.path.append("")

import numpy as np
import pyglet
import pymunk
import torch
try:
    from pyglet.window import FPSDisplay, key
except:
    pass
from pymunk import Vec2d
import functools
from pymunk.pyglet_util import DrawOptions
import pymunkoptions
pymunkoptions.options["debug"] = False
from configs.tasks import rand_tasks
from envs.utils import WINDOW_X, WINDOW_Y, PEG_DEPTH, DT, TASK_TYPES, ORIGIN, REACH_ORIGIN, norm_pos, denorm_pos, denorm_point, norm_point
import envs.MetaPeg2D as Peg2D
import envs.MetaReach2D as Reach2D
import envs.MetaKey2D as Key2D
import envs.MetaStick2D as Stick2D

def get_env_backend(env_key):
    """
    Input: STRING - Descriptor of task family
    Return: TUPLE - static and robotic environemnts for the type of task family
    """
    assert env_key in TASK_TYPES
    if env_key is "Peg2D" or env_key is "Peg2Dv":
        _env = Peg2D
    elif env_key is "Reach2D":
        _env = Reach2D
    elif env_key is "Key2D" or env_key is "Key2Dv":
        _env = Key2D
    elif env_key is "Stick2D" or env_key is "Stick2Dv":
        _env = Stick2D
    else:
        raise NotImplementedError("Not implemented in dev yet")
    return _env.StaticEnvironment, _env.RobotEnvironment

class ENV(object):
    def __init__(self, expert = False, task_families = ["Reach2D", "Peg2D", "Peg2Dv", "Stick2D", "Stick2Dv"], sparse_rewards = False):
        print(f"Avoiding Dynamic Environment: Key2D ...\nTraining on {task_families} ...") 
        self.spaces = []
        self.statics = []
        self.robos = []
        self.task_descriptor = []
        self.expert = expert
        arm_lengths = [325 + 50, 275 + 110, 200 + 75]
        self.arm_lengths = arm_lengths
        self.task_families = task_families
        self.sparse_rewards = sparse_rewards
        GOALS = []
        print(expert, task_families, sparse_rewards)
        for task_family in task_families:
            task_GOALS = [Vec2d(GOAL) for GOAL in rand_tasks[task_family]]
            # Create task repertoire for Peg2D
            for GOAL in task_GOALS:
                space = pymunk.Space()
                space.gravity = (0, 0) 
                space.damping = 0.95
                self.spaces.append(space)
                static, robo = get_env_backend(task_family)
                if task_family is "Peg2Dv" or task_family is "Stick2Dv":
                    static = static(space, GOAL = GOAL, expert = expert, vertical = True, sparse = sparse_rewards)
                    robo = robo(space, arm_lengths= arm_lengths, radii = [5, 7, 10], GOAL=GOAL, expert = expert, vertical = True)
                elif task_family is "Key2D" or task_family is "Key2Dv":
                    if task_family is "Key2D":
                        static = static(space, GOAL = GOAL, expert = expert, THETA_GOAL = 0.9*np.pi/6, sparse = sparse_rewards)
                        robo = robo(space, arm_lengths= arm_lengths, expert = expert, radii = [5, 7, 10], GOAL=GOAL)
                    else:
                        static = static(space, GOAL = GOAL, expert = expert, THETA_GOAL = 0.9*np.pi/6, vertical = True, sparse = sparse_rewards)
                        robo = robo(space, arm_lengths= arm_lengths, expert = expert, radii = [5, 7, 10], GOAL=GOAL, vertical = True)
                else:
                    static = static(space, GOAL = GOAL, expert = expert, sparse = sparse_rewards)
                    robo = robo(space, arm_lengths= arm_lengths, radii = [5, 7, 10], GOAL=GOAL, expert = expert)
                self.statics.append(static)
                self.robos.append(robo)
                self.task_descriptor.append(task_family)
                del space, static, robo
            GOALS = GOALS + task_GOALS

        self.GOALS = GOALS
        self.targets = [norm_pos(GOAL) for GOAL in GOALS]
        self.tmp_targets = [norm_pos(GOAL) for GOAL in GOALS]

        self.action_range = {'low':self.robos[0].action_low, 'high':self.robos[0].action_high}
        self.observation_space = np.zeros([4])
        self.action_space = np.zeros([3])
        self.encountered = 0
        self.task_idx = 0
        # soft velocity polyak transition
        self.mu_avg = 0.2

    def set_task_idx(self, idx):
        self.task_idx = idx

    def get_ob_space(self):
        return self.observation_space

    def get_action_space(self):
        return self.action_space

    def get_all_task_idx(self):
        return range(len(self.spaces))

    def _env_setup(self, initial_qpos):
        raise Exception(NotImplemented)     
        
    def _get_obs(self, idx = 0):
        data = self.robos[idx].get_obs()  
        # GET DATA FROM SIMULATION ENVIRONMENT
        # RETURN OBSERVATION DIMENSION
        return data

    def _get_goal(self, idx = None):
        if idx is None:
            return self.statics[self.task_idx].tmp_GOAL 
        else:
            return self.statics[idx].tmp_GOAL

    def alter_task(self, idx = None, alteration = None):
        idx = self.task_idx if idx is None else idx
        self.statics[idx].alter_task(alteration)
        self.robos[idx].init_arms()
        self.robos[idx].init_robot()
        self.robos[idx].point_b, self.robos[idx].point_a = None, None
        self.robos[idx].tmp_GOAL = self.statics[idx].tmp_GOAL
        self.tmp_targets[idx] = norm_pos(self.statics[idx].tmp_GOAL)
        # print(f"Altering task {idx}, setting new GOAL {self.statics[idx].tmp_GOAL} from {self.statics[idx].GOAL}) and target {self.targets[idx]} from {norm_pos(self.statics[idx].GOAL)}")

    def reset(self):
        idx = self.task_idx
        task = self.task_descriptor[idx]
        if task is "Key2D" or task is "Key2Dv":
            self.statics[idx].full_reset()
            self.robos[idx].init_arms()
            self.robos[idx].init_robot()
            self.robos[idx].point_b, self.robos[idx].point_a = None, None
        else:
            self.robos[idx].reset_bodies(random_reset = True)
        obs = self.robos[idx].get_obs() 
        return obs

    def reset_tasks(self, idxs):
        obs = []
        for idx in idxs:
            task = self.task_descriptor[idx]
            if task is "Key2D" or task is "Key2Dv":
                self.statics[idx].full_reset()
                self.robos[idx].init_arms()
                self.robos[idx].init_robot()
                self.robos[idx].point_b, self.robos[idx].point_a = None, None
            else:
                self.robos[idx].reset_bodies(random_reset = True)
            o_ = self.robos[idx].get_obs() 
            obs.append(o_)
        self.encountered = 0
        return obs

    def reset_task(self, idx):
        task = self.task_descriptor[idx]
        if task is "Key2D" or task is "Key2Dv":
            self.statics[idx].full_reset()
            self.robos[idx].init_arms()
            self.robos[idx].init_robot()
            self.robos[idx].point_b, self.robos[idx].point_a = None, None
        else:
            self.robos[idx].reset_bodies(random_reset = True)
        self.encountered = 0
        return self.robos[idx].get_obs()

    def step(self, action, dt = DT, step = 0):
        "Watch out with action dimensions and Normalisation wrappers"

        dummy = 0
        denormalised_action = self.robos[self.task_idx].denorm_action(action)
    
        self.update(dt = dt, action=denormalised_action)

        new_obs = self.robos[self.task_idx].get_obs()
        reward, done = self.reward_func(new_obs)

        return new_obs, reward, done, dummy

    def update(self, dt = DT, action = None):
        if action is not None:
            peg = self.robos[self.task_idx].bodies[-1]
            
            for _ in range(20): # WAS 30
                # print(action, peg.velocity, peg.angular_velocity)
                peg.velocity = (1-self.mu_avg)*peg.velocity + self.mu_avg*Vec2d(action[0], action[1])
                peg.angular_velocity = (1-self.mu_avg)*peg.angular_velocity + self.mu_avg*action[2]
                self.spaces[self.task_idx].step(dt/20)
        else:
            peg = self.robos[self.task_idx].bodies[-1]
            # print(peg.force, peg.torque)
            for _ in range(20):
                self.spaces[self.task_idx].step(dt/20)
            # print(peg.force, peg.torque)


    def reward_func(self, obs):
        return self.statics[self.task_idx].reward_func(obs)

    def _get_targets(self, indices, altered = False):
        if altered:
            targets = [self.tmp_targets[i] for i in indices]
        else:
            targets = [self.targets[i] for i in indices]
        return targets


class VisualiserWrapper(pyglet.window.Window, ENV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # ENV.__init__(self, expert = True)
        # print("Warning, Overriding TRUE EXPERT, remove for meta-training")
        # ENV.__init__(self, expert = False, task_families = ["Reach2D", "Stick2D", "Peg2D", "Peg2Dv", "Stick2Dv", "Key2D", "Key2Dv"])
        # ENV.__init__(self, expert = True, task_families = ["Key2Dv", "Peg2Dv", "Stick2Dv"])
        # ENV.__init__(self, expert=False,task_families = ["Key2Dv","Reach2D", "Stick2D", "Peg2D", "Peg2Dv", "Stick2Dv", "Key2D"])
        ENV.__init__(self, expert=False,task_families = ["Peg2D"])

        print("Not using experts, in for meta-training... \n")

        self.set_visibles()
        self.options = DrawOptions()
        self.tmp_trans = dict(
            obs = None,
            action = None
        )
        self.old_action = None
        self.old_obs = None
        self.traj_rollout_counter = 0
        self.rollout_counter = 0
        self.current_traj_belief = None
        self.all_paths = []
        self.target_line = []
        self.det_draw = False

    def init_trajectory(self):
        self.trajectory = {}
        self.trajectory['observations'] = [] #.append(obs)
        self.trajectory['actions'] = [] #.append(action_raw)
        self.trajectory['next_observations'] = [] #.append(next_obs)
        self.trajectory['rewards'] = [] #.append(r)
        self.trajectory["terminals"] = []
        self.trajectory["agent_infos"] = []
        self.trajectory["env_infos"] = []
        self.trajectory['belief'] = [] 
        self.trajectory['z_mu'] = []
        self.trajectory['z'] = []
        self.trajectory['z_vars'] = []

    def run_policy(self, agent):
        self.set_visible(visible = True)
        self.set_visibles()
        self.agent = agent
        pyglet.clock.schedule_interval(self.policy_update, 1/60)
        pyglet.app.run()

    def run_path(self, path):
        self.set_position(path['observations'][0])
        self.path_rollout = path
        self.n_actions = len(self.path_rollout['actions'])
        self.iter_action = 0
        self.set_visible(visible = True)
        self.set_visibles()
        pyglet.clock.schedule_interval(self.path_update, 1/60)
        pyglet.app.run()

    def set_position(self, obs):
        """Set the position initially for the path rollouts"""
        peg_body = self.robos[self.task_idx].bodies[-1]
        assert peg_body.id == 'peg'
        print(obs[0]*WINDOW_X + ORIGIN[0], obs[1]*WINDOW_Y + ORIGIN[1], np.arcsin(obs[-1]))

        peg = Vec2d(0,0)
        peg.x = (obs[0]*WINDOW_X + ORIGIN[0])
        peg.y = (obs[1]*WINDOW_Y + ORIGIN[1])
        
        peg_body.position = peg.x - 0.5*self.arm_lengths[-1]*Vec2d(obs[-2], obs[-1])
        peg_body.angle = np.arcsin(obs[-1])        

        for body in self.robos[self.task_idx].bodies:
            body.force = 0, 0
            body.torque = 0
            body.velocity = 0, 0
            body.angular_velocity = 0

    def get_all_task_idx(self):
        return range(len(self.spaces))

    def set_visibles(self):
        if self.visible:
            self.set_location(50,50) # Location To Initialise window
            self.fps = FPSDisplay(self)
            self.show = True
            self.show_iter = 0

    def reset_class(self, *args, **kwargs):
        self.__init__(*args, **kwargs)

    def draw_target(self):
        self.statics[self.task_idx].draw_target()

    def remove_target(self):
        self.statics[self.task_idx].remove_target()

    def save_transition(self, transition, agent = None):
        obs, action_raw, r, next_obs, d = transition
        self.trajectory['observations'].append(obs)
        self.trajectory['actions'].append(action_raw)
        self.trajectory['next_observations'].append(next_obs)
        self.trajectory['rewards'].append(r)
        self.trajectory['terminals'].append(d)
        self.trajectory['agent_infos'].append(0)
        self.trajectory['env_infos'].append(0)
        if agent is not None:
            self.trajectory['z_mu'].append(agent.z_means.detach())
            self.trajectory['z'].append(agent.z.detach())
            self.trajectory['z_vars'].append(agent.z_vars.detach())

    def infer_belief(self, agent, infer = True, prior = False):
        aux_posterior, sampled_belief, belief = None, None, None
        if infer:
            if not prior:
                # full_context = agent.context
                # _, nc, _ = full_context.size()
                # size = min(nc, 128)
                # memo_idx = 0 
                # idxes = np.random.randint(memo_idx, nc, size = size)
                # context = full_context[:,idxes,:]
                agent.infer_posterior(agent.context)
                # agent.infer_posterior(agent.context)
            else:
                agent.sample_z()
            aux_posterior = agent.aux_decoder(agent.z)
            sampled_belief = aux_posterior[0].flatten()
            belief = denorm_pos(Vec2d(tuple(sampled_belief.numpy())))
            self.current_traj_belief = tuple((belief, self.rollout_counter, sampled_belief, aux_posterior[5].detach()))
        self.trajectory['belief'].append(self.current_traj_belief)
        return aux_posterior, sampled_belief, belief

    def save_current_trajectory(self):
        self.trajectory['rollout_length'] = self.traj_rollout_counter
        self.trajectory['task_idx'] = self.task_idx
        self.trajectory['goal_pos'] = self.statics[self.task_idx].tmp_GOAL
        self.trajectories.append(self.trajectory)
        # Reset the trajectories
        self.traj_rollout_counter = 0
        self.init_trajectory()

    def rollout_update(self, dt, agent, accum_context, max_steps, sparse_rewards, continuous_update, deterministic = False):
        self.det_draw = deterministic
        if self.rollout_counter == 0:
            self.trajectories = []
            self.init_trajectory()
            self.infer_belief(agent, infer=True, prior=True)
        if continuous_update:
            if self.rollout_counter%max_steps == 0 and self.rollout_counter > 0:
                self.save_current_trajectory()
                self.finish_rollout()
                return
        else:
            if self.traj_rollout_counter%max_steps == 0 and self.traj_rollout_counter > 0:
                self.save_current_trajectory()
                self.finish_rollout()
                return
        obs = np.array(self.robos[self.task_idx].get_obs()) 
        action_raw, _ = agent.get_action(obs, deterministic)
        next_o, r, d, env_info = self.step(action_raw)
        self.save_transition(tuple((obs, action_raw, r, next_o, d)), agent = agent)
        if sparse_rewards:
            r = 0
            if d:
                r = 1
        if accum_context:
            agent.update_context([obs, action_raw, r, next_o, d, env_info])
        # else:
        #     self.accum = False
        # TODO in configs: context_window
        context_window = 25
        if continuous_update and self.traj_rollout_counter%context_window is 0:
            self.infer_belief(agent, infer = True, prior = False)
            print(f"Updating continuous belief at traj {self.traj_rollout_counter} and roll {self.rollout_counter}")
        elif not continuous_update:
            if d or (self.traj_rollout_counter + 1)%max_steps is 0:
                self.infer_belief(agent, infer = True, prior = False)
                print(f"Updating trajctory belief at traj {self.traj_rollout_counter} and roll {self.rollout_counter}")
            # elif (self.traj_rollout_counter + 1)%context_window is 0:
            #     self.infer_belief(agent, infer = True, prior = False)
        else:
            self.infer_belief(agent, infer=False, prior = False)
        if d:
            if continuous_update:
                # Save trajectory in trajectories list for current task & reset trajectory
                self.save_current_trajectory()
                # Reset environment
                self.reset()
            else:
                self.save_current_trajectory()
                # get out of trajectory
                self.finish_rollout()
        self.rollout_counter += 1
        self.traj_rollout_counter += 1

    def expert_rollout_update(self, dt, agent, max_steps):
        if self.rollout_counter == 0:
            self.trajectories = []
            self.init_trajectory()
        else:
            if self.traj_rollout_counter%max_steps == 0 and self.traj_rollout_counter > 0:
                self.save_current_trajectory()
                self.finish_rollout(done = False)
                return
        # assert self.GOALS[self.task_idx] == self.statics[self.task_idx].tmp_GOAL

        obs = np.array(self.robos[self.task_idx].get_obs()) 
        obs_ = norm_pos(self.statics[self.task_idx].tmp_GOAL - denorm_pos(Vec2d(obs[0], obs[1])))
        obs_ = np.array([obs_[0], obs_[1], obs[2], obs[3]])

        action_raw = agent.get_action(obs)/2
        next_o, r, d, _ = self.step(action_raw)
        next_o_ = norm_pos(self.statics[self.task_idx].tmp_GOAL - denorm_pos(Vec2d(next_o[0], next_o[1])))
        next_o_ = np.array([next_o_[0], next_o_[1], next_o[2], next_o[3]])
        r_ = r

        self.save_transition(tuple((obs_, action_raw, r_, next_o_, d)))

        if d:
            self.save_current_trajectory()
            self.finish_rollout(done = True)
        self.rollout_counter += 1
        self.traj_rollout_counter += 1

    def view_rollout(self, agent, accum_context, max_steps, sparse_rewards = False, continuous_update = False, deterministic = False):
        self.traj_rollout_counter = 0
        # self.accum = accum_context
        self.draw_target()
        print("Speeding out DT by factor 5")
        pyglet.clock.schedule_interval(self.rollout_update, DT/20, agent, accum_context, max_steps, sparse_rewards, continuous_update, deterministic)
        pyglet.app.run()

    def view_expert_rollout(self, agent, max_steps):
        self.done_traj = False
        self.traj_rollout_counter = 0
        print("Speeding out DT by factor 10")
        pyglet.clock.schedule_interval(self.expert_rollout_update, DT/20, agent, max_steps)
        pyglet.app.run()

    def finish_rollout(self, done = False):
        print("Ending Task ... \n")
        # if not self.continuous_update:
        self.done_traj = done
        pyglet.app.exit()
        if not self.expert:
            pyglet.clock.unschedule(self.rollout_update)
        else:
            pyglet.clock.unschedule(self.expert_rollout_update)

    def render(self):
        pyglet.app.run()

    def set_schedule(self, obs, action_raw):
        self.tmp_trans['obs'] = obs
        self.tmp_trans['action'] = self.robos[self.task_idx].denorm_action(action_raw)
        
    def policy_update(self, dt, obs = None, action_raw = None):
        obs = np.array(self.robos[self.task_idx].get_obs()) if obs is None else obs
        action_raw = self.agent.get_action(obs, deterministic = True)[0] if action_raw is None else action_raw
        action = self.robos[self.task_idx].denorm_action(action_raw)
        # print(action, pos)
        pos = self.robos[self.task_idx].get_peg_tip()
        if pos.x >self.statics[self.task_idx].tmp_GOAL.x + 150:
            pass
        # print("MADE IT, CONGRATS POLICY")
        self.update(action = action, dt = dt)

    def path_update(self, dt):
        if self.iter_action < self.n_actions:
            action = self.path_rollout['actions'][self.iter_action] 
            if action is not None:
                action = self.robos[self.task_idx].denorm_action(action)
                pos = self.robos[self.task_idx].get_peg_tip()
                if self.iter_action == 0:
                    print(pos)
                # print(self.iter_action, action, pos)
                if pos.x > self.statics[self.task_idx].tmp_GOAL.x + 150:
                    pass
                    # print("MADE IT, CONGRATS PATH")
                else:
                    self.update(action = action, dt = dt)
            self.iter_action += 1

    # @_draw_decorator      
    def on_draw(self):
        # if self.check:
        if self.visible:
            self.clear() # clear the buffer
            # Order matters here!
            self.spaces[self.task_idx].debug_draw(self.options)
            self.label = pyglet.text.Label(f'TASK {self.task_idx}',
                            font_size=36,
                            x=WINDOW_X*0.9, y=WINDOW_Y*0.95,
                            anchor_x='center', anchor_y='center')
            self.label.draw()
            self.fps.draw()

    def on_key_press(self, symbol, modifiers):

        if symbol == key.ESCAPE:
            pyglet.app.exit()
            # self.close()
            return True

        bodies = self.robos[self.task_idx].bodies

        for idx, body in enumerate(bodies):
            if idx is 0 and body.id is not 'root':
                raise Exception("Wrong body root")
            if body.id is f'Arm{idx + 1}' and 0 < idx < len(bodies):
                raise Exception("Wrong body listing arms")
            if body.id is not 'peg' and idx is len(bodies):
                raise Exception("Wrong body listing peg")

        if symbol == key.C:
            action = np.random.sample(3)*2-1
            self.robos[self.task_idx].action_buffered = self.robos[self.task_idx].denorm_action(action)

        if symbol == key.S:
            self.alter_task()

        # print(bodies[-1].force, bodies[-1].torque, bodies[-1].angle)

        c = 100
        if symbol == key.UP:
            bodies[-1].velocity = Vec2d(0,c)
        if symbol == key.DOWN:
            bodies[-1].velocity = -Vec2d(0,c)
        if symbol == key.LEFT:
            bodies[-1].velocity = -Vec2d(c,0)
        if symbol == key.RIGHT:
            bodies[-1].velocity = Vec2d(c,0)
        # print(bodies[-1].velocity, bodies[-1].angular_velocity)
        v = np.pi*5/16
        if symbol == pyglet.window.key.Q:
            bodies[-1].angular_velocity += v
        if symbol == pyglet.window.key.A:
            bodies[-1].angular_velocity -= v

        if symbol == key.R:
            self.reset()

        if symbol == key.Z:
            self.draw_target()
        
        if symbol == key.X:
            self.remove_target()

        if symbol == key.T:
            self.task_idx = self.task_idx + 1 if self.task_idx + 1 < len(self.spaces) else  0
            print(self.task_idx)
            self.reset()
    
        if symbol == key.P:
            pyglet.image.get_buffer_manager().get_color_buffer().save('RobotArm.png')

    def on_mouse_press(self, x, y, button, modifier):
        print(self.reward_func(self._get_obs(self.task_idx)))
        print(self._get_obs(self.task_idx))
        # print(self.spaces[self.task_idx].bodies[-1].velocity)
        point_q = self.spaces[self.task_idx].point_query_nearest((x,y), 0, pymunk.ShapeFilter())
        if point_q:
            print(point_q.shape, point_q.shape.body)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.robos[self.task_idx].bodies[-1].velocity = -(self.robos[self.task_idx].bodies[-1].position - Vec2d(x,y))


if __name__ == "__main__":
    env = VisualiserWrapper(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False)
    env.show = True
    env.task_idx = 0
    pyglet.clock.schedule_interval(env.update, 1/60)
    pyglet.app.run()