import sys
sys.path.append("")

import numpy as np
import pyglet
import pymunk
import torch
from pyglet.window import FPSDisplay, key
from pymunk import Vec2d
import functools
from pymunk.pyglet_util import DrawOptions
import pymunkoptions
pymunkoptions.options["debug"] = False

from envs.MetaPeg2D import RobotEnvironment, StaticEnvironment, WINDOW_X, WINDOW_Y, ORIGIN, PEG_DEPTH


class ENV(object):
    def __init__(self):

        self.spaces = []
        self.statics = []
        self.robos = []

        GOALS = [Vec2d(GOAL) for GOAL in [(975, 250), 
                                        (1025, 300),
                                        (995, 550),
                                        (1005, 350),
                                        (975, 150),
                                        (975, 450),
                                        (1025, 350),
                                        (1000, 400),
                                        (975, 300),
                                        (1010, 380),
                                        (970, 230),
                                        (990, 350),
                                        (960, 500),
                                        (1010, 200),
                                        (1020, 180),
                                        (965, 300),
                                        (1000, 250),
                                        (1025, 500),
                                        (990, 400),
                                        (985, 500),]
                                        ]
        # Create task repertoire
        for GOAL in GOALS:
            space = pymunk.Space()
            space.gravity = (0, 0) 
            space.damping = 0.975
            self.spaces.append(space)

            static = StaticEnvironment(space, GOAL = GOAL)
            self.statics.append(static)

            robo = RobotEnvironment(space, arm_lengths=[325, 260, 200], radii = [5, 7, 10], GOAL=GOAL)
            self.robos.append(robo)

            del space, static, robo

        self.GOALS = GOALS
        self.action_range = {'low':self.robos[0].action_low, 'high':self.robos[0].action_high}
        self.observation_space = np.zeros([4])
        self.action_space = np.zeros([3])

        self.encountered = 0
        self.task_idx = 0

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
        data = self.robos[idx].get_obs() # GET DATA FROM SIMULATION ENVIRONMENT
        # RETURN OBSERVATION DIMENSION
        return data

    def _get_goal(self, idx = None):
        if idx is None:
            return self.GOALS[self.task_idx]
        else:
            return self.GOALS[idx]

    def reset(self):
        self.robos[self.task_idx].reset_bodies(random_reset = True)
        obs = self.robos[self.task_idx].get_obs()
        return obs

    def reset_tasks(self, idxs):
        obs = []
        for idx in idxs:
            self.robos[idx].reset_bodies(random_reset = True)
            obs.append(self.robos[idx].get_obs())
        self.encountered = 0
        return obs

    def reset_task(self, idx):
        self.robos[idx].reset_bodies(random_reset = True)
        self.encountered = 0
        return self.robos[idx].get_obs()

    def step(self, action, dt = 1/60, step = 0):
        "Watch out with action dimensions and Normalisation wrappers"

        dummy = 0
        done = False
        denormalised_action = self.robos[self.task_idx].denorm_action(action)

        self.update(dt=dt, action=denormalised_action)

        new_obs = self.robos[self.task_idx].get_obs()
        reward, done = self.reward_func(done, new_obs)

        return new_obs, reward, done, dummy

    def update(self, dt = 1/60, action = None):

        if action is not None:
            peg = self.robos[self.task_idx].bodies[-1]
            peg.velocity = action[0], action[1]
            peg.angular_velocity = action[2]

        for r in range(10):
            self.spaces[self.task_idx].step(dt)

    def reward_func(self, done, obs):

        mask = done

        peg_tip = self.robos[self.task_idx].get_peg_tip()

        rpos_x = (self.GOALS[self.task_idx].x - peg_tip.x)/WINDOW_X
        rpos_y = (self.GOALS[self.task_idx].y - peg_tip.y)/WINDOW_Y

        reward =  -Vec2d(rpos_x, rpos_y).length + 0.1*abs(obs[-2]-1)         

        if peg_tip.x > 975 and abs(peg_tip.y - self.GOALS[self.task_idx].y) < 25 and abs(obs[-1]) < 0.1: #
            reward = min(reward + 0.05, 0)
        if peg_tip.x > self.GOALS[self.task_idx].x + 50: #
            reward = min(reward + 0.1, 0.05)
        if peg_tip.x > self.GOALS[self.task_idx].x + 100:
            reward = 10
            self.encountered += 1
            mask = True
            print(f"\n MADE IT {self.encountered} TIMES TO GOAL \n")

        return reward, mask


class VisualiserWrapper(pyglet.window.Window, ENV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ENV.__init__(self)
        self.set_visibles()
        self.options = DrawOptions()

    def policy_update(self, dt):
        obs = np.array(self.robos[self.task_idx].get_obs())
        action_raw, _ = self.agent.get_action(obs, deterministic = True)
        action = self.robos[self.task_idx].denorm_action(action_raw)

        pos = self.robos[self.task_idx].get_peg_tip()
        # print(action, pos)
        if pos.x >1100:
            print(pos, self.robos[self.task_idx].bodies[-1].position)
            print("MADE IT, CONGRATS")

        self.update(action = action, dt = dt)

    def run_policy(self, agent):
        self.set_visible(visible = True)
        self.set_visibles()
        self.agent = agent
        pyglet.clock.schedule_interval(self.policy_update, 1/60)
        pyglet.app.run()

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

    def _draw_decorator(func):
        @functools.wraps(func)
        def wrapped(inst, *args, **kwargs):
            if not inst.check():
                return
            return #func(inst, *args, **kwargs)
        return wrapped

    def check(self):
        if self.show or self.show_iter%250 == 0:
            self.show_iter = 0
            return True
        else:
            return False

    # @_draw_decorator
    def on_draw(self):
        # if self.check:
        self.clear() # clear the buffer
        # Order matters here!
        self.spaces[self.task_idx].debug_draw(self.options)
        self.fps.draw()

    def on_key_release(self, symbol, modifiers):
        for body in self.space.bodies:
            body.velocity *= 0.25*Vec2d(1,1)
            body.angular_velocity *= 0.5

    def on_key_press(self, symbol, modifiers):

        if symbol == key.ESCAPE:
            pyglet.app.exit()
            self.close()
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

        v = 0.2
        if symbol == key.Q:
            bodies[-1].angular_velocity += v
        if symbol == key.A:
            bodies[-1].angular_velocity -= v
        if symbol == key.W:
            bodies[-2].angular_velocity += v
        if symbol == key.S:
            bodies[-2].angular_velocity -= v
        if symbol == key.E:
            bodies[-1].angular_velocity += v
        if symbol == key.D:
            bodies[-1].angular_velocity -= v

        c = 10
        if symbol == key.UP:
            bodies[-1].velocity = Vec2d(0,c)
        if symbol == key.DOWN:
            bodies[-1].velocity = -Vec2d(0,c)
        if symbol == key.LEFT:
            bodies[-1].velocity = -Vec2d(c,0)
        if symbol == key.RIGHT:
            bodies[-1].velocity = Vec2d(c,0)

        if symbol == key.R:
            self.reset()

        if symbol == key.T:
            self.task_idx = self.task_idx + 1 if self.task_idx + 1 < len(self.spaces) else  0
            print(self.task_idx)
            self.reset()
            # self.robos[self.task_idx].reset_bodies()
    
        if symbol == key.P:
            pyglet.image.get_buffer_manager().get_color_buffer().save('RobotArm.png')

    def on_mouse_press(self, x, y, button, modifier):
        print(self.reward_func(False, self._get_obs(self.task_idx)))
        point_q = self.spaces[self.task_idx].point_query_nearest((x,y), 0, pymunk.ShapeFilter())
        if point_q:
            print(point_q.shape, point_q.shape.body)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.robos[self.task_idx].bodies[-1].velocity = -0.1*(self.robos[self.task_idx].bodies[-1].position - Vec2d(x,y))


if __name__ == "__main__":
    env = VisualiserWrapper(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False)
    env.show = True
    env.task_idx = 0
    pyglet.clock.schedule_interval(env.update, 1/60)
    pyglet.app.run()