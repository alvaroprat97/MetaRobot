import sys
sys.path.append("")

import pyglet
import pymunk
import pymunkoptions
from pymunk import constraint
from pymunk.pyglet_util import DrawOptions

# FPS Display is not available for colab... 
try:    
    from pyglet.window import key, FPSDisplay
except:
    pass

from math import degrees
from pymunk import Vec2d
from numpy import cos, sin
from numpy.random import random_sample
from numpy import pi as np_pi
import numpy as np
import functools
import torch
pymunkoptions.options["debug"] = False
from envs.utils import WINDOW_X, WINDOW_Y, PEG_DEPTH, DT, TASK_TYPES, ORIGIN, REACH_ORIGIN, GOAL, norm_pos, denorm_pos, denorm_point, norm_point, Arms, PivotJoint

GOAL_RADIUS = 20

class StaticEnvironment(object):
    def __init__(self, space, GOAL, sprites = None, expert = False, sparse = False):
        self.sprites = sprites
        self.space = space
        self.GOAL = GOAL
        self.tmp_GOAL = GOAL
        self.ORIGIN = REACH_ORIGIN
        self.theta = 0.0
        self.tmp_theta = 0.0
        self.expert = expert
        self.sparse = sparse
        self.target_sphere = []        
        self.env_statics = []
        self.encountered = 0
        self.make_static_env()

    def alter_task(self, alterations = None):
        """
        Modify task by some extent. 
        This is performed in evaluation (usually) in order to see if it overfits to the demonstration or not.
        If performed during meta-training then this would be "LEARNING FROM IMPERFECT DEMONSTRATIONS"
        """
        # Remove shapes from body
        for shape in self.space.shapes:
            self.space.remove(shape)
        for constraint in self.space.constraints:
            self.space.remove(constraint)

        self.env_statics = []
        if alterations is None:
            alterations = dict(
                peg_pos_x = (np.random.rand()-0.5)*WINDOW_X/40,
                peg_pos_y = (np.random.rand()-0.5)*WINDOW_Y/40
            )
        self.make_static_env(alterations = alterations)

    def make_static_env(self, alterations = None):
        """
        seg_list is a list of tuples containing args for add_static_segment
        settings is a hard-coded dictionary of task descriptors.
        """
        # Static Coordinates for peg
        if alterations is None:
            peg_pos_x = self.GOAL.x 
            mid_frame = self.GOAL.y
        else:
            # print("Altering the task...")
            peg_pos_x = self.GOAL.x + alterations['peg_pos_x']
            mid_frame = self.GOAL.y + alterations['peg_pos_y']

        self.goal_pos = Vec2d(peg_pos_x, mid_frame)
        self.tmp_GOAL = self.goal_pos

    def draw_target(self):
        try:
            self.remove_target()
        except:
            pass
        finally:
            body = self.space.static_body
            target_sphere = pymunk.Circle(body, radius = GOAL_RADIUS, offset=self.tmp_GOAL)
            target_sphere.filter = pymunk.ShapeFilter(group=1)
            target_sphere.color = (255, 0, 0)
            self.target_sphere.append(target_sphere)
            self.space.add(target_sphere)

    def remove_target(self):
        self.space.remove(self.target_sphere)
        self.target_sphere = []
        
    def reward_func(self, obs):

        mask = False
        r_peg_tip = denorm_pos(Vec2d(obs[0], obs[1]))
        if self.expert:
            r_peg_tip = self.tmp_GOAL - denorm_pos(Vec2d(obs[0], obs[1]))
        peg_tip = r_peg_tip + self.ORIGIN
        
        rpos_x = (self.tmp_GOAL.x - peg_tip.x)/WINDOW_X
        rpos_y = (self.tmp_GOAL.y - peg_tip.y)/WINDOW_Y
        reward = -2*Vec2d(rpos_x, rpos_y).length 

        if (peg_tip - self.tmp_GOAL).length < GOAL_RADIUS//2:
            mask = True
        if self.sparse:
            return int(mask), mask
        return reward, mask

class RobotEnvironment(object):
    def __init__(self, space, arm_lengths = [250, 200, 200], radii = [5, 5, 10], GOAL = (1000, 400), expert = False):
        self.space = space
        self.GOAL = GOAL
        self.tmp_GOAL = GOAL
        self.ORIGIN = REACH_ORIGIN
        self.PivotPoints = [None]*(1+len(arm_lengths))
        self.PivotPoints[0] = (self.ORIGIN,0)
        self.Arms = Arms(arm_lengths, radii = radii)
        self.range_multiplier = 5
        self.set_action_range(vals = [(2, 20), (1, np_pi/8)]) # 2 Velocities and 1 angular velocity
        self.expert = expert
        self.init_arms()
        self.init_robot()
        self.obs = self.get_obs()
        self.point_b, self.point_a = None, None
        
    def init_arms(self):
        self.random_arm_generator()
        # self.joints = []

    def set_action_range(self, vals):
        # Max velocity actions
        nums = np.array([val[0] for val in vals])
        highs = []
        for val in vals:
            for idx in range(val[0]):
                highs.append(val[1])
        high = np.array(highs)
        self.num_action_types = nums
        self.action_high = high*self.range_multiplier
        self.action_low = -high*self.range_multiplier
        self.action_buffered = np.zeros((sum(nums),))

    def random_arm_generator(self):
        got = False
        iterator = 1
        while got is False:
            len_x = 0
            len_y = 0
            for idx in range(self.Arms.n):
                rand_angle = 2*np_pi*(random_sample() - 0.5)
                if idx is self.Arms.n:
                    raise ValueError("Wrong init")
                length = self.Arms.lengths[idx]
                len_x += length*cos(rand_angle)
                len_y += length*sin(rand_angle)
                self.Arms.pos[idx] = (length, rand_angle)
            peg_tip = Vec2d(self.ORIGIN[0] + len_x, self.ORIGIN[1] + len_y)
            if (0 < (len_x + self.ORIGIN[0]) < WINDOW_X and 0 < (self.ORIGIN[1] + len_y) < WINDOW_Y) and (self.tmp_GOAL - peg_tip).length > 250:
                got = True
                if iterator > 1000:
                    print(f"Got at {iterator}")
                    raise ArithmeticError("Cannot get an environment like that, change the origin")
            else:
                iterator += 1
                self.Arms.pos = [None]*self.Arms.n

    def get_vertices(self, arm_data):
        thc = self.Arms.thickness
        return [(-arm_data[0]//2,-thc),(arm_data[0]//2,-thc),(arm_data[0]//2,thc),(-arm_data[0]//2,thc)]

    def get_pos(self, arm_data, idx):
        self.arm_vector_curr = Vec2d((arm_data[0]*cos(arm_data[1]), arm_data[0]*sin(arm_data[1])))
        self.Arms.arm_vecs[idx] = self.arm_vector_curr
        if idx is 0:
            self.point_a = self.ORIGIN
        else:
            self.point_a = self.point_b
        self.point_b = self.point_a + self.arm_vector_curr
        return self.point_a + self.arm_vector_curr//2

    def get_transform(self, arm_data):
        t = pymunk.Transform(a = cos(arm_data[1]), c = -sin(arm_data[1]), tx = 0,
                             b = sin(arm_data[1]), d = cos(arm_data[1]), ty = 0)
        return t

    def save_pivot(self, idx):
        self.PivotPoints[idx+1] = (self.point_a + self.arm_vector_curr, idx+1)

    def reset_bodies(self, random_reset = False):
        if not random_reset:
            for body in self.space.bodies:
                if not hasattr(body, 'start_position'):
                    continue
                body.position = Vec2d(body.start_position)
                body.force = 0, 0
                body.torque = 0
                body.velocity = 0, 0
                body.angular_velocity = 0
                body.angle = body.start_angle
        else:
            self.init_arms()
            idx = 0
            for idx, arm_data in enumerate(self.Arms.pos):

                pos = self.get_pos(arm_data, idx)
                angle = arm_data[1]
                body = self.bodies[idx+1]
                body.position = pos
                body.start_position = pos
                body.angle = arm_data[1]
                body.start_angle = arm_data[1]
                self.save_pivot(idx)
                body.velocity = 0, 0
                body.angular_velocity = 0
                body.start_position = pos
                body.start_angle = angle
                body.position =  pos
                body.force = 0, 0
                body.torque = 0
                body.angle = angle
                idx += 1

    def init_robot(self):

        b0 = self.space.static_body
        b0.id = 'root'
        self.bodies = [b0]
        self.joints = []
        self.motors = []

        for idx, arm_data in enumerate(self.Arms.pos):
            vertices = self.get_vertices(arm_data)
            pos = self.get_pos(arm_data, idx)
            t = self.get_transform((0, 0))
            self.save_pivot(idx)

            # Make peg heavy to gain control
            if idx+1 is len(self.Arms.pos):
                mass = 1
                moment = 10
            else:
                mass = 1
                moment = 10
            body = pymunk.Body(mass=mass, moment=moment)
            body.position = pos
            body.start_position = pos
            body.angle = arm_data[1]
            body.start_angle = arm_data[1]

            body.id = 'peg' if idx+1 is len(self.Arms.pos) else f'arm{idx+1}'
            body.velocity_func = self.limit_velox
            shape = pymunk.Poly(body, vertices, t,  radius = self.Arms.radii[idx])
            shape.filter = pymunk.ShapeFilter(group = 1)
            shape.elasticity = 0.5
            shape.color = (0, 255, 0, 0)
            shape.friction = 0.8
            shape.density = 1

            self.space.add(body, shape)
            self.bodies.append(body)

        self.add_constraints()

    def limit_velox(self, body, gravity, damping, dt):
        max_velox = 100
        max_velox_angular = np_pi
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        l = body.velocity.length
        if l > max_velox:
            try:
                scale = max_velox/l
                body.velocity = body.velocity*scale
            except:
                print('OVERPOWERED VELOCITY')

        w = body.angular_velocity
        if abs(w) > max_velox_angular:
            try:
                body.angular_velocity = max_velox_angular*w/abs(w)
            except:
                print('OVERPOWERED ANGULAR VELOCITY')

    def add_constraints(self):
        for idx, body in enumerate(self.bodies):
            if idx < len(self.bodies) - 1:
                pj = PivotJoint(self.space, body, self.bodies[idx + 1], body.world_to_local(self.PivotPoints[idx][0]),
                           self.bodies[idx+1].world_to_local(self.PivotPoints[idx][0]))
                if self.bodies[idx + 1].id != 'peg':
                    motor = pymunk.constraint.SimpleMotor(body, self.bodies[idx + 1], 0)
                    motor.max_force = 100000
                    self.space.add(motor)
                    self.motors.append(motor)
                else:
                    self.motors.append("NO MOTOR IN PEG")
                self.joints.append(pj.joint)
        # for idx, body in enumerate(self.bodies):
        #     if idx < len(self.bodies) - 1:
        #         pj = PivotJoint(self.space, body, self.bodies[idx + 1], body.world_to_local(self.PivotPoints[idx][0]),
        #                    self.bodies[idx+1].world_to_local(self.PivotPoints[idx][0]))
        #         self.joints.append(pj.joint)

    def denorm_action(self, action):
        """
        action is the angular velocity of each joint, normalised between [-1, 1]
        -- numpy array
        """
        p1 = (self.action_high - self.action_low)/2
        p2 = (self.action_high + self.action_low)/2

        return action*p1 + p2

    def norm_action(self, action):
        """
        action is the angular velocity of each joint, normalised between [-1, 1]
        -- numpy array
        """
        p1 = 2/(self.action_high - self.action_low)
        p2 = (self.action_high + self.action_low)/2
        return p1*(action-p2)

    def get_peg_tip(self):
        peg_tip = Vec2d(self.ORIGIN) if isinstance(self.ORIGIN, tuple) else self.ORIGIN
        lengths = self.Arms.lengths
        idx = 0
        for body in self.bodies:
            # if body is static, pass
            if body.body_type is 2:
                continue
            peg_tip += lengths[idx]*Vec2d(np.cos(body.angle), np.sin(body.angle))
            idx += 1
        return peg_tip

    def get_obs(self):
        """Get obs, currently showing tuples for each body:
            [0] Vec2d POSITION, relative to goal when using expert agents.
            [1] Scalar VELOCITY (mid-center of body arms).
            [2,3] Scalar cos, sin of ANGLE.
        """
        r_peg = self.get_peg_tip() - self.ORIGIN
        body = self.bodies[-1]
        true_angle = body.angle
        assert body.id is 'peg'

        # Meta RL obs
        r_peg_ = norm_pos(r_peg)
        # Expert case, use relative distance to the goal.
        if self.expert:
            # print("Expert Active")
            r_peg_ = norm_pos(self.tmp_GOAL - r_peg) 

        self.obs = [r_peg_.x, r_peg_.y, np.cos(true_angle), np.sin(true_angle)]
        
        # RL algorithm obs
        # self.obs = [self.GOAL.x - peg.x, self.GOAL.y - peg.y, np.cos(true_angle), np.sin(true_angle)]
        return np.array(self.obs)
        

class Frontend(pyglet.window.Window):
    # CHANGE TO VIZ
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_visibles()
        self.options = DrawOptions()

        self.spaces = []
        self.envs = []
        self.robos = []

        space = pymunk.Space()
        space.gravity = (0, 0) 
        space.damping = 0.95
        self.space = space
        env = StaticEnvironment(self.space, GOAL = GOAL, expert = True)
        self.env = env
        robo = RobotEnvironment(self.space, arm_lengths=[300, 250, 200], radii = [10, 7, 10], GOAL=GOAL, expert = True)
        self.robo = robo

        self.action_range = {'low':self.robo.action_low, 'high':self.robo.action_high}
        self.num_obss = 2 + 2 # FROM PEG obs
        self.num_actions = 2 + 1 # FROM PEG VELOX
        self._max_episode_steps = 0
        self._denorm_process = True
        self._print_counter = 0
        self.evaluate = True
        self.encountered = 0

    def get_all_task_idx(self):
        return range(len(self.spaces))

    @property
    def denorm_process(self):
        return self._denorm_process

    @denorm_process.setter
    def denorm_process(self, denorm):
        if isinstance(denorm, bool):
            self._denorm_process = denorm
        else:
            print("Please enter a boolean if you want to denorm or not the actions")

    @property
    def max_episode_steps(self):
        return self._max_episode_steps

    @max_episode_steps.setter
    def max_episode_steps(self, step):
        if step > 0 and isinstance(step, int):
            self._max_episode_steps = step
        else:
            print("Please enter a valid maximum step size")

    def random_action(self):
        rand = np.random.random(3)*2-1
        # if self.denorm_process:
        #     return self.robo.denorm_action(rand)
        return self.robo.denorm_action(rand)

    def set_visibles(self):
        if self.visible:
            self.set_location(50,50) # Location To Initialise window
            self.fps = FPSDisplay(self)
            self.show = True
            self.show_iter = 0

    def reset_class(self, *args, **kwargs):
        self.__init__(*args, **kwargs)

    def reset(self):
        self.robo.reset_bodies(random_reset = True)
        return self.robo.get_obs()

    def step_func(self, action, dt = 1/60, step = 0):
        """Action comes normalised in DDPG and TD3
        Not in SAC so if it is SAC we shouldn't denormalise"""

        dummy = 0
        done = False

        if self.denorm_process:
            print("Denormalising")
            action = self.robo.denorm_action(action)

        self.update(dt=dt, action=action)

        new_obs = self.robo.get_obs()
        reward, done = self.reward_func(done, new_obs)

        if (step+1) % self.max_episode_steps == 0:
            # print('Episode Run-Out')
            done = True

        return new_obs, reward, done, dummy

    def update(self, dt = DT, action = None):
        if action is not None:
            peg = self.robo.bodies[-1]
            
            for _ in range(20): # WAS 30
                # print(action, peg.velocity, peg.angular_velocity)
                peg.velocity = (1-self.mu_avg)*peg.velocity + self.mu_avg*Vec2d(action[0], action[1])
                peg.angular_velocity = (1-self.mu_avg)*peg.angular_velocity + self.mu_avg*action[2]
                self.space.step(dt/20)
        else:
            for _ in range(20):
                self.space.step(dt/20)

    def reward_func(self, obs):
        return self.env.reward_func(obs)

    def policy_update(self, dt):
        obs = np.array(self.robo.get_obs())
        if self._denorm_process:
            # DDPG
            action = self.agent.get_action(obs, evaluate = self.evaluate)
            action = self.robo.denorm_action(action)
        else:
            #SAC, TD3
            action = self.agent.get_action(obs, evaluate = self.evaluate)
        pos = self.robo.get_peg_tip()

        if pos.x >1100:
            print(pos, self.robo.bodies[-1].position)
            print("MADE IT, CONGRATS")

        self.update(action = action, dt = dt)

    def run_policy(self, agent):
        self.set_visible(visible = True)
        self.set_visibles()
        self.agent = agent
        pyglet.clock.schedule_interval(self.policy_update, 1/60)
        pyglet.app.run()

    # @_draw_decorator
    def on_draw(self):
        # if self.check:
        self.clear() # clear the buffer
        # Order matters here!
        self.space.debug_draw(self.options)
        self.fps.draw()

    def on_key_release(self, symbol, modifiers):
        for body in self.space.bodies:
            continue
            if body.id == 'peg':
                body.velocity *= 0.1*Vec2d(1,1)
                body.angular_velocity *= 0.1

    def on_key_press(self, symbol, modifiers):

        if symbol == pyglet.window.key.ESCAPE:
            pyglet.app.exit()
            self.close()
            return True

        bodies = self.robo.bodies

        for idx, body in enumerate(bodies):
            if idx is 0 and body.id is not 'root':
                raise Exception("Wrong body root")
            if body.id is f'Arm{idx + 1}' and 0 < idx < len(bodies):
                raise Exception("Wrong body listing arms")
            if body.id is not 'peg' and idx is len(bodies):
                raise Exception("Wrong body listing peg")

        if symbol == pyglet.window.key.C:
            action = np.random.sample(3)*2-1
            self.robo.action_buffered = self.robo.denorm_action(action)

        c = 100
        if symbol == pyglet.window.key.UP:
            bodies[-1].velocity += Vec2d(0,c)
        if symbol == pyglet.window.key.DOWN:
            bodies[-1].velocity -= Vec2d(0,c)
        if symbol == pyglet.window.key.LEFT:
            bodies[-1].velocity -= Vec2d(c,0)
        if symbol == pyglet.window.key.RIGHT:
            bodies[-1].velocity += Vec2d(c,0)

        if symbol == key.Z:
            self.env.draw_target()
        
        if symbol == key.X:
            self.env.remove_target()

        if symbol == pyglet.window.key.S:
            self.env.alter_task()

        if symbol == pyglet.window.key.R:
            self.reset()
            # self.robo.reset_bodies()

        if symbol == pyglet.window.key.P:
            pyglet.image.get_buffer_manager().get_color_buffer().save('RobotArm.png')

    def on_mouse_press(self, x, y, button, modifier):
        r, d = self.reward_func(self.robo.get_obs())

        print(r, d)

        point_q = self.space.point_query_nearest((x,y), 0, pymunk.ShapeFilter())
        if point_q:
            print(point_q.shape, point_q.shape.body)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.robo.bodies[-1].velocity = -0.1*(self.robo.bodies[-1].position - Vec2d(x,y))

def coll_begin(arbiter, space, data):
    # print(f'New Collision {arbiter.shapes}')
    pass
    return True

def coll_pre(arbiter, space, data):
    """First touch"""
    # RETURN FALSE IF BODIES IN CONTACT ARE JUST THE ARMS AND PEGS or ROOT AND ARMS
    return True

def coll_post(arbiter, space, data):
    # print('Postprocess collision')
    """Calls during contact"""
    pass

def coll_separate(arbiter, space, data):
    """Calls when objects are not touching"""
    # print('No more collision')
    pass

if __name__ == "__main__":

    frontend = Frontend(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False)
    frontend.show = True

    # TODO - Set motors instead.

    handler = frontend.space.add_default_collision_handler()
    handler.begin = coll_begin
    handler.pre_solve = coll_pre
    handler.post_solve = coll_post
    handler.separate = coll_separate

    pyglet.clock.schedule_interval(frontend.update, 1/60)
    pyglet.app.run()