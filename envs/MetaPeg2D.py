import pyglet
import pymunk
import pymunkoptions
from pymunk import constraint
from pymunk.pyglet_util import DrawOptions
from pyglet.window import key, FPSDisplay
from math import degrees
from pymunk import Vec2d
from numpy import cos, sin
from numpy.random import random_sample
from numpy import pi as np_pi
import numpy as np
import functools
import torch
pymunkoptions.options["debug"] = False

# Meta Variables
WINDOW_X = 1280
WINDOW_Y = 720
ORIGIN = (400, 360) # Robot gripper origin
PEG_DEPTH = 200

class StaticEnvironment:
    def __init__(self, space, GOAL, sprites = None):
        self.sprites = sprites
        self.space = space
        self.GOAL = GOAL
        self.make_static_env()
        

    def add_static_segment(self, position, endpoint_e, radius = 3.0, elasticity = 0.8, friction = 0.7):
        """
        *args
            endpoint_s -- Vec2D argument
            endpoint_e -- Vec2D argument
            elasticity [0, 1]
            friction [0, 1]
        ** kwargs
            radius
            Elasticity
            friction
        """
        # Make a static segment (floor)
        segment_shape = pymunk.Segment(self.space.static_body, (0, 0), endpoint_e, radius)
        #segment_shape.id = 1
        segment_shape.body.position = position
        segment_shape.elasticity = elasticity
        segment_shape.friction = friction
        segment_shape.set_neighbors(endpoint_e, position)

        self.space.add(segment_shape)

    def make_static_env(self):
        """
        seg_list is a list of tuples containing args for add_static_segment
        """

        # Static Coordinates for peg
        peg_pos_x = self.GOAL.x
        mid_frame = self.GOAL.y
        peg_girth = 40
        peg_depth = PEG_DEPTH
        thresh = 1000

        seg_list = [(Vec2d(peg_pos_x,0 - thresh),Vec2d(0,mid_frame - peg_girth//2 + thresh)),
                     (Vec2d(peg_pos_x,mid_frame - peg_girth//2),Vec2d(peg_depth,0)),
                     (Vec2d(peg_pos_x + peg_depth,mid_frame - peg_girth//2),Vec2d(0,peg_girth)),
                     (Vec2d(peg_pos_x + peg_depth,mid_frame + peg_girth//2),Vec2d(-peg_depth,0)),
                     (Vec2d(peg_pos_x,mid_frame + peg_girth//2),Vec2d(0,thresh + WINDOW_Y - mid_frame - peg_girth//2))]

        self.goal_pos = Vec2d(peg_pos_x, mid_frame)

        for segment in seg_list:
            self.add_static_segment(*segment)

class Arms:
    def __init__(self, arm_lengths, thickness = 4, radii = [5,5,10]):
        self.lengths = arm_lengths
        self.n = len(self.lengths)
        self.radii = radii
        self.thickness = thickness
        self.pos = [None]*self.n
        self.arm_vecs = [None]*self.n

class PivotJoint:
    def __init__(self, space, b, b2, a=(0, 0), a2=(0, 0), collide=True):
        joint = pymunk.constraint.PinJoint(b, b2, a, a2)
        joint.collide_bodies = collide
        self.joint = joint
        space.add(joint)

class Segment:
    def __init__(self, space, p0, v, radius=10):
        self.body = pymunk.Body()
        self.body.position = p0
        shape = pymunk.Segment(self.body, (0, 0), v, radius)
        shape.density = 0.1
        shape.elasticity = 0.5
        shape.filter = pymunk.ShapeFilter(group=1)
        shape.color = (0, 255, 0, 0)
        space.add(self.body, shape)

class RobotEnvironment:
    def __init__(self, space, arm_lengths = [250, 200, 200], radii = [5, 5, 10], GOAL = (1000, 400)):
        self.space = space
        self.GOAL = GOAL
        self.PivotPoints = [None]*(1+len(arm_lengths))
        self.PivotPoints[0] = (ORIGIN,0)
        self.Arms = Arms(arm_lengths, radii = radii)
        self.set_action_range(vals = [(2, 50), (1, np_pi/8)]) # 2 Velocities and 1 angular velocity
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
        self.action_high = high
        self.action_low = -high
        self.action_buffered = np.zeros((sum(nums),))

    def random_arm_generator(self):
        got = False
        iterator = 1
        while got is False:
            len_x = 0
            len_y = 0
            for idx in range(self.Arms.n):
                rand_angle = 0.5*np_pi*(random_sample() - 0.5)
                if idx is 0:
                    rand_angle *= 2
                length = self.Arms.lengths[idx]
                len_x += length*cos(rand_angle)
                len_y += length*sin(rand_angle)
                self.Arms.pos[idx] = (length, rand_angle)
            if len_x < self.GOAL.x - ORIGIN[0] and ( self.GOAL.x - ORIGIN[0] - len_x) < 150 and abs(len_y + ORIGIN[1] - self.GOAL.y) < 150:
                got = True
                if iterator > 500:
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
            self.point_a = ORIGIN
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
                mass = 10
                moment = 100
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
                # motor = pymunk.constraint.SimpleMotor(body, self.bodies[idx + 1], 0)
                # motor.max_force = 10000000
                # self.space.add(motor)
                self.joints.append(pj.joint)
                # self.motors.append(motor)

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
        peg_tip = Vec2d(ORIGIN) if isinstance(ORIGIN, tuple) else ORIGIN
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
            [0] Vec2d POSITION
            [1] Scalar VELOCITY (mid-center of body arms)
            [2] Scalar ANGLE (Of each body in radians)
        """

        peg = self.get_peg_tip()
        body = self.bodies[-1]
        true_angle = body.angle
        assert body.id is 'peg'
        # peg_tip = self.get_peg_tip()
        # oscillations = peg.angle//np.pi
        # remainder = peg.angle - oscillations*np.pi

        # Meta RL obs
        self.obs = [(peg.x - ORIGIN[0])/WINDOW_X, (peg.y - ORIGIN[1])/WINDOW_Y, np.cos(true_angle), np.sin(true_angle)]

        # RL algorithm obs
        # self.obs = [self.GOAL.x - peg.x, self.GOAL.y - peg.y, np.cos(true_angle), np.sin(true_angle)]
        # self.obs = [(peg.x - 1000), (peg.y - 360)]
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
        space.damping = 0.99
        self.space = space
        env = StaticEnvironment(self.space, GOAL = GOAL)
        self.env = env
        robo = RobotEnvironment(self.space, arm_lengths=[300, 250, 200], radii = [10, 7, 10], GOAL=GOAL)
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

    def update(self, dt = 1/60, action = None):

        if action is not None:
            peg = self.robo.bodies[-1]
            peg.velocity = action[0], action[1]
            peg.angular_velocity = action[2]

        for r in range(10):
            self.space.step(dt)

    def reward_func(self, done, obs):

        mask = done

        # obs = self.robo.get_obs()

        reward =  -Vec2d(obs[0], obs[1]).length/1000 #(obs[0] - WINDOW_X)/WINDOW_X #

        peg_tip = self.robo.get_peg_tip()

        if peg_tip.x > 975 and abs(peg_tip.y - self.env.GOAL.y) < 50 and abs(obs[-1]) < 0.20: #
            # print("ALmost Made it")
            reward += 0.1

        if peg_tip.x > self.env.GOAL.x and abs(peg_tip.y - self.env.GOAL.y) < 25 and abs(obs[-1]) < 0.10: #
            reward += 0.5

        if peg_tip.x > self.env.GOAL.x + 50:
            reward += 1

        if peg_tip.x > self.GOAL.x + 100:
            reward += 20
            self.encountered += 1
            mask = True
            print(f"\n MADE IT {self.encountered} TIMES TO GOAL \n")

        return reward, mask

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
        self.space.debug_draw(self.options)
        self.fps.draw()

    def on_key_release(self, symbol, modifiers):
        for body in self.space.bodies:
            body.velocity *= 0.25*Vec2d(1,1)
            body.angular_velocity *= 0.5

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

        v = 0.2
        if symbol == pyglet.window.key.Q:
            bodies[-1].angular_velocity += v
        if symbol == pyglet.window.key.A:
            bodies[-1].angular_velocity -= v
        if symbol == pyglet.window.key.W:
            bodies[-2].angular_velocity += v
        if symbol == pyglet.window.key.S:
            bodies[-2].angular_velocity -= v
        if symbol == pyglet.window.key.E:
            bodies[-1].angular_velocity += v
        if symbol == pyglet.window.key.D:
            bodies[-1].angular_velocity -= v

        c = 10
        if symbol == pyglet.window.key.UP:
            bodies[-1].velocity += Vec2d(0,c)
        if symbol == pyglet.window.key.DOWN:
            bodies[-1].velocity -= Vec2d(0,c)
        if symbol == pyglet.window.key.LEFT:
            bodies[-1].velocity -= Vec2d(c,0)
        if symbol == pyglet.window.key.RIGHT:
            bodies[-1].velocity += Vec2d(c,0)

        if symbol == pyglet.window.key.R:
            self.reset()
            # self.robo.reset_bodies()

        if symbol == pyglet.window.key.P:
            pyglet.image.get_buffer_manager().get_color_buffer().save('RobotArm.png')

    def on_mouse_press(self, x, y, button, modifier):
        r, d, angle, peg_tip = self.reward_func(False)

        print(r, d, angle, peg_tip)

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
