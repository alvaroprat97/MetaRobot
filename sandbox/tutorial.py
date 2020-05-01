import pyglet
import pymunk
import pymunkoptions
from pymunk import constraint
from pymunk.pyglet_util import DrawOptions
from pyglet.window import key
from math import degrees
from pymunk import Vec2d
import numpy as np

pymunkoptions.options["debug"] = False

WINDOW_X = 1280
WINDOW_Y = 720
ORIGIN = (380.0, 300.0) # Robot gripper origin

# config = pyglet.gl.Config(sample_buffers=1, samples=2, double_buffer=True)
window = pyglet.window.Window(WINDOW_X, WINDOW_Y, "RoboPeg2D Simulation", vsync = False, resizable = False)

options = DrawOptions()

space = pymunk.Space()
space.gravity = (0, 0) # Gravity-less 2D simulation (2D robot operating on x,y plane)
space.damping = 0.99

def update(dt):
    for r in range(10):
        space.step(dt)

class Environment:
    def __init__(self, sprites = None):
        self.sprites = sprites

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
        segment_shape = pymunk.Segment(space.static_body, (0, 0), endpoint_e, radius)
        #segment_shape.id = 1
        segment_shape.body.position = position
        segment_shape.elasticity = elasticity
        segment_shape.friction = friction
        segment_shape.set_neighbors(endpoint_e, position)

        space.add(segment_shape)

    def make_static_env(self):
        """
        seg_list is a list of tuples containing args for add_static_segment
        """

        # Static Coordinates for peg
        peg_pos_x = 1000
        mid_frame = 360
        peg_girth = 40
        peg_depth = 200

        seg_list = [(Vec2d(peg_pos_x,0),Vec2d(0,mid_frame - peg_girth//2)),
                     (Vec2d(peg_pos_x,mid_frame - peg_girth//2),Vec2d(peg_depth,0)),
                     (Vec2d(peg_pos_x + peg_depth,mid_frame - peg_girth//2),Vec2d(0,peg_girth)),
                     (Vec2d(peg_pos_x + peg_depth,mid_frame + peg_girth//2),Vec2d(-peg_depth,0)),
                     (Vec2d(peg_pos_x,mid_frame + peg_girth//2),Vec2d(0,WINDOW_Y - mid_frame - peg_girth//2))]

        for segment in seg_list:
            self.add_static_segment(*segment)

    def make_dynamic_env(self):
        pass

class Arms:
    def __init__(self, arm_lengths, thickness = 4, radii = [5,5,10]):
        self.lengths = arm_lengths
        self.n = len(self.lengths)
        self.radii = radii
        self.thickness = thickness
        self.pos = []
        self.arm_vecs = []

class PivotJoint:
    def __init__(self, b, b2, a=(0, 0), a2=(0, 0), collide=True):
        joint = pymunk.constraint.PinJoint(b, b2, a, a2)
        joint.collide_bodies = collide
        space.add(joint)

class Segment:
    def __init__(self, p0, v, radius=10):
        self.body = pymunk.Body()
        self.body.position = p0
        shape = pymunk.Segment(self.body, (0, 0), v, radius)
        shape.density = 0.1
        shape.elasticity = 0.5
        shape.filter = pymunk.ShapeFilter(group=1)
        shape.color = (0, 255, 0, 0)
        space.add(self.body, shape)

class RobotEnvironment:
    def __init__(self, arm_lengths, radii = [5, 5, 10]):
        self.state = None
        self.Arms = Arms(arm_lengths, radii = radii)
        self.random_arm_generator()
        self.PivotPoints = [(ORIGIN,0)]

    def random_arm_generator(self):
        got = False
        iterator = 1
        while got is False:
            len_x = 0
            for idx in range(self.Arms.n):
                rand_angle = np.random.random_sample()*np.pi/2 - np.pi/4
                length = self.Arms.lengths[idx]
                len_x += length*np.cos(rand_angle)
                self.Arms.pos.append((length, rand_angle))
            if len_x < 1000 - ORIGIN[0]:
                print(f"Found at iteration {iterator}")
                got = True
            else:
                iterator += 1
                self.Arms.pos = []

    def get_vertices(self, arm_data):
        thc = self.Arms.thickness
        return [(-arm_data[0]//2,-thc),(arm_data[0]//2,-thc),(arm_data[0]//2,thc),(-arm_data[0]//2,thc)]

    def get_pos(self, arm_data, idx):
        self.arm_vector_curr = Vec2d((arm_data[0]*np.cos(arm_data[1]), arm_data[0]*np.sin(arm_data[1])))
        self.Arms.arm_vecs.append(self.arm_vector_curr)
        if idx is 0:
            self.point_a = ORIGIN
        else:
            self.point_a = self.point_b
        self.point_b = self.point_a + self.arm_vector_curr
        return self.point_a + self.arm_vector_curr//2

    def get_transform(self, arm_data):
        t = pymunk.Transform(a = np.cos(arm_data[1]), c = -np.sin(arm_data[1]), tx = 0,
                             b = np.sin(arm_data[1]), d = np.cos(arm_data[1]), ty = 0)
        return t

    def save_pivot(self, idx):
        self.PivotPoints.append((self.point_a + self.arm_vector_curr, idx))

    def reset_bodies(self):
        for body in space.bodies:
            if not hasattr(body, 'start_position'):
                continue
            body.position = Vec2d(body.start_position)
            body.force = 0, 0
            body.torque = 0
            body.velocity = 0, 0
            body.angular_velocity = 0
            body.angle = body.start_angle

    def init_robot(self):

        self.bodies = []

        b0= space.static_body
        b0.id = 'root'
#         space.add(b0)
        self.bodies.append(b0)
#         body = pymunk.Body(body_type = pymunk.Body.STATIC)
#         body.position = ORIGIN
#         shape = pymunk.Circle(body, radius = 0.2)
#         shape.filter = pymunk.ShapeFilter(group = 1)
#         space.add(shape, shape.body)
#         self.bodies.append(body)

        for idx, arm_data in enumerate(self.Arms.pos):
            vertices = self.get_vertices(arm_data)
            pos = self.get_pos(arm_data, idx)
            t = self.get_transform((0, 0))
            self.save_pivot(idx+1)

            body = pymunk.Body(10, pymunk.inf)
            body.position = pos
            body.start_position = pos
            body.angle = arm_data[1]
            body.start_angle = arm_data[1]

            body.id = 'peg' if idx+1 is len(self.Arms.pos) else f'arm{idx+1}'

            shape = pymunk.Poly(body, vertices, t,  radius = self.Arms.radii[idx])
            shape.filter = pymunk.ShapeFilter(group = 1)
            shape.elasticity = 0.5
            shape.color = (0, 255, 0, 0)
            shape.friction = 0.9
            shape.density = 1

            space.add(body, shape)
            self.bodies.append(body)
#             body.activate()

        self.add_constraints()

    def add_constraints(self):

        for idx, body in enumerate(self.bodies):
            if idx < len(self.bodies) - 1:
                print(f"Pivoing at {self.PivotPoints[idx][0]}")
                print(body, self.bodies[idx + 1])
                PivotJoint(body, self.bodies[idx + 1], body.world_to_local(self.PivotPoints[idx][0]),
                           self.bodies[idx+1].world_to_local(self.PivotPoints[idx][0]))

    def get_state(self):
        return [angle[1] for angle in self.Arms.pos]

def coll_begin(arbiter, space, data):
    print('New Collision')
    pass
    return True

def coll_pre(arbiter, space, data):
    print('Preprocess collision')
    """First touch"""
    return True

def coll_post(arbiter, space, data):
    print('Postprocess collision')
    """Calls during contact"""
    pass

def coll_separate(arbiter, space, data):
    """Calls when objects are not touching"""
    print('No more collision')
    pass

# handler = space.add_default_collision_handler()
# handler.begin = coll_begin
# handler.pre_solve = coll_pre
# handler.post_solve = coll_post
# handler.separate = coll_separate

@window.event
def on_key_release(symbol, modifiers):
    space.bodies[-1].velocity = Vec2d(0,0)

@window.event
def on_key_press(symbol, modifiers):
    bodies = robo.bodies
    for idx, body in enumerate(bodies):
        if idx is 0 and body.id is not 'root':
            raise Exception("Wrong body root")
        if body.id is f'Arm{idx + 1}' and 0 < idx < len(bodies):
            raise Exception("Wrong body listing arms")
        if body.id is not 'peg' and idx is len(bodies):
            raise Exception("Wrong body listing peg")
    v = 0.1
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

    c = 5
    if symbol == pyglet.window.key.UP:
        bodies[-1].velocity += Vec2d(0,c)
    if symbol == pyglet.window.key.DOWN:
        bodies[-1].velocity -= Vec2d(0,c)
    if symbol == pyglet.window.key.LEFT:
        bodies[-1].velocity -= Vec2d(c,0)
    if symbol == pyglet.window.key.RIGHT:
        bodies[-1].velocity += Vec2d(c,0)

    if symbol == pyglet.window.key.R:
        robo.reset_bodies()

@window.event
def on_draw():
    window.clear() # clear the buffer
    # Order matters here!
    space.debug_draw(options)

@window.event
def on_mouse_press(x, y, button, modifier):
    point_q = space.point_query_nearest((x,y), 0, pymunk.ShapeFilter())
    if point_q:
        print(point_q.shape, point_q.shape.body)

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    robo.bodies[-1].velocity = -0.1*(robo.bodies[-1].position - Vec2d(x,y))

if __name__ == "__main__":
    env = Environment()
    robo = RobotEnvironment([250, 320.0, 230.0], radii = [5, 5, 10])
    env.make_static_env()
    robo.init_robot()
    pyglet.clock.schedule_interval(update, 1.0/60)
    pyglet.app.run()
