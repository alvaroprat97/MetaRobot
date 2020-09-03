import pyglet
import pymunk
import pymunkoptions
from pymunk import constraint
from pymunk.pyglet_util import DrawOptions
from pymunk import Vec2d

# Meta Variables
WINDOW_X = 1280
WINDOW_Y = 720
ORIGIN = (200, 360) # Robot gripper origin
REACH_ORIGIN = tuple(Vec2d(WINDOW_X, WINDOW_Y)/2)
VERTICAL_ORIGIN = (WINDOW_X/2, WINDOW_Y + 360)
PEG_DEPTH = 200
PEG_GIRTH = 40
STICK_GIRTH = 28
DT = 1/5
GOAL = Vec2d(1000, 300)
TASK_TYPES = ["Reach2D", "Stick2D", "Peg2D", "Stick2Dv", "Peg2Dv", "Key2D", "Key2Dv"]

def norm_pos(pos):
    assert isinstance(pos, Vec2d)
    return Vec2d((pos.x-640)/100, (pos.y/100))

def denorm_pos(pos):
    assert isinstance(pos, Vec2d)
    return Vec2d(pos.x*100 +640, pos.y*100)

def denorm_point(point, is_x = False, is_y = False):
    assert is_x is not is_y
    if is_x:
        return point*100 + 640
    if is_y:
        return point*100 

def norm_point(point, is_x = False, is_y = False):
    assert is_x is not is_y
    if is_x:
        return (point - 640)/100
    if is_y:
        return point/100

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