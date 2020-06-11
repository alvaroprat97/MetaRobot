import numpy as np
import sys
sys.path.append("")
from backend.policies.base import SerializablePolicy


class RandomPolicy(SerializablePolicy):
    """
    Policy that always outputs zero.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def get_action(self, obs):
        return self.action_space.sample(), {}
