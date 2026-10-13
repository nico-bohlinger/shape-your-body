import numpy as np


class NoneDRUnseenRobotFunction:
    def __init__(self, env):
        self.env = env


    def init(self):
        self.env.internal_state["position_offsets"] = np.zeros(self.env.nr_actuator_joints)


    def sample(self):
        return
