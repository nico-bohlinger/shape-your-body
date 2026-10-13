import numpy as np


class DefaultDRUnseenRobotFunction:
    def __init__(self, env):
        self.env = env

        self.position_offset = env.env_config["domain_randomization"]["unseen_robot"]["position_offset"]


    def init(self):
        self.env.internal_state["position_offsets"] = np.zeros(self.env.nr_actuator_joints)


    def sample(self):
        self.env.internal_state["position_offsets"] = self.env.internal_state["env_curriculum_coeff"] * self.env.np_rng.uniform(low=-self.position_offset, high=self.position_offset, size=(self.env.nr_actuator_joints,))
