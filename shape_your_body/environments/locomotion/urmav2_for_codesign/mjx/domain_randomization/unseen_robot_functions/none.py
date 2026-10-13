import jax.numpy as jnp


class NoneDRUnseenRobotFunction:
    def __init__(self, env):
        self.env = env


    def init(self, internal_state):
        internal_state["position_offsets"] = jnp.zeros(self.env.nr_actuator_joints)


    def sample(self, internal_state, should_randomize, key):
        return
