import jax
import jax.numpy as jnp


class DefaultDRUnseenRobotFunction:
    def __init__(self, env):
        self.env = env

        self.position_offset = env.env_config["domain_randomization"]["unseen_robot"]["position_offset"]


    def init(self, internal_state):
        internal_state["position_offsets"] = jnp.zeros(self.env.nr_actuator_joints)


    def sample(self, internal_state, should_randomize, key):
        keys = jax.random.split(key, 10)
        internal_state["position_offsets"] = jnp.where(should_randomize, internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[9], minval=-self.position_offset, maxval=self.position_offset, shape=(self.env.nr_actuator_joints,)), internal_state["position_offsets"])
