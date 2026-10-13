import numpy as np
import gymnasium as gym


class MultiEnvironment(gym.Wrapper):
    def __init__(self, env, train_robot_short_names):
        super(MultiEnvironment, self).__init__(env)

        self.train_robot_short_names = train_robot_short_names

        self.nr_train_robots = len(train_robot_short_names)
        self.nr_envs_per_train_robot = self.env.call("nr_envs")[0]
        self.joint_description_size = self.env.call("joint_description_size")[0]
        self.foot_description_size = self.env.call("foot_description_size")[0]
        self.joint_observation_size = self.env.call("joint_observation_size")[0]
        self.foot_observation_size = self.env.call("foot_observation_size")[0]
        self.horizon = self.env.call("horizon")[0]
        self.nr_actuator_joints = self.env.call("nr_actuator_joints")[0]
        self.nr_feet = self.env.call("nr_feet")[0]
        self.policy_general_obs_idx = self.env.call("policy_general_obs_idx")[0]
        self.critic_general_obs_idx = self.env.call("critic_general_obs_idx")[0]
        self.per_robot_factor_dims = tuple(int(factor_dim) for factor_dim in self.env.call("factor_dim"))
        self.max_factor_dim = max(self.per_robot_factor_dims)


    def factor_dim(self):
        return self.max_factor_dim


    def set_factor_params(self, factors):
        for i, env in enumerate(self.envs):
            env.set_factor_params(factors[i, :self.per_robot_factor_dims[i]])


    def get_curriculum_coefficients(self):
        return np.array([env.internal_state["env_curriculum_coeff"] for env in self.envs])
