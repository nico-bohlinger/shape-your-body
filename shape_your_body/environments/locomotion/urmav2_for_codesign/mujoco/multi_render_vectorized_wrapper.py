import os
from copy import deepcopy
import time
import numpy as np
import gymnasium as gym
import pygame

from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.viewer import MujocoViewer


class MultiRenderVectorEnv(gym.vector.SyncVectorEnv):
    def __init__(self, env_fns, file_name="multi_render.txt", observation_space=None, action_space=None, copy=True):
        super().__init__(env_fns, observation_space, action_space, copy)

        self.file_name = file_name
        self.nr_envs = len(env_fns)
        self.single_env_mode = False

        pygame.init()
        pygame.joystick.init()
        self.joystick_present = False
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            self.joystick_present = True
            self.last_joystick_change = time.time()

        self.env_to_render = 0
        self.env_to_render = self.get_env_to_render()
        model = self.envs[self.env_to_render].internal_state["mj_model"]
        dt = self.envs[self.env_to_render].dt
        self.viewer = MujocoViewer(model, dt)


    def get_env_to_render(self):
        if self.joystick_present:
            pygame.event.pump()
            current_time = time.time()
            if current_time - self.last_joystick_change < 0.3:
                return self.env_to_render
            if self.joystick.get_button(5):
                self.last_joystick_change = current_time
                return (self.env_to_render + 1) % self.nr_envs
            elif self.joystick.get_button(4):
                self.last_joystick_change = current_time
                return (self.env_to_render - 1) % self.nr_envs
            return self.env_to_render
        elif os.path.isfile(self.file_name):
            with open(self.file_name, "r") as f:
                env_to_render = f.read()
            return min(int(env_to_render), self.nr_envs-1)
        return 0


    def step_async(self, action):
        if self.single_env_mode:
            self.action = action[0]
        else:
            super().step_async(action)


    def reset_wait(self, **kwargs):
        if self.single_env_mode:
            observations, infos = super().reset_wait(**kwargs)
            return observations[self.env_to_render].reshape(1, -1), infos
        else:
            return super().reset_wait(**kwargs)


    def step_wait(self):
        if self.single_env_mode:
            env = self.envs[self.env_to_render]
            infos = {}
            observation, reward, terminated, truncated, info = env.step(self.action)
            if terminated or truncated:
                old_observation, old_info = observation, info
                observation, info = env.reset()
                info["final_observation"] = old_observation
                info["final_info"] = old_info
            infos = self._add_info(infos, info, 0)
            observation = np.array([observation])
            reward = np.array([reward])
            terminated = np.array([terminated])
            truncated = np.array([truncated])

            env_to_render = self.get_env_to_render()
            env = self.envs[env_to_render]

            if self.env_to_render != env_to_render:
                self.env_to_render = env_to_render
                observation = np.array([env.reset()[0]])
                env.internal_state["robot_randomized"] = True

            if env.internal_state["robot_randomized"]:
                self.viewer.load_new_model(env.internal_state["mj_model"])
                env.internal_state["robot_randomized"] = False

            env.render()

            self.viewer.render(env.internal_state["data"])

            return (
                deepcopy(observation) if self.copy else observation,
                np.copy(reward),
                np.copy(terminated),
                np.copy(truncated),
                infos,
            )
        else:
            return super().step_wait()


    def activate_single_env_mode(self):
        self.single_env_mode = True


    def close(self):
        self.viewer.close()
        pygame.quit()
        return super().close()
