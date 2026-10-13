import importlib
from pathlib import Path
import gymnasium as gym

from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.environment import URMAv2ForCodesignEnv
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.general_properties import GeneralProperties
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.wrappers import RLXInfo, RecordEpisodeStatistics
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.multi_render_vectorized_wrapper import MultiRenderVectorEnv
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.multi_environment import MultiEnvironment


def create_env(config):
    train_robot_configs = []
    train_robot_short_names = []
    for robot in config.environment.train_robots:
        robot_config = importlib.import_module(f"shape_your_body.environments.robots.{robot}.robot_config").robot_config
        robot_config["directory_path"] = Path(__file__).parent.parent.parent.parent / "robots" / robot
        train_robot_configs.append(robot_config)
        train_robot_short_names.append(robot_config["short_name"])

    nr_envs_per_train_robot = config.environment.nr_envs // len(config.environment.train_robots)
    if nr_envs_per_train_robot * len(config.environment.train_robots) != config.environment.nr_envs:
        raise ValueError("Number of train environments must be divisible by the number of robots")

    if nr_envs_per_train_robot != 1:
        raise NotImplementedError("Only one environment per robot is supported at the moment")
    if config.runner.mode == "test" and not config.environment.multi_render:
        raise NotImplementedError("Multi-rendering is required in test mode")

    def make_env(robot_config, seed, nr_envs, max_observation_size, max_action_size, train):
        spaces_check = max_observation_size == -1 and max_action_size == -1

        def thunk():
            env = URMAv2ForCodesignEnv(
                robot_config=robot_config,
                runner_mode=config.runner.mode,
                seed=seed,
                render=config.environment.multi_render and not spaces_check,
                env_config=config.environment,
                nr_envs=nr_envs,
                multi_robot_max_observation_size=max_observation_size,
                multi_robot_max_action_size=max_action_size,
            )
            if train:
                env = RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    observation_sizes = []
    action_sizes = []
    for robot_config in train_robot_configs:
        env = make_env(robot_config, seed=0, nr_envs=1, max_observation_size=-1, max_action_size=-1, train=False)()
        observation_sizes.append(env.observation_space.shape[0])
        action_sizes.append(env.action_space.shape[0])
        env.close()
        del env
    max_observation_size = max(observation_sizes)
    max_action_size = max(action_sizes)

    train_envs = []
    for robot_config in train_robot_configs:
        for i in range(nr_envs_per_train_robot):
            train_envs.append(make_env(robot_config, seed=config.environment.seed + i, nr_envs=1, max_observation_size=max_observation_size, max_action_size=max_action_size, train=config.runner.mode == "train"))

    if config.environment.multi_render:
        train_env = MultiRenderVectorEnv(train_envs)
        train_env.activate_single_env_mode()
    else:
        train_env = gym.vector.SyncVectorEnv(train_envs)
    train_env = RLXInfo(train_env)
    train_env = MultiEnvironment(train_env, train_robot_short_names)

    train_env.general_properties = GeneralProperties

    class DummyEvalEnv:
        def close(self):
            pass
    eval_env = DummyEvalEnv()

    return train_env, eval_env
