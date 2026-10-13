import importlib
from pathlib import Path

from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.environment import URMAv2ForCodesignEnv
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.multi_environment import MultiEnvironment
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.general_properties import GeneralProperties


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

    def get_env(robot_config, nr_envs, max_observation_size, max_action_size):
        return URMAv2ForCodesignEnv(
            robot_config=robot_config,
            env_config=config.environment,
            nr_envs=nr_envs,
            multi_robot_max_observation_size=max_observation_size,
            multi_robot_max_action_size=max_action_size,
        )

    observation_sizes = []
    action_sizes = []
    for robot_config in train_robot_configs:
        env = get_env(robot_config, nr_envs=1, max_observation_size=-1, max_action_size=-1)
        observation_sizes.append(env.single_observation_space.shape[0])
        action_sizes.append(env.single_action_space.shape[0])
        env.close()
        del env
    max_observation_size = max(observation_sizes)
    max_action_size = max(action_sizes)

    train_envs = []
    for robot_config in train_robot_configs:
        env = get_env(robot_config, nr_envs=nr_envs_per_train_robot, max_observation_size=max_observation_size, max_action_size=max_action_size)
        train_envs.append(env)

    train_env = MultiEnvironment(train_envs, train_robot_short_names)
    train_env.train_robot_short_names = train_robot_short_names
    train_env.general_properties = GeneralProperties

    class DummyEvalEnv:
        def close(self):
            pass
    eval_env = DummyEvalEnv()

    return train_env, eval_env
