from copy import deepcopy
from pathlib import Path
import gymnasium as gym
import mujoco
from dm_control import mjcf
import pygame
import numpy as np
from scipy.spatial.transform import Rotation

from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.control_functions.handler import get_control_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.command_functions.handler import get_command_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.gait_manager_functions.handler import get_gait_manager_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.initial_state_functions.handler import get_initial_state_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.sampling_functions.handler import get_sampling_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.reward_functions.handler import get_reward_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.termination_functions.handler import get_termination_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.action_delay_functions.handler import get_domain_randomization_action_delay_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.unseen_robot_functions.handler import get_domain_randomization_unseen_robot_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.seen_robot_functions.handler import get_domain_randomization_seen_robot_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.observation_noise_functions.handler import get_observation_noise_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.exteroceptive_observation_functions.handler import get_exteroceptive_observation_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.terrain_functions.handler import get_terrain_function


class URMAv2ForCodesignEnv(gym.Env):
    def __init__(self, robot_config, runner_mode, seed, render, env_config, nr_envs, multi_robot_max_observation_size, multi_robot_max_action_size):

        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.add_goal_arrow = env_config["add_goal_arrow"]
        self.nr_envs = nr_envs

        self.np_rng = np.random.default_rng(seed)

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)

        xml_handle.option.iterations = 100
        xml_handle.option.ls_iterations = 50
        xml_handle.option.flag.eulerdamp = "enable"

        if self.should_render and self.add_goal_arrow:
            trunk = xml_handle.find("body", "trunk")
            trunk.add("site", name="dir_arrow_ball", type="sphere", size=".02", pos="-.1 0 .15", group="0")
            trunk.add("site", name="dir_arrow", type="cylinder", size=".01", fromto="0 0 .05 0 0 .25", group="0")

        self.initial_mj_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        self.initial_mj_model.opt.timestep = env_config["timestep"]
        self.data = mujoco.MjData(self.initial_mj_model)
        self.c_model = deepcopy(self.initial_mj_model)
        self.c_data = mujoco.MjData(self.c_model)
        self.c_data.qpos = self.initial_mj_model.keyframe("home").qpos
        mujoco.mj_forward(self.c_model, self.c_data)

        self.imu_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.trunk_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.actuator_joint_max_velocities = np.array(robot_config["actuator_joint_max_velocities"])
        self.initial_qpos = np.array(self.initial_mj_model.keyframe("home").qpos)
        self.initial_imu_orientation_rotation_inverse = Rotation.from_matrix(self.c_data.site_xmat[self.imu_site_id].reshape(3, 3)).inv()
        self.initial_imu_height = self.c_data.site_xpos[self.imu_site_id, 2]
        self.actuator_joint_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]) for actuator_trnid in self.initial_mj_model.actuator_trnid]
        self.actuator_joint_mask_joints = np.array([self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qpos = np.array([self.initial_mj_model.joint(joint_name).qposadr[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qvel = np.array([self.initial_mj_model.joint(joint_name).dofadr[0] for joint_name in self.actuator_joint_names])
        self.nr_actuator_joints = len(self.actuator_joint_names)
        self.nr_joints = self.initial_mj_model.njnt

        imu_angular_velocity_sensor_id = self.initial_mj_model.sensor("imu_angular_velocity").id
        self.imu_angular_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_angular_velocity_sensor_id]
        self.imu_angular_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_angular_velocity_sensor_id]
        imu_linear_velocity_sensor_id = self.initial_mj_model.sensor("imu_linear_velocity").id
        self.imu_linear_velocity_sensor_adr = self.initial_mj_model.sensor_adr[imu_linear_velocity_sensor_id]
        self.imu_linear_velocity_sensor_dim = self.initial_mj_model.sensor_dim[imu_linear_velocity_sensor_id]

        geom_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) for geom_id in range(self.initial_mj_model.ngeom)]
        self.feet_names = [geom_name for geom_name in geom_names if geom_name and "foot" in geom_name]
        self.foot_geom_indices = np.array([mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.feet_names])
        self.nr_feet = len(self.feet_names)

        feet_xpos = self.c_data.geom_xpos[self.foot_geom_indices]
        x_pos, y_pos, z_pos = feet_xpos[:, 0], feet_xpos[:, 1], feet_xpos[:, 2]
        abs_y_feet_xpos = np.array([x_pos, np.abs(y_pos), z_pos]).T
        distances_between_abs_y_feet = np.linalg.norm(abs_y_feet_xpos[:, None] - abs_y_feet_xpos[None], axis=-1)
        min_dist_indices = np.argmin(distances_between_abs_y_feet + np.eye(len(abs_y_feet_xpos)) * 1000, axis=1)
        feet_symmetry_set = set([(min(i, min_dist_indices[i]), max(i, min_dist_indices[i])) for i in range(len(min_dist_indices)) if min_dist_indices[min_dist_indices[i]] == i])
        self.feet_symmetry_pairs = np.array([list(pair) for pair in feet_symmetry_set])
        self.body_ids_of_feet = np.array([self.initial_mj_model.geom(geom_id).bodyid[0] for geom_id in self.foot_geom_indices])
        all_feet_are_sphere = np.all(self.initial_mj_model.geom_type[self.foot_geom_indices] == 2)
        all_feet_are_box = np.all(self.initial_mj_model.geom_type[self.foot_geom_indices] == 6)
        if not all_feet_are_sphere | all_feet_are_box:
            raise ValueError("Foot geoms are not all of type sphere or box.")
        self.foot_type = "sphere" if all_feet_are_sphere else "box"
        self.foot_type_int = 0 if self.foot_type == "sphere" else 1

        feet_global_linear_velocity_sensor_ids = [self.initial_mj_model.sensor(f"{foot_name}_global_linear_velocity").id for foot_name in self.feet_names]
        self.feet_global_linear_velocity_sensor_adrs_start = np.array([self.initial_mj_model.sensor_adr[sensor_id] for sensor_id in feet_global_linear_velocity_sensor_ids])

        self.body_to_parentid = np.array([self.initial_mj_model.body(body_id).parentid[0] for body_id in range(self.initial_mj_model.nbody)])
        body_to_children_count = np.array([np.sum(self.body_to_parentid == body_id) for body_id in range(self.initial_mj_model.nbody)])
        self.body_ids_of_actuator_joints = np.array([self.initial_mj_model.joint(joint_name).bodyid[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_nr_direct_child_actuator_joints = body_to_children_count[self.body_ids_of_actuator_joints]

        self.floor_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self.reward_collision_sphere_geom_ids = np.array([geom.id for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)] if geom.group[0] == 5])

        self.reward_collision_sphere_geoms_and_feet_geoms_ids = np.concatenate((self.reward_collision_sphere_geom_ids, self.foot_geom_indices))
        self.dim_geom_ids = self.reward_collision_sphere_geoms_and_feet_geoms_ids - 1

        self.has_equality_constraints = len(self.initial_mj_model.eq_data) > 0

        self.env_curriculum_nr_levels = env_config["env_curriculum_nr_levels"]
        self.env_curriculum_level_success_episode_return = env_config["env_curriculum_level_success_episode_return"]

        self.control_function = get_control_function(env_config["control_type"], self)
        self.control_frequency_hz = self.control_function.control_frequency_hz
        self.nr_substeps = int(round(1 / self.control_frequency_hz / env_config["timestep"]))
        self.dt = env_config["timestep"] * self.nr_substeps
        self.horizon = int(round(env_config["episode_length_in_seconds"] * self.control_frequency_hz))
        self.command_function = get_command_function(env_config["command"]["type"], self)
        self.command_sampling_function = get_sampling_function(env_config["command"]["sampling_type"], self)
        self.initial_state_function = get_initial_state_function(env_config["domain_randomization"]["initial_state"]["type"], self)
        self.gait_manager_function = get_gait_manager_function(env_config["gait_manager"]["type"] if "gait_manager" in dir(env_config) else "default", self)
        self.reward_function = get_reward_function(env_config["reward"]["type"], self)
        self.termination_function = get_termination_function(env_config["termination"]["type"], self)
        self.policy_exteroceptive_observation_function = get_exteroceptive_observation_function(env_config["policy_exteroceptive_observation_type"], self)
        self.critic_exteroceptive_observation_function = get_exteroceptive_observation_function(env_config["critic_exteroceptive_observation_type"], self)
        self.terrain_function = get_terrain_function(env_config["terrain"]["type"], self)
        self.domain_randomization_sampling_function = get_sampling_function(env_config["domain_randomization"]["sampling_type"], self)
        self.domain_randomization_action_delay_function = get_domain_randomization_action_delay_function(env_config["domain_randomization"]["action_delay"]["type"], self)
        self.domain_randomization_mujoco_model_function = get_domain_randomization_mujoco_model_function(env_config["domain_randomization"]["mujoco_model"]["type"], self)
        self.domain_randomization_unseen_robot_function = get_domain_randomization_unseen_robot_function(env_config["domain_randomization"]["unseen_robot"]["type"], self)
        self.domain_randomization_seen_robot_function = get_domain_randomization_seen_robot_function(env_config["domain_randomization"]["seen_robot"]["type"], self)
        self.seen_robot_active = env_config["domain_randomization"]["seen_robot"]["type"] != "none"
        self.domain_randomization_perturbation_function = get_domain_randomization_perturbation_function(env_config["domain_randomization"]["perturbation"]["type"], self)
        self.domain_randomization_perturbation_sampling_function = get_sampling_function(env_config["domain_randomization"]["perturbation"]["sampling_type"], self)
        self.observation_noise_function = get_observation_noise_function(env_config["domain_randomization"]["observation_noise"]["type"], self)

        self.co_design_joint_description_indices = env_config["embodiment_info"]["co_design_joint_description_indices"]
        self.co_design_foot_description_indices = env_config["embodiment_info"]["co_design_foot_description_indices"]
        self.co_design_general_obs_indices = env_config["embodiment_info"]["co_design_general_obs_indices"]

        action_space_size = self.nr_actuator_joints
        self.missing_nr_of_actions = 0
        if multi_robot_max_action_size != -1:
            self.missing_nr_of_actions = multi_robot_max_action_size - action_space_size
            action_space_size = multi_robot_max_action_size
        action_space_low = -np.ones(action_space_size) * np.inf
        action_space_high = np.ones(action_space_size) * np.inf
        self.action_space = gym.spaces.Box(low=action_space_low, high=action_space_high, shape=(action_space_size,), dtype=np.float32)

        self.observation_space = self.get_observation_space(multi_robot_max_observation_size)

        self.observation_noise_function.init_attributes()

        eval_mode = False
        self.internal_state = {
            "mj_model": deepcopy(self.initial_mj_model),
            "data": mujoco.MjData(self.initial_mj_model),
            "in_eval_mode": eval_mode,
            "env_curriculum_coeff": np.where(eval_mode, 1.0, 0.0),
            "env_curriculum_levels_in_a_row": 0.0,
            "actuator_joint_nominal_positions": self.initial_qpos[self.actuator_joint_mask_qpos],
            "actuator_joint_max_velocities": self.actuator_joint_max_velocities,
            "goal_velocities": np.array([0.0, 0.0, 0.0]),
            "imu_orientation_rotation": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
            "imu_orientation_rotation_inverse": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]).inv(),
            "imu_orientation_euler": np.array([0.0, 0.0, 0.0]),
            "last_action": np.zeros(self.nr_actuator_joints),
            "second_last_action": np.zeros(self.nr_actuator_joints),
            "seen_gains_and_action_scaling_factor_normalized": np.zeros(3),
            "unseen_gains_and_action_scaling_factor_normalized": np.zeros(3),
            "seen_mass_normalized": np.zeros(1),
            "unseen_mass_normalized": np.zeros(1),
            "robot_dimensions_normalized": np.zeros(3),
            "max_command_velocity": 0.0,
            "nr_collisions_in_nominal": 0,
            "scaling_factor": self.robot_config["scaling_factor"],
            "robot_nominal_qpos_height_over_ground": self.initial_qpos[2],
            "robot_nominal_imu_height_over_ground": self.initial_imu_height,
            "robot_randomized": False,
            "robot_set_by_factor": False,
            "info": {
                f"rollout/episode_return/{self.robot_config['short_name']}": 0.0,
                f"rollout/episode_length/{self.robot_config['short_name']}": 0,
                f"env_curriculum/coefficient/{self.robot_config['short_name']}": np.where(eval_mode, 1.0, 0.0),
            },
            "info_episode_store": {
                "episode_return": 0.0,
                "episode_step": 0,
                "episode_total_xy_velocity_diff_abs": 0.0,
            },
        }
        self.gait_manager_function.init()
        self.command_function.init()
        self.reward_function.init()
        self.terrain_function.init()
        self.domain_randomization_action_delay_function.init()
        self.domain_randomization_unseen_robot_function.init()
        self.domain_randomization_seen_robot_function.init()
        policy_joint_descriptions, critic_joint_descriptions, critic_feet_descriptions = self.get_description_vectors()
        self.internal_state["policy_joint_descriptions"] = policy_joint_descriptions
        self.internal_state["critic_joint_descriptions"] = critic_joint_descriptions
        self.internal_state["critic_feet_descriptions"] = critic_feet_descriptions
        self.reward_function.reward_and_info(np.zeros(self.nr_actuator_joints))

        if self.should_render:
            self.dir_arrow_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "dir_arrow")

            pygame.init()
            pygame.joystick.init()
            self.joystick_present = False
            if pygame.joystick.get_count() > 0:
                self.joystick = pygame.joystick.Joystick(0)
                self.joystick.init()
                self.joystick_present = True
        del self.c_model, self.c_data


    def render(self):
        if self.runner_mode == "test":
            explicit_velocity_commands = False
            if self.joystick_present:
                pygame.event.pump()
                goal_x_velocity = -self.joystick.get_axis(1)
                goal_y_velocity = -self.joystick.get_axis(0)
                goal_yaw_velocity = -self.joystick.get_axis(3)
                explicit_velocity_commands = True
            elif Path("commands/trunk_velocity.txt").is_file():
                with open("commands/trunk_velocity.txt", "r") as f:
                    commands = f.readlines()
                if len(commands) == 3:
                    goal_x_velocity = float(commands[0])
                    goal_y_velocity = float(commands[1])
                    goal_yaw_velocity = float(commands[2])
                    explicit_velocity_commands = True
            if explicit_velocity_commands:
                goal_velocities = np.array([goal_x_velocity, goal_y_velocity, goal_yaw_velocity])
                goal_velocities = np.where(np.abs(goal_velocities) < (self.command_function.zero_clip_threshold_percentage * self.internal_state["max_command_velocity"]), 0.0, goal_velocities)
                self.internal_state["goal_velocities"] = np.clip(goal_velocities, -self.internal_state["max_command_velocity"], self.internal_state["max_command_velocity"])
                actuator_keep_nominal_commands = np.where(np.all(goal_velocities == 0.0), np.ones(self.nr_actuator_joints, dtype=bool), self.command_function.default_actuator_joint_keep_nominal)
                self.internal_state["actuator_joint_keep_nominal"] = actuator_keep_nominal_commands
            explicit_actuator_keep_nominal_commands = False
            if Path(f"commands/{self.robot_config['short_name']}.txt").is_file():
                with open(f"commands/{self.robot_config['short_name']}.txt", "r") as f:
                    commands = f.readlines()
                if len(commands) == self.nr_actuator_joints:
                    actuator_keep_nominal_commands = [bool(int(command.replace("\n",""))) for command in commands]
                    actuator_keep_nominal_commands = np.array(actuator_keep_nominal_commands)
                    explicit_actuator_keep_nominal_commands = True
            if explicit_actuator_keep_nominal_commands:
                self.internal_state["actuator_joint_keep_nominal"] = actuator_keep_nominal_commands

        if self.add_goal_arrow:
            goal_velocities = self.internal_state["goal_velocities"]
            trunk_rotation = self.internal_state["imu_orientation_euler"][2]
            desired_angle = trunk_rotation + np.arctan2(goal_velocities[1], goal_velocities[0])
            rot_mat = Rotation.from_euler('xyz', (np.array([np.pi/2, 0, np.pi/2 + desired_angle]))).as_matrix()
            self.internal_state["data"].site("dir_arrow").xmat = rot_mat.reshape((9,))
            magnitude = np.sqrt(np.sum(np.square([goal_velocities[0], goal_velocities[1]])))
            self.internal_state["mj_model"].site_size[self.dir_arrow_id, 1] = magnitude * 0.1
            arrow_offset = -(0.1 - (magnitude * 0.1))
            arrow_center = self.internal_state["data"].site("dir_arrow").xpos.copy()
            self.internal_state["data"].site("dir_arrow").xpos = arrow_center + [arrow_offset * np.sin(np.pi/2 + desired_angle), -arrow_offset * np.cos(np.pi/2 + desired_angle), 0]
            self.internal_state["data"].site("dir_arrow_ball").xpos = arrow_center + [-0.1 * np.sin(np.pi/2 + desired_angle), 0.1 * np.cos(np.pi/2 + desired_angle), 0]


    def reset(self, seed=None):
        self.terrain_function.sample()

        qpos, qvel = self.initial_state_function.setup()
        self.internal_state["data"] = mujoco.MjData(self.internal_state["mj_model"])
        self.internal_state["data"].qpos = qpos
        self.internal_state["data"].qvel = qvel
        self.internal_state["data"].ctrl = self.internal_state["actuator_joint_nominal_positions"].copy()
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])

        episode_success = self.internal_state["info_episode_store"]["episode_return"] >= self.env_curriculum_level_success_episode_return
        self.internal_state["env_curriculum_levels_in_a_row"] = np.where(episode_success,
            np.where(self.internal_state["env_curriculum_levels_in_a_row"] >= 0,
                self.internal_state["env_curriculum_levels_in_a_row"] + 1,
                1
            ),
            np.where(self.internal_state["env_curriculum_levels_in_a_row"] < 0,
                self.internal_state["env_curriculum_levels_in_a_row"] - 1,
                -1
            )
        )
        self.internal_state["env_curriculum_coeff"] = np.clip(self.internal_state["env_curriculum_coeff"] + self.internal_state["env_curriculum_levels_in_a_row"] / self.env_curriculum_nr_levels, 0.0, 1.0)
        self.internal_state["env_curriculum_coeff"] = np.where(self.internal_state["in_eval_mode"], 1.0, self.internal_state["env_curriculum_coeff"])

        self.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(self.internal_state["data"].site_xmat[self.imu_site_id].reshape(3, 3))
        self.internal_state["imu_orientation_rotation_inverse"] = self.internal_state["imu_orientation_rotation"].inv()
        self.internal_state["imu_orientation_euler"] = self.internal_state["imu_orientation_rotation"].as_euler("xyz")
        self.internal_state["last_action"] = np.zeros(self.nr_actuator_joints)
        self.internal_state["second_last_action"] = np.zeros(self.nr_actuator_joints)
        self.gait_manager_function.setup()
        self.reward_function.setup()
        self.domain_randomization_action_delay_function.setup()
        self.command_function.get_next_command()
        self.handle_domain_randomization(is_episode_start=True)

        next_observation = self.get_observation(np.zeros(self.nr_actuator_joints))
        self.internal_state["info_episode_store"] = {
            "episode_return": 0.0,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
        }

        return next_observation, self.internal_state["info"]


    def step(self, action):
        chosen_action = action[:self.nr_actuator_joints]
        delayed_action = self.domain_randomization_action_delay_function.delay_action(chosen_action)

        target_joint_positions = self.control_function.process_action(delayed_action)
        vmax_delta = self.internal_state["actuator_joint_max_velocities"] * self.dt
        prev_target = self.internal_state["data"].ctrl
        target_joint_positions = np.clip(target_joint_positions, prev_target - vmax_delta, prev_target + vmax_delta)

        self.internal_state["data"].ctrl = target_joint_positions
        mujoco.mj_step(self.internal_state["mj_model"], self.internal_state["data"], self.nr_substeps)
        max_qvel = 100 * np.ones(self.initial_mj_model.nv)
        self.internal_state["data"].qvel = np.clip(self.internal_state["data"].qvel, -max_qvel, max_qvel)

        self.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(self.internal_state["data"].site_xmat[self.imu_site_id].reshape(3, 3))
        self.internal_state["imu_orientation_rotation_inverse"] = self.internal_state["imu_orientation_rotation"].inv()
        self.internal_state["imu_orientation_euler"] = self.internal_state["imu_orientation_rotation"].as_euler("xyz")

        self.handle_domain_randomization(is_episode_start=False)

        self.terrain_function.pre_step()

        reward = self.reward_function.reward_and_info(chosen_action)

        should_sample_commands = self.command_sampling_function.step()
        if should_sample_commands:
            self.command_function.get_next_command()

        next_observation = self.get_observation(chosen_action)
        terminated = self.termination_function.should_terminate() | np.any(np.abs(self.internal_state["data"].qvel[:3]) == 100.0)
        truncated = self.internal_state["info_episode_store"]["episode_step"] >= (self.horizon - 1)
        done = terminated | truncated

        self.terrain_function.post_step()
        self.reward_function.step()
        self.gait_manager_function.step()

        self.internal_state["second_last_action"] = self.internal_state["last_action"].copy()
        self.internal_state["last_action"] = chosen_action.copy()
        self.internal_state["info_episode_store"]["episode_step"] += 1
        self.internal_state["info_episode_store"]["episode_return"] += reward
        self.internal_state["info_episode_store"]["episode_total_xy_velocity_diff_abs"] += self.internal_state["info"][f"env_info/xy_vel_diff_abs/{self.robot_config['short_name']}"]
        self.internal_state["info"][f"rollout/episode_return/{self.robot_config['short_name']}"] = np.where(done, self.internal_state["info_episode_store"]["episode_return"], self.internal_state["info"][f"rollout/episode_return/{self.robot_config['short_name']}"])
        self.internal_state["info"][f"rollout/episode_length/{self.robot_config['short_name']}"] = np.where(done, self.internal_state["info_episode_store"]["episode_step"], self.internal_state["info"][f"rollout/episode_length/{self.robot_config['short_name']}"])
        self.internal_state["info"][f"env_curriculum/coefficient/{self.robot_config['short_name']}"] = self.internal_state["env_curriculum_coeff"]

        return next_observation, reward, terminated, truncated, self.internal_state["info"]


    def get_observation(self, action):
        # Dynamic observations
        policy_joint_observations = np.concatenate([
            self.internal_state["policy_joint_descriptions"],
            (self.internal_state["data"].qpos[self.actuator_joint_mask_qpos] - self.internal_state["actuator_joint_nominal_positions"]).reshape(-1, 1),
            self.internal_state["data"].qvel[self.actuator_joint_mask_qvel].reshape(-1, 1),
            action.reshape(-1, 1),
            self.internal_state["actuator_joint_keep_nominal"].reshape(-1, 1),
        ], axis=-1).reshape(-1)

        critic_joint_observations = np.concatenate([
            self.internal_state["critic_joint_descriptions"],
            (self.internal_state["data"].qpos[self.actuator_joint_mask_qpos] - self.internal_state["actuator_joint_nominal_positions"]).reshape(-1, 1),
            self.internal_state["data"].qvel[self.actuator_joint_mask_qvel].reshape(-1, 1),
            action.reshape(-1, 1),
            self.internal_state["actuator_joint_keep_nominal"].reshape(-1, 1),
        ], axis=-1).reshape(-1)

        critic_foot_observations = np.concatenate([
            self.internal_state["critic_feet_descriptions"],
            self.terrain_function.check_feet_floor_contact().reshape(self.nr_feet, 1),
            self.internal_state["feet_time_on_ground"].reshape(self.nr_feet, 1),
            self.internal_state["feet_time_in_air"].reshape(self.nr_feet, 1)
        ], axis=-1).reshape(-1)

        # General observations
        critic_imu_linear_velocity = self.internal_state["data"].sensordata[self.imu_linear_velocity_sensor_adr:self.imu_linear_velocity_sensor_adr + self.imu_linear_velocity_sensor_dim]
        policy_imu_angular_velocity = self.internal_state["data"].sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim]
        critic_imu_angular_velocity = self.internal_state["data"].sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim]
        policy_goal_velocities = self.internal_state["goal_velocities"]
        critic_goal_velocities = self.internal_state["goal_velocities"]
        policy_phase_features = self.gait_manager_function.get_phase_features()
        critic_phase_features = policy_phase_features
        policy_projected_gravity_vector = self.internal_state["imu_orientation_rotation_inverse"].apply(np.array([0.0, 0.0, -1.0]))
        critic_projected_gravity_vector = self.internal_state["imu_orientation_rotation_inverse"].apply(np.array([0.0, 0.0, -1.0]))
        policy_exteroception = np.array([self.policy_exteroceptive_observation_function.get_exteroceptive_observation()]).reshape(-1)
        critic_exteroception = np.array([self.critic_exteroceptive_observation_function.get_exteroceptive_observation()]).reshape(-1)
        critic_curriculum_coeff = np.array([self.internal_state["env_curriculum_coeff"]])

        # General robot context
        policy_gains_and_action_scaling_factor = self.internal_state["seen_gains_and_action_scaling_factor_normalized"]
        critic_gains_and_action_scaling_factor = self.internal_state["unseen_gains_and_action_scaling_factor_normalized"]
        policy_mass = self.internal_state["seen_mass_normalized"]
        critic_mass = self.internal_state["unseen_mass_normalized"]
        policy_robot_dimensions = self.internal_state["robot_dimensions_normalized"]
        critic_robot_dimensions = self.internal_state["robot_dimensions_normalized"]
        policy_trunk_attributes = self.internal_state["seen_trunk_attributes_normalized"]
        critic_trunk_attributes = self.internal_state["unseen_trunk_attributes_normalized"]

        # Padding
        padding = np.zeros(self.missing_nr_of_observations)

        observation = np.concatenate([
            policy_joint_observations,
            critic_joint_observations,
            critic_foot_observations,
            critic_imu_linear_velocity,
            policy_imu_angular_velocity,
            critic_imu_angular_velocity,
            policy_goal_velocities,
            critic_goal_velocities,
            policy_phase_features,
            critic_phase_features,
            policy_projected_gravity_vector,
            critic_projected_gravity_vector,
            policy_exteroception,
            critic_exteroception,
            critic_curriculum_coeff,
            policy_gains_and_action_scaling_factor,
            critic_gains_and_action_scaling_factor,
            policy_mass,
            critic_mass,
            policy_robot_dimensions,
            critic_robot_dimensions,
            policy_trunk_attributes,
            critic_trunk_attributes,
            padding
        ], axis=-1)

        # Add noise
        self.observation_noise_function.modify_observation(observation)

        # Normalize and clip
        observation[self.policy_joint_positions_obs_idx] /= 6.5
        observation[self.critic_joint_positions_obs_idx] /= 6.5
        observation[self.policy_joint_velocities_obs_idx] /= 180.0
        observation[self.critic_joint_velocities_obs_idx] /= 180.0
        observation[self.policy_joint_previous_actions_obs_idx] /= 10.0
        observation[self.critic_joint_previous_actions_obs_idx] /= 10.0
        observation[self.critic_feet_ground_contact_obs_idx] = (observation[self.critic_feet_ground_contact_obs_idx] / 0.5) - 1.0
        observation[self.critic_feet_time_on_ground_obs_idx] = np.clip((observation[self.critic_feet_time_on_ground_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0)
        observation[self.critic_feet_time_in_air_obs_idx] = np.clip((observation[self.critic_feet_time_in_air_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0)
        observation[self.critic_imu_linear_vel_obs_idx] = np.clip(observation[self.critic_imu_linear_vel_obs_idx] / 10.0, -1.0, 1.0)
        observation[self.policy_imu_angular_vel_obs_idx] = np.clip(observation[self.policy_imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0)
        observation[self.critic_imu_angular_vel_obs_idx] = np.clip(observation[self.critic_imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0)
        if len(self.policy_exteroception_obs_idx) > 0:
            observation[self.policy_exteroception_obs_idx] = np.clip((observation[self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0)
        if len(self.critic_exteroception_obs_idx) > 0:
            observation[self.critic_exteroception_obs_idx] = np.clip((observation[self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0)
        observation[self.critic_curriculum_coeff_obs_idx] = np.clip((observation[self.critic_curriculum_coeff_obs_idx] / (1.0 / 2)) - 1.0, -1.0, 1.0)

        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = np.clip(observation, -10.0, 10.0)

        return observation


    def handle_domain_randomization(self, is_episode_start=False):
        should_randomize_domain_episode_start = self.domain_randomization_sampling_function.setup()
        should_randomize_domain_perturbation_episode_start = self.domain_randomization_perturbation_sampling_function.setup()
        should_randomize_domain_step = self.domain_randomization_sampling_function.step()
        should_randomize_domain_perturbation_step = self.domain_randomization_perturbation_sampling_function.step()
        should_randomize_domain = np.where(is_episode_start, should_randomize_domain_episode_start | self.internal_state["in_eval_mode"], should_randomize_domain_step)
        should_randomize_domain_perturbation = np.where(is_episode_start, should_randomize_domain_perturbation_episode_start, should_randomize_domain_perturbation_step)

        if should_randomize_domain:
            self.domain_randomization_unseen_robot_function.sample()
            self.domain_randomization_seen_robot_function.sample()
            self.domain_randomization_mujoco_model_function.sample()
            self.domain_randomization_action_delay_function.sample()

        if should_randomize_domain_perturbation:
            self.domain_randomization_perturbation_function.sample()


    def get_observation_space(self, multi_robot_max_observation_size):
        self.joint_description_size = 30
        self.foot_description_size = 21
        self.joint_observation_size = 4
        self.foot_observation_size = 3

        current_observation_idx = 0

        self.policy_joint_descriptions_obs_idx = []
        self.policy_joint_positions_obs_idx = []
        self.policy_joint_velocities_obs_idx = []
        self.policy_joint_previous_actions_obs_idx = []
        self.policy_joint_keep_nominal_obs_idx = []
        for _ in range(self.nr_actuator_joints):
            self.policy_joint_descriptions_obs_idx.extend([current_observation_idx + i for i in range(self.joint_description_size)])
            current_observation_idx += self.joint_description_size
            self.policy_joint_positions_obs_idx.append(current_observation_idx)
            current_observation_idx += 1
            self.policy_joint_velocities_obs_idx.append(current_observation_idx)
            current_observation_idx += 1
            self.policy_joint_previous_actions_obs_idx.append(current_observation_idx)
            current_observation_idx += 1
            self.policy_joint_keep_nominal_obs_idx.append(current_observation_idx)
            current_observation_idx += 1
        self.policy_joint_descriptions_obs_idx = np.array(self.policy_joint_descriptions_obs_idx)
        self.policy_joint_positions_obs_idx = np.array(self.policy_joint_positions_obs_idx)
        self.policy_joint_velocities_obs_idx = np.array(self.policy_joint_velocities_obs_idx)
        self.policy_joint_previous_actions_obs_idx = np.array(self.policy_joint_previous_actions_obs_idx)
        self.policy_joint_keep_nominal_obs_idx = np.array(self.policy_joint_keep_nominal_obs_idx)
        self.policy_joint_observations_obs_idx = np.concatenate([
            self.policy_joint_positions_obs_idx,
            self.policy_joint_velocities_obs_idx,
            self.policy_joint_previous_actions_obs_idx,
            self.policy_joint_keep_nominal_obs_idx
        ])

        self.critic_joint_descriptions_obs_idx = []
        self.critic_joint_positions_obs_idx = []
        self.critic_joint_velocities_obs_idx = []
        self.critic_joint_previous_actions_obs_idx = []
        self.critic_joint_keep_nominal_obs_idx = []
        for _ in range(self.nr_actuator_joints):
            self.critic_joint_descriptions_obs_idx.extend([current_observation_idx + i for i in range(self.joint_description_size)])
            current_observation_idx += self.joint_description_size
            self.critic_joint_positions_obs_idx.append(current_observation_idx)
            current_observation_idx += 1
            self.critic_joint_velocities_obs_idx.append(current_observation_idx)
            current_observation_idx += 1
            self.critic_joint_previous_actions_obs_idx.append(current_observation_idx)
            current_observation_idx += 1
            self.critic_joint_keep_nominal_obs_idx.append(current_observation_idx)
            current_observation_idx += 1
        self.critic_joint_descriptions_obs_idx = np.array(self.critic_joint_descriptions_obs_idx)
        self.critic_joint_positions_obs_idx = np.array(self.critic_joint_positions_obs_idx)
        self.critic_joint_velocities_obs_idx = np.array(self.critic_joint_velocities_obs_idx)
        self.critic_joint_previous_actions_obs_idx = np.array(self.critic_joint_previous_actions_obs_idx)
        self.critic_joint_keep_nominal_obs_idx = np.array(self.critic_joint_keep_nominal_obs_idx)
        self.critic_joint_observations_obs_idx = np.concatenate([
            self.critic_joint_positions_obs_idx,
            self.critic_joint_velocities_obs_idx,
            self.critic_joint_previous_actions_obs_idx,
            self.critic_joint_keep_nominal_obs_idx
        ], dtype=int)

        self.critic_feet_descriptions_obs_idx = []
        self.critic_feet_ground_contact_obs_idx = []
        self.critic_feet_time_on_ground_obs_idx = []
        self.critic_feet_time_in_air_obs_idx = []
        for _ in range(self.nr_feet):
            self.critic_feet_descriptions_obs_idx.extend([current_observation_idx + i for i in range(self.foot_description_size)])
            current_observation_idx += self.foot_description_size
            self.critic_feet_ground_contact_obs_idx.append(current_observation_idx)
            current_observation_idx += 1
            self.critic_feet_time_on_ground_obs_idx.append(current_observation_idx)
            current_observation_idx += 1
            self.critic_feet_time_in_air_obs_idx.append(current_observation_idx)
            current_observation_idx += 1
        self.critic_feet_descriptions_obs_idx = np.array(self.critic_feet_descriptions_obs_idx)
        self.critic_feet_ground_contact_obs_idx = np.array(self.critic_feet_ground_contact_obs_idx)
        self.critic_feet_time_on_ground_obs_idx = np.array(self.critic_feet_time_on_ground_obs_idx)
        self.critic_feet_time_in_air_obs_idx = np.array(self.critic_feet_time_in_air_obs_idx)
        self.critic_feet_observations_obs_idx = np.concatenate([
            self.critic_feet_ground_contact_obs_idx,
            self.critic_feet_time_on_ground_obs_idx,
            self.critic_feet_time_in_air_obs_idx
        ], dtype=int)

        self.critic_imu_linear_vel_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3

        self.policy_imu_angular_vel_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3

        self.critic_imu_angular_vel_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3

        self.policy_goal_velocity_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3

        self.critic_goal_velocity_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3

        self.policy_gait_phase_obs_idx = np.array([current_observation_idx + i for i in range(4)], dtype=int)
        current_observation_idx += 4

        self.critic_gait_phase_obs_idx = np.array([current_observation_idx + i for i in range(4)], dtype=int)
        current_observation_idx += 4

        self.policy_gravity_vector_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3

        self.critic_gravity_vector_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3

        self.policy_exteroception_obs_idx = np.array([current_observation_idx + i for i in range(self.policy_exteroceptive_observation_function.nr_exteroceptive_observations)], dtype=int)
        current_observation_idx += self.policy_exteroceptive_observation_function.nr_exteroceptive_observations

        self.critic_exteroception_obs_idx = np.array([current_observation_idx + i for i in range(self.critic_exteroceptive_observation_function.nr_exteroceptive_observations)], dtype=int)
        current_observation_idx += self.critic_exteroceptive_observation_function.nr_exteroceptive_observations

        self.critic_curriculum_coeff_obs_idx = np.array([current_observation_idx], dtype=int)
        current_observation_idx += 1

        self.policy_gains_and_action_scaling_factor_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3

        self.critic_gains_and_action_scaling_factor_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3

        self.policy_mass_obs_idx = np.array([current_observation_idx], dtype=int)
        current_observation_idx += 1

        self.critic_mass_obs_idx = np.array([current_observation_idx], dtype=int)
        current_observation_idx += 1

        self.policy_robot_dimensions_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3

        self.critic_robot_dimensions_obs_idx = np.array([current_observation_idx + i for i in range(3)], dtype=int)
        current_observation_idx += 3

        self.policy_trunk_attributes_obs_idx = np.array([current_observation_idx + i for i in range(7)], dtype=int)
        current_observation_idx += 7

        self.critic_trunk_attributes_obs_idx = np.array([current_observation_idx + i for i in range(7)], dtype=int)
        current_observation_idx += 7

        self.policy_general_obs_idx = np.concatenate([
            self.policy_imu_angular_vel_obs_idx,
            self.policy_goal_velocity_obs_idx,
            self.policy_gait_phase_obs_idx,
            self.policy_gravity_vector_obs_idx,
            self.policy_exteroception_obs_idx,
            self.policy_gains_and_action_scaling_factor_obs_idx,
            self.policy_mass_obs_idx,
            self.policy_robot_dimensions_obs_idx,
            self.policy_trunk_attributes_obs_idx
        ], dtype=int)

        self.critic_general_obs_idx = np.concatenate([
            self.critic_imu_linear_vel_obs_idx,
            self.critic_imu_angular_vel_obs_idx,
            self.critic_goal_velocity_obs_idx,
            self.critic_gait_phase_obs_idx,
            self.critic_gravity_vector_obs_idx,
            self.critic_curriculum_coeff_obs_idx,
            self.critic_gains_and_action_scaling_factor_obs_idx,
            self.critic_mass_obs_idx,
            self.critic_robot_dimensions_obs_idx,
            self.critic_trunk_attributes_obs_idx,
            self.critic_exteroception_obs_idx,
        ], dtype=int)

        self.missing_nr_of_observations = 0
        if multi_robot_max_observation_size != -1:
            self.missing_nr_of_observations = multi_robot_max_observation_size - current_observation_idx
            current_observation_idx += self.missing_nr_of_observations

        observation_space_low = -np.ones(current_observation_idx) * np.inf
        observation_space_high = np.ones(current_observation_idx) * np.inf

        return gym.spaces.Box(low=observation_space_low, high=observation_space_high, shape=(current_observation_idx,), dtype=np.float32)


    def get_description_vectors(self):
        qpos = self.initial_qpos.copy()
        qpos[self.actuator_joint_mask_qpos] = self.internal_state["actuator_joint_nominal_positions"]
        qpos[2] = self.internal_state["robot_nominal_qpos_height_over_ground"] + self.internal_state["center_height"]
        data = mujoco.MjData(self.internal_state["mj_model"])
        data.qpos = qpos
        mujoco.mj_forward(self.internal_state["mj_model"], data)

        imu_pos = data.site_xpos[self.imu_site_id]
        imu_orientation_rotation_inverse = self.initial_imu_orientation_rotation_inverse

        # Calculate robot width, length and height
        relative_geom_positions = data.geom_xpos[1:] - imu_pos  # Ignore the first geom (floor)
        relative_geom_positions = imu_orientation_rotation_inverse.apply(relative_geom_positions)

        relative_joint_positions = data.xanchor[self.initial_mj_model.actuator_trnid[:,0]] - imu_pos
        relative_joint_positions = imu_orientation_rotation_inverse.apply(relative_joint_positions)

        relative_joint_axis = data.xaxis[self.initial_mj_model.actuator_trnid[:,0]]
        relative_joint_axis = imu_orientation_rotation_inverse.apply(relative_joint_axis)

        relative_com_positions = data.xipos - imu_pos
        relative_com_positions = imu_orientation_rotation_inverse.apply(relative_com_positions)

        ximat = data.ximat.reshape(-1, 3, 3)

        geom_xmats = data.geom_xmat[1:].reshape(-1, 3, 3)
        global_frame_geom_sizes = np.matmul(geom_xmats, self.internal_state["mj_model"].geom_size[1:].reshape(-1, 3, 1)).reshape(-1, 3)
        trunk_frame_geom_sizes = imu_orientation_rotation_inverse.apply(global_frame_geom_sizes)
        abs_rotated_geom_sizes = np.abs(trunk_frame_geom_sizes)

        global_frame_body_inertias = np.matmul(ximat, self.internal_state["mj_model"].body_inertia.reshape(-1, 3, 1)).reshape(-1, 3)
        trunk_frame_body_inertias = imu_orientation_rotation_inverse.apply(global_frame_body_inertias)

        relative_geom_positions_minus_geom_size = relative_geom_positions[self.dim_geom_ids] - abs_rotated_geom_sizes[self.dim_geom_ids]
        relative_geom_positions_plus_geom_size = relative_geom_positions[self.dim_geom_ids] + abs_rotated_geom_sizes[self.dim_geom_ids]

        min_x = np.min(np.concatenate([relative_geom_positions_minus_geom_size[:, 0], relative_geom_positions_plus_geom_size[:, 0], relative_joint_positions[:, 0], relative_com_positions[:, 0]]))
        min_y = np.min(np.concatenate([relative_geom_positions_minus_geom_size[:, 1], relative_geom_positions_plus_geom_size[:, 1], relative_joint_positions[:, 1], relative_com_positions[:, 1]]))
        min_z = np.min(np.concatenate([relative_geom_positions_minus_geom_size[:, 2], relative_geom_positions_plus_geom_size[:, 2], relative_joint_positions[:, 2], relative_com_positions[:, 2]]))
        max_x = np.max(np.concatenate([relative_geom_positions_minus_geom_size[:, 0], relative_geom_positions_plus_geom_size[:, 0], relative_joint_positions[:, 0], relative_com_positions[:, 0]]))
        max_y = np.max(np.concatenate([relative_geom_positions_minus_geom_size[:, 1], relative_geom_positions_plus_geom_size[:, 1], relative_joint_positions[:, 1], relative_com_positions[:, 1]]))
        max_z = np.max(np.concatenate([relative_geom_positions_minus_geom_size[:, 2], relative_geom_positions_plus_geom_size[:, 2], relative_joint_positions[:, 2], relative_com_positions[:, 2]]))
        mins = np.array([min_x, min_y, min_z])
        robot_length = max_x - min_x
        robot_width = max_y - min_y
        robot_height = max_z - min_z
        robot_dimensions = np.array([robot_length, robot_width, robot_height])
        robot_dimensions_mean = np.mean(robot_dimensions)

        relative_joint_pos_normalized = (relative_joint_positions - mins) / robot_dimensions
        relative_feet_pos_normalized = (relative_geom_positions[self.foot_geom_indices - 1] - mins) / robot_dimensions
        relative_com_pos_normalized = (relative_com_positions - mins) / robot_dimensions

        gains_and_action_scaling_factor = np.array([
            self.internal_state["mj_model"].actuator_gainprm[0, 0],
            -self.internal_state["mj_model"].actuator_biasprm[0, 2],
            self.internal_state["scaling_factor"],
        ])
        mass = np.array([np.sum(self.internal_state["mj_model"].body_mass[1:])])

        gains_and_action_scaling_factor_normalized = (gains_and_action_scaling_factor / np.array([170.0 / 2, 2.0 / 2, 1.5 / 2])) - 1.0
        mass_normalized = (mass / (900.0 / 2)) - 1.0
        self.internal_state["seen_gains_and_action_scaling_factor_normalized"] = gains_and_action_scaling_factor_normalized
        self.internal_state["unseen_gains_and_action_scaling_factor_normalized"] = gains_and_action_scaling_factor_normalized
        self.internal_state["seen_mass_normalized"] = mass_normalized
        self.internal_state["unseen_mass_normalized"] = mass_normalized
        self.internal_state["robot_dimensions_mean"] = robot_dimensions_mean
        self.internal_state["robot_dimensions_normalized"] = (robot_dimensions / (3.0 / 2)) - 1.0
        self.internal_state["max_command_velocity"] = np.minimum(robot_dimensions_mean * self.command_function.max_velocity_per_m_factor, self.command_function.clip_max_velocity)
        trunk_attributes_normalized = np.concatenate([
            (self.internal_state["mj_model"].body_mass[self.trunk_body_id].reshape(-1) / (400.0 / 2)) - 1.0,
            (relative_com_pos_normalized[self.trunk_body_id].reshape(-1) / 0.5) - 1.0,
            (trunk_frame_body_inertias[self.trunk_body_id].reshape(-1) / (26.0 / 2)) - 1.0,
        ], axis=-1)
        self.internal_state["seen_trunk_attributes_normalized"] = trunk_attributes_normalized
        self.internal_state["unseen_trunk_attributes_normalized"] = trunk_attributes_normalized

        critic_feet_description_vectors = np.concatenate([
            (relative_feet_pos_normalized / 0.5) - 1.0,
            (self.internal_state["mj_model"].body_mass[self.body_ids_of_feet].reshape(-1, 1) / (15.0 / 2)) - 1.0,
            (relative_com_pos_normalized[self.body_ids_of_feet] / 0.5) - 1.0,
            (trunk_frame_body_inertias[self.body_ids_of_feet] / (0.65 / 2)) - 1.0,
            (trunk_frame_geom_sizes[self.foot_geom_indices - 1] / (0.2 / 2)) - 1.0,
            np.tile((self.foot_type_int / (1.0 / 2)) - 1.0, (self.nr_feet, 1)),
            np.tile(self.internal_state["unseen_gains_and_action_scaling_factor_normalized"], (self.nr_feet, 1)),
            np.tile(self.internal_state["unseen_mass_normalized"], (self.nr_feet, 1)),
            np.tile(self.internal_state["robot_dimensions_normalized"], (self.nr_feet, 1))
        ], axis=-1)

        joint_description_vectors = np.concatenate([
            (relative_joint_pos_normalized / 0.5) - 1.0,
            relative_joint_axis,
            (self.internal_state["mj_model"].body_mass[self.body_ids_of_actuator_joints].reshape(-1, 1) / (400.0 / 2)) - 1.0,
            (relative_com_pos_normalized[self.body_ids_of_actuator_joints] / 0.5) - 1.0,
            (trunk_frame_body_inertias[self.body_ids_of_actuator_joints] / (26.0 / 2)) - 1.0,
            (self.actuator_joint_nr_direct_child_actuator_joints.reshape(-1, 1) / (5.0 / 2)) - 1.0,
            self.internal_state["actuator_joint_nominal_positions"].reshape(-1, 1) / 6.5,
            (self.internal_state["mj_model"].actuator_forcerange[:, 1].reshape(-1, 1) / (1600.0 / 2)) - 1.0,
            (self.internal_state["actuator_joint_max_velocities"].reshape(-1, 1) / (180.0 / 2)) - 1.0,
            (self.internal_state["mj_model"].dof_damping[self.actuator_joint_mask_qvel].reshape(-1, 1) / (0.2 / 2)) - 1.0,
            (self.internal_state["mj_model"].dof_armature[self.actuator_joint_mask_qvel].reshape(-1, 1) / (2.7 / 2)) - 1.0,
            (self.internal_state["mj_model"].jnt_stiffness[self.actuator_joint_mask_joints].reshape(-1, 1) / (0.2 / 2)) - 1.0,
            (self.internal_state["mj_model"].dof_frictionloss[self.actuator_joint_mask_qvel].reshape(-1, 1) / (37.0 / 2)) - 1.0,
            self.internal_state["mj_model"].jnt_range[self.actuator_joint_mask_joints] / 6.5,
            np.tile(gains_and_action_scaling_factor_normalized, (self.nr_actuator_joints, 1)),
            np.tile(mass_normalized, (self.nr_actuator_joints, 1)),
            np.tile(self.internal_state["robot_dimensions_normalized"], (self.nr_actuator_joints, 1)),
        ], axis=-1)

        policy_joint_description_vectors = joint_description_vectors
        critic_joint_description_vectors = joint_description_vectors

        return policy_joint_description_vectors, critic_joint_description_vectors, critic_feet_description_vectors


    def factor_dim(self):
        if not self.seen_robot_active:
            return 0
        return self.domain_randomization_seen_robot_function.total_factor_dim()


    def set_factor_params(self, factor):
        if not self.seen_robot_active:
            return
        self.domain_randomization_seen_robot_function.sample(fixed_factor=np.asarray(factor, dtype=np.float64))
        policy_joint_descriptions, critic_joint_descriptions, critic_feet_descriptions = self.get_description_vectors()
        self.internal_state["policy_joint_descriptions"] = policy_joint_descriptions
        self.internal_state["critic_joint_descriptions"] = critic_joint_descriptions
        self.internal_state["critic_feet_descriptions"] = critic_feet_descriptions
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])
        self.internal_state["robot_randomized"] = True
        self.internal_state["robot_set_by_factor"] = True


    def close(self):
        return
