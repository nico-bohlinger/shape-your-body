from copy import deepcopy
from functools import partial
import mujoco
from mujoco import mjx
from dm_control import mjcf
import numpy as np
from jax.scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp

from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.state import State
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.box_space import BoxSpace
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.control_functions.handler import get_control_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.command_functions.handler import get_command_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.gait_manager_functions.handler import get_gait_manager_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.domain_randomization.initial_state_functions.handler import get_initial_state_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.sampling_functions.handler import get_sampling_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.reward_functions.handler import get_reward_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.termination_functions.handler import get_termination_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.domain_randomization.action_delay_functions.handler import get_domain_randomization_action_delay_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.domain_randomization.mujoco_model_functions.handler import get_domain_randomization_mujoco_model_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.domain_randomization.unseen_robot_functions.handler import get_domain_randomization_unseen_robot_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.domain_randomization.seen_robot_functions.handler import get_domain_randomization_seen_robot_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.domain_randomization.perturbation_functions.handler import get_domain_randomization_perturbation_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.domain_randomization.observation_noise_functions.handler import get_observation_noise_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.exteroceptive_observation_functions.handler import get_exteroceptive_observation_function
from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.terrain_functions.handler import get_terrain_function


class URMAv2ForCodesignEnv:
    def __init__(self, robot_config, env_config, nr_envs, multi_robot_max_observation_size, multi_robot_max_action_size):

        self.robot_config = robot_config
        self.env_config = env_config
        self.nr_envs = nr_envs

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)

        for texture in xml_handle.asset.find_all("texture"):
            texture.remove()
        for material in xml_handle.asset.find_all("material"):
            material.remove()
        for mesh in xml_handle.asset.find_all("mesh"):
            mesh.remove()

        for geom in xml_handle.find_all("geom"):
            is_foot_geom = geom.name and "foot" in geom.name
            is_floor_geom = geom.name == "floor"
            is_reward_collision_sphere_geom = geom.dclass and geom.dclass.dclass == "reward_collision_sphere"
            if not is_foot_geom and not is_floor_geom and not is_reward_collision_sphere_geom:
                geom.remove()
            if is_floor_geom:
                geom.material = ""

        self.initial_mj_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        self.initial_mj_model.opt.timestep = env_config["timestep"]
        self.data = mujoco.MjData(self.initial_mj_model)
        self.initial_mjx_model = mjx.put_model(self.initial_mj_model)
        self.mjx_data = mjx.make_data(self.initial_mjx_model)
        self.mjx_data = mjx.forward(self.initial_mjx_model, self.mjx_data)  # Necessary because of an error with some robots
        self.c_model = deepcopy(self.initial_mj_model)
        self.c_data = mujoco.MjData(self.c_model)
        self.c_data.qpos = self.initial_mj_model.keyframe("home").qpos
        mujoco.mj_forward(self.c_model, self.c_data)

        self.imu_site_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_SITE, "imu")
        self.trunk_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.actuator_joint_max_velocities = jnp.array(robot_config["actuator_joint_max_velocities"])
        self.initial_qpos = jnp.array(self.initial_mj_model.keyframe("home").qpos)
        self.initial_imu_orientation_rotation_inverse = Rotation.from_matrix(self.c_data.site_xmat[self.imu_site_id].reshape(3, 3)).inv()
        self.initial_imu_height = self.c_data.site_xpos[self.imu_site_id, 2]
        self.actuator_joint_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, actuator_trnid[0]) for actuator_trnid in self.initial_mj_model.actuator_trnid]
        self.actuator_joint_mask_joints = jnp.array([self.initial_mj_model.joint(joint_name).id for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qpos = jnp.array([self.initial_mj_model.joint(joint_name).qposadr[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_mask_qvel = jnp.array([self.initial_mj_model.joint(joint_name).dofadr[0] for joint_name in self.actuator_joint_names])
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
        self.foot_geom_indices = jnp.array([mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, foot_name) for foot_name in self.feet_names])
        self.nr_feet = len(self.feet_names)

        feet_xpos = self.c_data.geom_xpos[self.foot_geom_indices]
        x_pos, y_pos, z_pos = feet_xpos[:, 0], feet_xpos[:, 1], feet_xpos[:, 2]
        abs_y_feet_xpos = np.array([x_pos, jnp.abs(y_pos), z_pos]).T
        distances_between_abs_y_feet = np.linalg.norm(abs_y_feet_xpos[:, None] - abs_y_feet_xpos[None], axis=-1)
        min_dist_indices = np.argmin(distances_between_abs_y_feet + np.eye(len(abs_y_feet_xpos)) * 1000, axis=1)
        feet_symmetry_set = set([(min(i, min_dist_indices[i]), max(i, min_dist_indices[i])) for i in range(len(min_dist_indices)) if min_dist_indices[min_dist_indices[i]] == i])
        self.feet_symmetry_pairs = jnp.array([list(pair) for pair in feet_symmetry_set])
        self.body_ids_of_feet = jnp.array([self.initial_mj_model.geom(geom_id).bodyid[0] for geom_id in self.foot_geom_indices])
        all_feet_are_sphere = jnp.all(self.initial_mjx_model.geom_type[self.foot_geom_indices] == 2)
        all_feet_are_box = jnp.all(self.initial_mjx_model.geom_type[self.foot_geom_indices] == 6)
        if not all_feet_are_sphere | all_feet_are_box:
            raise ValueError("Foot geoms are not all of type sphere or box.")
        self.foot_type = "sphere" if all_feet_are_sphere else "box"
        self.foot_type_int = 0 if self.foot_type == "sphere" else 1

        feet_global_linear_velocity_sensor_ids = [self.initial_mj_model.sensor(f"{foot_name}_global_linear_velocity").id for foot_name in self.feet_names]
        self.feet_global_linear_velocity_sensor_adrs_start = jnp.array([self.initial_mj_model.sensor_adr[sensor_id] for sensor_id in feet_global_linear_velocity_sensor_ids])

        self.body_to_parentid = jnp.array([self.initial_mj_model.body(body_id).parentid[0] for body_id in range(self.initial_mj_model.nbody)])
        body_to_children_count = jnp.array([jnp.sum(self.body_to_parentid == body_id) for body_id in range(self.initial_mj_model.nbody)])
        self.body_ids_of_actuator_joints = jnp.array([self.initial_mj_model.joint(joint_name).bodyid[0] for joint_name in self.actuator_joint_names])
        self.actuator_joint_nr_direct_child_actuator_joints = body_to_children_count[self.body_ids_of_actuator_joints]

        self.floor_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

        self.reward_collision_sphere_geom_ids = jnp.array([geom.id for geom in [self.initial_mj_model.geom(geom_id) for geom_id in range(self.initial_mj_model.ngeom)] if geom.group[0] == 5])

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
        self.single_action_space = BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(action_space_size,), dtype=jnp.float32)

        self.single_observation_space = self.get_observation_space(multi_robot_max_observation_size)

        self.observation_noise_function.init_attributes()

        del self.c_model, self.c_data


    @partial(jax.jit, static_argnums=(0,))
    def train_reset(self, key):
        return self.reset(key, False)


    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        mjx_model = self.initial_mjx_model
        data = self.mjx_data

        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        reward = 0.0
        terminated = False
        truncated = False

        internal_state = {
            "in_eval_mode": eval_mode,
            "env_curriculum_coeff": jnp.where(eval_mode, 1.0, 0.0),
            "env_curriculum_levels_in_a_row": 0.0,
            "actuator_joint_nominal_positions": self.initial_qpos[self.actuator_joint_mask_qpos],
            "actuator_joint_max_velocities": self.actuator_joint_max_velocities,
            "goal_velocities": jnp.array([0.0, 0.0, 0.0]),
            "imu_orientation_rotation": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]),
            "imu_orientation_rotation_inverse": Rotation.from_quat([0.0, 0.0, 0.0, 1.0]).inv(),
            "imu_orientation_euler": jnp.array([0.0, 0.0, 0.0]),
            "last_action": jnp.zeros(self.nr_actuator_joints),
            "second_last_action": jnp.zeros(self.nr_actuator_joints),
            "seen_gains_and_action_scaling_factor_normalized": jnp.zeros(3),
            "unseen_gains_and_action_scaling_factor_normalized": jnp.zeros(3),
            "seen_mass_normalized": jnp.zeros(1),
            "unseen_mass_normalized": jnp.zeros(1),
            "robot_dimensions_normalized": jnp.zeros(3),
            "max_command_velocity": 0.0,
            "nr_collisions_in_nominal": 0,
            "scaling_factor": self.robot_config["scaling_factor"],
            "robot_nominal_qpos_height_over_ground": self.initial_qpos[2],
            "robot_nominal_imu_height_over_ground": self.initial_imu_height,
        }
        self.gait_manager_function.init(internal_state)
        self.command_function.init(internal_state)
        self.reward_function.init(internal_state, mjx_model)
        self.terrain_function.init(internal_state)
        self.domain_randomization_action_delay_function.init(internal_state)
        self.domain_randomization_unseen_robot_function.init(internal_state)
        self.domain_randomization_seen_robot_function.init(internal_state)
        policy_joint_descriptions, critic_joint_descriptions, critic_feet_descriptions = self.get_description_vectors(internal_state, mjx_model)
        internal_state["policy_joint_descriptions"] = policy_joint_descriptions
        internal_state["critic_joint_descriptions"] = critic_joint_descriptions
        internal_state["critic_feet_descriptions"] = critic_feet_descriptions

        info = {}
        self.reward_function.reward_and_info(data, mjx_model, internal_state, jnp.zeros(self.nr_actuator_joints), info)
        info[f"rollout/episode_return/{self.robot_config['short_name']}"] = reward
        info[f"rollout/episode_length/{self.robot_config['short_name']}"] = 0
        info[f"env_curriculum/coefficient/{self.robot_config['short_name']}"] = internal_state["env_curriculum_coeff"]
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
        }

        state = State(mjx_model, data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, internal_state, key)

        return self._reset(state)


    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, initial_state_key, terrain_key, domain_randomization_key, observation_key, gait_manager_key, command_key = jax.random.split(state.key, 7)
        state = state.replace(key=key)

        mjx_model = self.terrain_function.sample(state.mjx_model, state.internal_state, terrain_key)

        data = self.mjx_data
        qpos, qvel = self.initial_state_function.setup(mjx_model, state.internal_state, initial_state_key)
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=state.internal_state["actuator_joint_nominal_positions"])

        new_state = state

        episode_success = new_state.info_episode_store["episode_return"] >= self.env_curriculum_level_success_episode_return
        new_state.internal_state["env_curriculum_levels_in_a_row"] = jnp.where(episode_success,
            jnp.where(new_state.internal_state["env_curriculum_levels_in_a_row"] >= 0,
                new_state.internal_state["env_curriculum_levels_in_a_row"] + 1,
                1
            ),
            jnp.where(new_state.internal_state["env_curriculum_levels_in_a_row"] < 0,
                new_state.internal_state["env_curriculum_levels_in_a_row"] - 1,
                -1
            )
        )
        new_state.internal_state["env_curriculum_coeff"] = jnp.clip(new_state.internal_state["env_curriculum_coeff"] + new_state.internal_state["env_curriculum_levels_in_a_row"] / self.env_curriculum_nr_levels, 0.0, 1.0)
        new_state.internal_state["env_curriculum_coeff"] = jnp.where(new_state.internal_state["in_eval_mode"], 1.0, new_state.internal_state["env_curriculum_coeff"])

        new_state.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[self.imu_site_id].reshape(3, 3))
        new_state.internal_state["imu_orientation_rotation_inverse"] = new_state.internal_state["imu_orientation_rotation"].inv()
        new_state.internal_state["imu_orientation_euler"] = new_state.internal_state["imu_orientation_rotation"].as_euler("xyz")
        new_state.internal_state["last_action"] = jnp.zeros(self.nr_actuator_joints)
        new_state.internal_state["second_last_action"] = jnp.zeros(self.nr_actuator_joints)
        self.gait_manager_function.setup(new_state.internal_state, gait_manager_key)
        self.reward_function.setup(new_state.internal_state)
        self.domain_randomization_action_delay_function.setup(new_state.internal_state)
        self.command_function.get_next_command(new_state.internal_state, jnp.array(True), command_key)
        data, mjx_model = self.handle_domain_randomization(new_state.internal_state, mjx_model, data, domain_randomization_key, is_episode_start=True)

        next_observation = self.get_observation(data, mjx_model, new_state.internal_state, observation_key, jnp.zeros(self.nr_actuator_joints))
        reward = 0.0
        terminated = False
        truncated = False
        info_episode_store = {
            "episode_return": reward,
            "episode_step": 0,
            "episode_total_xy_velocity_diff_abs": 0.0,
        }

        # Reset everything besides parts of the internal_state, info and the key
        new_state = new_state.replace(
            mjx_model=mjx_model,
            data=data,
            next_observation=next_observation, actual_next_observation=next_observation,
            reward=reward,
            terminated=terminated, truncated=truncated,
            info_episode_store=info_episode_store
        )

        return new_state


    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)


    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        key, action_delay_key, domain_randomization_key, command_sampling_key, command_key, observation_key, terrain_key = jax.random.split(state.key, 7)
        state = state.replace(key=key)

        chosen_action = action[:self.nr_actuator_joints]
        delayed_action = self.domain_randomization_action_delay_function.delay_action(chosen_action, state.internal_state, action_delay_key)

        target_joint_positions = self.control_function.process_action(delayed_action, state.internal_state)
        vmax_delta = state.internal_state["actuator_joint_max_velocities"] * self.dt
        target_joint_positions = jnp.clip(target_joint_positions, state.data.ctrl - vmax_delta, state.data.ctrl + vmax_delta)

        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(state.mjx_model, data.replace(ctrl=target_joint_positions)), None),
            init=state.data,
            xs=(),
            length=self.nr_substeps,
            unroll=True,
        )
        max_qvel = 100 * jnp.ones(self.initial_mj_model.nv)
        data = data.replace(qvel=jnp.clip(data.qvel, -max_qvel, max_qvel))

        state.internal_state["imu_orientation_rotation"] = Rotation.from_matrix(data.site_xmat[self.imu_site_id].reshape(3, 3))
        state.internal_state["imu_orientation_rotation_inverse"] = state.internal_state["imu_orientation_rotation"].inv()
        state.internal_state["imu_orientation_euler"] = state.internal_state["imu_orientation_rotation"].as_euler("xyz")

        data, mjx_model = self.handle_domain_randomization(state.internal_state, state.mjx_model, data, domain_randomization_key)
        state = state.replace(data=data, mjx_model=mjx_model)

        self.terrain_function.pre_step(data, state.internal_state)

        reward = self.reward_function.reward_and_info(data, mjx_model, state.internal_state, chosen_action, state.info)

        should_sample_commands = self.command_sampling_function.step(command_sampling_key)
        self.command_function.get_next_command(state.internal_state, should_sample_commands, command_key)

        next_observation = self.get_observation(data, mjx_model, state.internal_state, observation_key, chosen_action)
        terminated = self.termination_function.should_terminate(state.internal_state) | jnp.any(jnp.abs(data.qvel[:3]) == 100.0)
        truncated = state.info_episode_store["episode_step"] >= (self.horizon - 1)
        done = terminated | truncated

        data = self.terrain_function.post_step(data, mjx_model, state.internal_state, terrain_key)
        self.reward_function.step(data, state.internal_state)
        self.gait_manager_function.step(state.internal_state)

        state.internal_state["second_last_action"] = state.internal_state["last_action"]
        state.internal_state["last_action"] = chosen_action
        state.info_episode_store["episode_step"] += 1
        state.info_episode_store["episode_return"] += reward
        state.info_episode_store["episode_total_xy_velocity_diff_abs"] += state.info[f"env_info/xy_vel_diff_abs/{self.robot_config['short_name']}"]
        state.info[f"rollout/episode_return/{self.robot_config['short_name']}"] = jnp.where(done, state.info_episode_store["episode_return"], state.info[f"rollout/episode_return/{self.robot_config['short_name']}"])
        state.info[f"rollout/episode_length/{self.robot_config['short_name']}"] = jnp.where(done, state.info_episode_store["episode_step"], state.info[f"rollout/episode_length/{self.robot_config['short_name']}"])
        state.info[f"env_curriculum/coefficient/{self.robot_config['short_name']}"] = state.internal_state["env_curriculum_coeff"]

        def when_done(_):
            start_state = self._reset(state)
            start_state = start_state.replace(actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
            return start_state
        def when_not_done(_):
            return state.replace(data=data, next_observation=next_observation, actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
        state = jax.lax.cond(done, when_done, when_not_done, None)

        return state


    def get_observation(self, data, mjx_model, internal_state, key, action):
        # Dynamic observations
        policy_joint_observations = jnp.concatenate([
            internal_state["policy_joint_descriptions"],
            (data.qpos[self.actuator_joint_mask_qpos] - internal_state["actuator_joint_nominal_positions"]).reshape(-1, 1),
            data.qvel[self.actuator_joint_mask_qvel].reshape(-1, 1),
            action.reshape(-1, 1),
            internal_state["actuator_joint_keep_nominal"].reshape(-1, 1),
        ], axis=-1).reshape(-1)

        critic_joint_observations = jnp.concatenate([
            internal_state["critic_joint_descriptions"],
            (data.qpos[self.actuator_joint_mask_qpos] - internal_state["actuator_joint_nominal_positions"]).reshape(-1, 1),
            data.qvel[self.actuator_joint_mask_qvel].reshape(-1, 1),
            action.reshape(-1, 1),
            internal_state["actuator_joint_keep_nominal"].reshape(-1, 1),
        ], axis=-1).reshape(-1)

        critic_foot_observations = jnp.concatenate([
            internal_state["critic_feet_descriptions"],
            self.terrain_function.check_feet_floor_contact(data).reshape(self.nr_feet, 1),
            internal_state["feet_time_on_ground"].reshape(self.nr_feet, 1),
            internal_state["feet_time_in_air"].reshape(self.nr_feet, 1),
        ], axis=-1).reshape(-1)

        # General observations
        critic_imu_linear_velocity = data.sensordata[self.imu_linear_velocity_sensor_adr:self.imu_linear_velocity_sensor_adr + self.imu_linear_velocity_sensor_dim]
        policy_imu_angular_velocity = data.sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim]
        critic_imu_angular_velocity = data.sensordata[self.imu_angular_velocity_sensor_adr:self.imu_angular_velocity_sensor_adr + self.imu_angular_velocity_sensor_dim]
        policy_goal_velocities = internal_state["goal_velocities"]
        critic_goal_velocities = internal_state["goal_velocities"]
        policy_phase_features = self.gait_manager_function.get_phase_features(internal_state)
        critic_phase_features = policy_phase_features
        policy_projected_gravity_vector = internal_state["imu_orientation_rotation_inverse"].apply(jnp.array([0.0, 0.0, -1.0]))
        critic_projected_gravity_vector = internal_state["imu_orientation_rotation_inverse"].apply(jnp.array([0.0, 0.0, -1.0]))
        policy_exteroception = jnp.array([self.policy_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state)]).reshape(-1)
        critic_exteroception = jnp.array([self.critic_exteroceptive_observation_function.get_exteroceptive_observation(data, mjx_model, internal_state)]).reshape(-1)
        critic_curriculum_coeff = jnp.array([internal_state["env_curriculum_coeff"]])

        # General robot context
        policy_gains_and_action_scaling_factor = internal_state["seen_gains_and_action_scaling_factor_normalized"]
        critic_gains_and_action_scaling_factor = internal_state["unseen_gains_and_action_scaling_factor_normalized"]
        policy_mass = internal_state["seen_mass_normalized"]
        critic_mass = internal_state["unseen_mass_normalized"]
        policy_robot_dimensions = internal_state["robot_dimensions_normalized"]
        critic_robot_dimensions = internal_state["robot_dimensions_normalized"]
        policy_trunk_attributes = internal_state["seen_trunk_attributes_normalized"]
        critic_trunk_attributes = internal_state["unseen_trunk_attributes_normalized"]

        # Padding
        padding = jnp.zeros(self.missing_nr_of_observations)

        observation = jnp.concatenate([
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
        observation = self.observation_noise_function.modify_observation(internal_state, observation, key)

        # Normalize and clip
        observation = observation.at[self.policy_joint_positions_obs_idx].set(observation[self.policy_joint_positions_obs_idx] / 6.5)
        observation = observation.at[self.critic_joint_positions_obs_idx].set(observation[self.critic_joint_positions_obs_idx] / 6.5)
        observation = observation.at[self.policy_joint_velocities_obs_idx].set(observation[self.policy_joint_velocities_obs_idx] / 180.0)
        observation = observation.at[self.critic_joint_velocities_obs_idx].set(observation[self.critic_joint_velocities_obs_idx] / 180.0)
        observation = observation.at[self.policy_joint_previous_actions_obs_idx].set(observation[self.policy_joint_previous_actions_obs_idx] / 10.0)
        observation = observation.at[self.critic_joint_previous_actions_obs_idx].set(observation[self.critic_joint_previous_actions_obs_idx] / 10.0)
        observation = observation.at[self.critic_feet_ground_contact_obs_idx].set((observation[self.critic_feet_ground_contact_obs_idx] / 0.5) - 1.0)
        observation = observation.at[self.critic_feet_time_on_ground_obs_idx].set(jnp.clip((observation[self.critic_feet_time_on_ground_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.critic_feet_time_in_air_obs_idx].set(jnp.clip((observation[self.critic_feet_time_in_air_obs_idx] / (5.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.critic_imu_linear_vel_obs_idx].set(jnp.clip(observation[self.critic_imu_linear_vel_obs_idx] / 10.0, -1.0, 1.0))
        observation = observation.at[self.policy_imu_angular_vel_obs_idx].set(jnp.clip(observation[self.policy_imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0))
        observation = observation.at[self.critic_imu_angular_vel_obs_idx].set(jnp.clip(observation[self.critic_imu_angular_vel_obs_idx] / 50.0, -1.0, 1.0))
        if len(self.policy_exteroception_obs_idx) > 0:
            observation = observation.at[self.policy_exteroception_obs_idx].set(jnp.clip((observation[self.policy_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))
        if len(self.critic_exteroception_obs_idx) > 0:
            observation = observation.at[self.critic_exteroception_obs_idx].set(jnp.clip((observation[self.critic_exteroception_obs_idx] / (10.0 / 2)) - 1.0, -1.0, 1.0))
        observation = observation.at[self.critic_curriculum_coeff_obs_idx].set(jnp.clip((observation[self.critic_curriculum_coeff_obs_idx] / (1.0 / 2)) - 1.0, -1.0, 1.0))

        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = jnp.clip(observation, -10.0, 10.0)

        return observation


    def handle_domain_randomization(self, internal_state, mjx_model, data, key, is_episode_start=False):
        domain_sampling_key, domain_perturbation_sampling_key, unseen_robot_key, seen_robot_key, mujoco_model_key, action_delay_key, perturbation_key = jax.random.split(key, 7)

        should_randomize_domain_episode_start = self.domain_randomization_sampling_function.setup(domain_sampling_key)
        should_randomize_domain_perturbation_episode_start = self.domain_randomization_perturbation_sampling_function.setup(domain_perturbation_sampling_key, internal_state["env_curriculum_coeff"])
        should_randomize_domain_step = self.domain_randomization_sampling_function.step(domain_sampling_key)
        should_randomize_domain_perturbation_step = self.domain_randomization_perturbation_sampling_function.step(domain_perturbation_sampling_key, internal_state["env_curriculum_coeff"])
        should_randomize_domain = jnp.where(is_episode_start, should_randomize_domain_episode_start | internal_state["in_eval_mode"], should_randomize_domain_step)
        should_randomize_domain_perturbation = jnp.where(is_episode_start, should_randomize_domain_perturbation_episode_start, should_randomize_domain_perturbation_step)

        self.domain_randomization_unseen_robot_function.sample(internal_state, should_randomize_domain, unseen_robot_key)
        mjx_model, data = self.domain_randomization_seen_robot_function.sample(internal_state, mjx_model, data, should_randomize_domain, seen_robot_key)
        mjx_model = self.domain_randomization_mujoco_model_function.sample(internal_state, mjx_model, should_randomize_domain, mujoco_model_key)
        self.domain_randomization_action_delay_function.sample(internal_state, should_randomize_domain, action_delay_key)

        data = self.domain_randomization_perturbation_function.sample(internal_state, mjx_model, data, should_randomize_domain_perturbation, perturbation_key)

        if self.seen_robot_active and "policy_joint_descriptions" in internal_state:
            new_policy_jd, new_critic_jd, new_critic_fd = self.get_description_vectors(internal_state, mjx_model)
            internal_state["policy_joint_descriptions"] = jnp.where(should_randomize_domain, new_policy_jd, internal_state["policy_joint_descriptions"])
            internal_state["critic_joint_descriptions"] = jnp.where(should_randomize_domain, new_critic_jd, internal_state["critic_joint_descriptions"])
            internal_state["critic_feet_descriptions"] = jnp.where(should_randomize_domain, new_critic_fd, internal_state["critic_feet_descriptions"])

        return data, mjx_model


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
        self.policy_joint_descriptions_obs_idx = jnp.array(self.policy_joint_descriptions_obs_idx)
        self.policy_joint_positions_obs_idx = jnp.array(self.policy_joint_positions_obs_idx)
        self.policy_joint_velocities_obs_idx = jnp.array(self.policy_joint_velocities_obs_idx)
        self.policy_joint_previous_actions_obs_idx = jnp.array(self.policy_joint_previous_actions_obs_idx)
        self.policy_joint_keep_nominal_obs_idx = jnp.array(self.policy_joint_keep_nominal_obs_idx)
        self.policy_joint_observations_obs_idx = jnp.concatenate([
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
        self.critic_joint_descriptions_obs_idx = jnp.array(self.critic_joint_descriptions_obs_idx)
        self.critic_joint_positions_obs_idx = jnp.array(self.critic_joint_positions_obs_idx)
        self.critic_joint_velocities_obs_idx = jnp.array(self.critic_joint_velocities_obs_idx)
        self.critic_joint_previous_actions_obs_idx = jnp.array(self.critic_joint_previous_actions_obs_idx)
        self.critic_joint_keep_nominal_obs_idx = jnp.array(self.critic_joint_keep_nominal_obs_idx)
        self.critic_joint_observations_obs_idx = jnp.concatenate([
            self.critic_joint_positions_obs_idx,
            self.critic_joint_velocities_obs_idx,
            self.critic_joint_previous_actions_obs_idx,
            self.critic_joint_keep_nominal_obs_idx
        ])

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
        self.critic_feet_descriptions_obs_idx = jnp.array(self.critic_feet_descriptions_obs_idx)
        self.critic_feet_ground_contact_obs_idx = jnp.array(self.critic_feet_ground_contact_obs_idx)
        self.critic_feet_time_on_ground_obs_idx = jnp.array(self.critic_feet_time_on_ground_obs_idx)
        self.critic_feet_time_in_air_obs_idx = jnp.array(self.critic_feet_time_in_air_obs_idx)
        self.critic_feet_observations_obs_idx = jnp.concatenate([
            self.critic_feet_ground_contact_obs_idx,
            self.critic_feet_time_on_ground_obs_idx,
            self.critic_feet_time_in_air_obs_idx
        ])

        self.critic_imu_linear_vel_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3

        self.policy_imu_angular_vel_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3

        self.critic_imu_angular_vel_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3

        self.policy_goal_velocity_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3

        self.critic_goal_velocity_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3

        self.policy_gait_phase_obs_idx = jnp.array([current_observation_idx + i for i in range(4)])
        current_observation_idx += 4

        self.critic_gait_phase_obs_idx = jnp.array([current_observation_idx + i for i in range(4)])
        current_observation_idx += 4

        self.policy_gravity_vector_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3

        self.critic_gravity_vector_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3

        self.policy_exteroception_obs_idx = jnp.array([current_observation_idx + i for i in range(self.policy_exteroceptive_observation_function.nr_exteroceptive_observations)])
        current_observation_idx += self.policy_exteroceptive_observation_function.nr_exteroceptive_observations

        self.critic_exteroception_obs_idx = jnp.array([current_observation_idx + i for i in range(self.critic_exteroceptive_observation_function.nr_exteroceptive_observations)])
        current_observation_idx += self.critic_exteroceptive_observation_function.nr_exteroceptive_observations

        self.critic_curriculum_coeff_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1

        self.policy_gains_and_action_scaling_factor_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3

        self.critic_gains_and_action_scaling_factor_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3

        self.policy_mass_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1

        self.critic_mass_obs_idx = jnp.array([current_observation_idx])
        current_observation_idx += 1

        self.policy_robot_dimensions_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3

        self.critic_robot_dimensions_obs_idx = jnp.array([current_observation_idx + i for i in range(3)])
        current_observation_idx += 3

        self.policy_trunk_attributes_obs_idx = jnp.array([current_observation_idx + i for i in range(7)])
        current_observation_idx += 7

        self.critic_trunk_attributes_obs_idx = jnp.array([current_observation_idx + i for i in range(7)])
        current_observation_idx += 7

        self.policy_general_obs_idx = jnp.concatenate([
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

        self.critic_general_obs_idx = jnp.concatenate([
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

        return BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(current_observation_idx,), dtype=jnp.float32)


    def get_description_vectors(self, internal_state, mjx_model):
        qpos = self.initial_qpos.at[self.actuator_joint_mask_qpos].set(internal_state["actuator_joint_nominal_positions"])
        qpos = qpos.at[2].set(internal_state["robot_nominal_qpos_height_over_ground"] + internal_state["center_height"])
        data = self.mjx_data.replace(qpos=qpos)
        data = mjx.forward(mjx_model, data)

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
        global_frame_geom_sizes = jnp.matmul(geom_xmats, mjx_model.geom_size[1:].reshape(-1, 3, 1)).reshape(-1, 3)
        trunk_frame_geom_sizes = imu_orientation_rotation_inverse.apply(global_frame_geom_sizes)
        abs_rotated_geom_sizes = jnp.abs(trunk_frame_geom_sizes)

        global_frame_body_inertias = jnp.matmul(ximat, mjx_model.body_inertia.reshape(-1, 3, 1)).reshape(-1, 3)
        trunk_frame_body_inertias = imu_orientation_rotation_inverse.apply(global_frame_body_inertias)

        relative_geom_positions_minus_geom_size = relative_geom_positions - abs_rotated_geom_sizes
        relative_geom_positions_plus_geom_size = relative_geom_positions + abs_rotated_geom_sizes

        min_x = jnp.min(jnp.concatenate([relative_geom_positions_minus_geom_size[:, 0], relative_geom_positions_plus_geom_size[:, 0], relative_joint_positions[:, 0], relative_com_positions[:, 0]]))
        min_y = jnp.min(jnp.concatenate([relative_geom_positions_minus_geom_size[:, 1], relative_geom_positions_plus_geom_size[:, 1], relative_joint_positions[:, 1], relative_com_positions[:, 1]]))
        min_z = jnp.min(jnp.concatenate([relative_geom_positions_minus_geom_size[:, 2], relative_geom_positions_plus_geom_size[:, 2], relative_joint_positions[:, 2], relative_com_positions[:, 2]]))
        max_x = jnp.max(jnp.concatenate([relative_geom_positions_minus_geom_size[:, 0], relative_geom_positions_plus_geom_size[:, 0], relative_joint_positions[:, 0], relative_com_positions[:, 0]]))
        max_y = jnp.max(jnp.concatenate([relative_geom_positions_minus_geom_size[:, 1], relative_geom_positions_plus_geom_size[:, 1], relative_joint_positions[:, 1], relative_com_positions[:, 1]]))
        max_z = jnp.max(jnp.concatenate([relative_geom_positions_minus_geom_size[:, 2], relative_geom_positions_plus_geom_size[:, 2], relative_joint_positions[:, 2], relative_com_positions[:, 2]]))
        mins = jnp.array([min_x, min_y, min_z])
        robot_length = max_x - min_x
        robot_width = max_y - min_y
        robot_height = max_z - min_z
        robot_dimensions = jnp.array([robot_length, robot_width, robot_height])
        robot_dimensions_mean = jnp.mean(robot_dimensions)

        relative_joint_pos_normalized = (relative_joint_positions - mins) / robot_dimensions
        relative_feet_pos_normalized = (relative_geom_positions[self.foot_geom_indices - 1] - mins) / robot_dimensions
        relative_com_pos_normalized = (relative_com_positions - mins) / robot_dimensions

        seen_gains_and_action_scaling_factor = jnp.array([mjx_model.actuator_gainprm[0, 0], -mjx_model.actuator_biasprm[0, 2], internal_state["scaling_factor"]])
        seen_mass = jnp.array([jnp.sum(mjx_model.body_mass[1:])])

        unseen_gains_and_action_scaling_factor = jnp.array([mjx_model.actuator_gainprm[0, 0], -mjx_model.actuator_biasprm[0, 2], internal_state["scaling_factor"]])
        unseen_mass = jnp.array([jnp.sum(mjx_model.body_mass[1:])])

        internal_state["seen_gains_and_action_scaling_factor_normalized"] = (seen_gains_and_action_scaling_factor / jnp.array([170.0 / 2, 2.0 / 2, 1.5 / 2])) - 1.0
        internal_state["unseen_gains_and_action_scaling_factor_normalized"] = (unseen_gains_and_action_scaling_factor / jnp.array([170.0 / 2, 2.0 / 2, 1.5 / 2])) - 1.0
        internal_state["seen_mass_normalized"] = (seen_mass / (900.0 / 2)) - 1.0
        internal_state["unseen_mass_normalized"] = (unseen_mass / (900.0 / 2)) - 1.0
        internal_state["robot_dimensions_mean"] = robot_dimensions_mean
        internal_state["robot_dimensions_normalized"] = (robot_dimensions / (3.0 / 2)) - 1.0
        internal_state["max_command_velocity"] = jnp.minimum(robot_dimensions_mean * self.command_function.max_velocity_per_m_factor, self.command_function.clip_max_velocity)
        internal_state["seen_trunk_attributes_normalized"] = jnp.concatenate([
            (mjx_model.body_mass[self.trunk_body_id].reshape(-1) / (400.0 / 2)) - 1.0,
            (relative_com_pos_normalized[self.trunk_body_id].reshape(-1) / 0.5) - 1.0,
            (trunk_frame_body_inertias[self.trunk_body_id].reshape(-1) / (26.0 / 2)) - 1.0,
        ], axis=-1)
        internal_state["unseen_trunk_attributes_normalized"] = jnp.concatenate([
            (mjx_model.body_mass[self.trunk_body_id].reshape(-1) / (400.0 / 2)) - 1.0,
            (relative_com_pos_normalized[self.trunk_body_id].reshape(-1) / 0.5) - 1.0,
            (trunk_frame_body_inertias[self.trunk_body_id].reshape(-1) / (26.0 / 2)) - 1.0,
        ], axis=-1)

        critic_feet_description_vectors = jnp.concatenate([
            (relative_feet_pos_normalized / 0.5) - 1.0,
            (mjx_model.body_mass[self.body_ids_of_feet].reshape(-1, 1) / (15.0 / 2)) - 1.0,
            (relative_com_pos_normalized[self.body_ids_of_feet] / 0.5) - 1.0,
            (trunk_frame_body_inertias[self.body_ids_of_feet] / (0.65 / 2)) - 1.0,
            (trunk_frame_geom_sizes[self.foot_geom_indices - 1] / (0.2 / 2)) - 1.0,
            jnp.tile((self.foot_type_int / (1.0 / 2)) - 1.0, (self.nr_feet, 1)),
            jnp.tile(internal_state["unseen_gains_and_action_scaling_factor_normalized"], (self.nr_feet, 1)),
            jnp.tile(internal_state["unseen_mass_normalized"], (self.nr_feet, 1)),
            jnp.tile(internal_state["robot_dimensions_normalized"], (self.nr_feet, 1))
        ], axis=-1)

        policy_joint_description_vectors = jnp.concatenate([
            (relative_joint_pos_normalized / 0.5) - 1.0,
            relative_joint_axis,
            (mjx_model.body_mass[self.body_ids_of_actuator_joints].reshape(-1, 1) / (400.0 / 2)) - 1.0,
            (relative_com_pos_normalized[self.body_ids_of_actuator_joints] / 0.5) - 1.0,
            (trunk_frame_body_inertias[self.body_ids_of_actuator_joints] / (26.0 / 2)) - 1.0,
            (self.actuator_joint_nr_direct_child_actuator_joints.reshape(-1, 1) / (5.0 / 2)) - 1.0,
            internal_state["actuator_joint_nominal_positions"].reshape(-1, 1) / 6.5,
            (mjx_model.actuator_forcerange[:, 1].reshape(-1, 1) / (1600.0 / 2)) - 1.0,
            (internal_state["actuator_joint_max_velocities"].reshape(-1, 1) / (180.0 / 2)) - 1.0,
            (mjx_model.dof_damping[self.actuator_joint_mask_qvel].reshape(-1, 1) / (0.2 / 2)) - 1.0,
            (mjx_model.dof_armature[self.actuator_joint_mask_qvel].reshape(-1, 1) / (2.7 / 2)) - 1.0,
            (mjx_model.jnt_stiffness[self.actuator_joint_mask_joints].reshape(-1, 1) / (0.2 / 2)) - 1.0,
            (mjx_model.dof_frictionloss[self.actuator_joint_mask_qvel].reshape(-1, 1) / (37.0 / 2)) - 1.0,
            mjx_model.jnt_range[self.actuator_joint_mask_joints] / 6.5,
            jnp.tile(internal_state["seen_gains_and_action_scaling_factor_normalized"], (self.nr_actuator_joints, 1)),
            jnp.tile(internal_state["seen_mass_normalized"], (self.nr_actuator_joints, 1)),
            jnp.tile(internal_state["robot_dimensions_normalized"], (self.nr_actuator_joints, 1)),
        ], axis=-1)

        critic_joint_description_vectors = policy_joint_description_vectors

        return policy_joint_description_vectors, critic_joint_description_vectors, critic_feet_description_vectors


    @partial(jax.jit, static_argnums=(0,))
    def extract_embodiment_observation_patch(self, state):
        desc_parts = []

        if len(self.co_design_joint_description_indices) > 0:
            joint_description_indices = jnp.array(self.co_design_joint_description_indices, dtype=jnp.int32)
            joint_descriptions = state.internal_state["critic_joint_descriptions"][:, :, joint_description_indices]
            desc_parts.append(joint_descriptions.reshape(state.next_observation.shape[0], -1))

        if len(self.co_design_foot_description_indices) > 0:
            foot_description_indices = jnp.array(self.co_design_foot_description_indices, dtype=jnp.int32)
            foot_descriptions = state.internal_state["critic_feet_descriptions"][:, :, foot_description_indices]
            desc_parts.append(foot_descriptions.reshape(state.next_observation.shape[0], -1))

        if len(self.co_design_general_obs_indices) > 0:
            observation = state.next_observation
            observation = observation.at[:, self.critic_gains_and_action_scaling_factor_obs_idx].set(state.internal_state["unseen_gains_and_action_scaling_factor_normalized"])
            observation = observation.at[:, self.critic_mass_obs_idx].set(state.internal_state["unseen_mass_normalized"])
            observation = observation.at[:, self.critic_robot_dimensions_obs_idx].set(state.internal_state["robot_dimensions_normalized"])
            observation = observation.at[:, self.critic_trunk_attributes_obs_idx].set(state.internal_state["unseen_trunk_attributes_normalized"])
            general_indices = self.critic_general_obs_idx[jnp.array(self.co_design_general_obs_indices, dtype=jnp.int32)]
            desc_parts.append(observation[:, general_indices])

        if len(desc_parts) == 0:
            return jnp.zeros((state.next_observation.shape[0], 0), dtype=jnp.float32)
        return jnp.concatenate(desc_parts, axis=-1)


    def factor_dim(self):
        if not self.seen_robot_active:
            return 0
        return self.domain_randomization_seen_robot_function.total_factor_dim()


    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def set_factor_params(self, state, factor):
        if not self.seen_robot_active:
            return state
        fake_key = jax.random.key(0)
        new_mjx_model, new_data = self.domain_randomization_seen_robot_function.sample(
            state.internal_state, state.mjx_model, state.data,
            should_randomize=jnp.array(True), key=fake_key, fixed_factor=factor,
        )
        policy_jd, critic_jd, critic_fd = self.get_description_vectors(state.internal_state, new_mjx_model)
        new_internal_state = dict(state.internal_state)
        new_internal_state["policy_joint_descriptions"] = policy_jd
        new_internal_state["critic_joint_descriptions"] = critic_jd
        new_internal_state["critic_feet_descriptions"] = critic_fd
        return state.replace(mjx_model=new_mjx_model, data=new_data, internal_state=new_internal_state)


    @partial(jax.jit, static_argnums=(0,))
    def get_factor_observation_patch(self, state, factor):
        state = self.set_factor_params(state, factor)
        return self.extract_embodiment_observation_patch(state)


    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def sample_initial_command(self, state, key):
        new_internal_state = dict(state.internal_state)
        self.command_function.get_next_command(new_internal_state, jnp.array(True), key)
        return state.replace(internal_state=new_internal_state)


    @partial(jax.jit, static_argnums=(0,))
    def get_curriculum_coefficients(self, state):
        return state.internal_state["env_curriculum_coeff"]


    def close(self):
        pass
