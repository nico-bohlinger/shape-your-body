import os
import shutil
import tempfile
import json
from copy import deepcopy
import logging
import time
import tree
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from shape_your_body.algorithms.urmav2_for_codesign.mjx.general_properties import GeneralProperties
from shape_your_body.algorithms.urmav2_for_codesign.mjx.policy import get_policy
from shape_your_body.algorithms.urmav2_for_codesign.mjx.critic import get_critic
from shape_your_body.algorithms.design_search.surrogate import run_surrogate_search_urma
from shape_your_body.algorithms.design_search.eval_utils import run_state_bank_export, run_design_evaluation

rlx_logger = logging.getLogger("rl_x")


class URMAv2ForCodesign:
    def __init__(self, config, train_env, eval_env, run_path, writer):
        # Arguments
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.writer = writer


        # Simple config values
        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.start_learning_rate = config.algorithm.start_learning_rate
        self.end_learning_rate = config.algorithm.end_learning_rate
        self.learning_rate_schedule = config.algorithm.learning_rate_schedule
        self.nr_steps = config.algorithm.nr_steps
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.kl_threshold = config.algorithm.kl_threshold
        self.nr_critic_heads = config.algorithm.nr_critic_heads
        self.mean_critic_returns = config.algorithm.mean_critic_returns
        self.embodiment_local_resample_probability = config.algorithm.embodiment_local_resample_probability
        self.embodiment_local_resample_range_fraction = config.algorithm.embodiment_local_resample_range_fraction
        self.evaluation_and_save_frequency = config.algorithm.evaluation_and_save_frequency


        # Derived config values
        self.nr_train_robots = len(config.environment.train_robots)
        if self.nr_envs % self.nr_train_robots != 0:
            raise ValueError(f"nr_envs ({self.nr_envs}) must be divisible by nr_train_robots ({self.nr_train_robots})")
        self.nr_envs_per_train_robot = self.nr_envs // self.nr_train_robots

        self.batch_size = self.nr_envs * self.nr_steps
        if self.track_wandb:
            wandb.config["algorithm"]["batch_size"] = self.batch_size

        if config.runner.mode == "train":
            assert self.total_timesteps % self.batch_size == 0, "Total timesteps must be a multiple of batch size"
        self.nr_updates = self.total_timesteps // self.batch_size
        if self.track_wandb:
            wandb.config["algorithm"]["nr_updates"] = self.nr_updates

        if self.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * self.nr_updates
        if self.track_wandb:
            wandb.config["algorithm"]["evaluation_and_save_frequency"] = self.evaluation_and_save_frequency
        if config.runner.mode == "train":
            assert self.evaluation_and_save_frequency % self.batch_size == 0, "Evaluation and save frequency must be a multiple of batch size"
        self.nr_updates_per_multi_learning_iteration = self.evaluation_and_save_frequency // self.batch_size

        if config.runner.mode == "train":
            assert self.total_timesteps % self.evaluation_and_save_frequency == 0, "Total timesteps must be a multiple of evaluation and save frequency"
        self.nr_multi_learning_and_eval_save_iterations = self.total_timesteps // self.evaluation_and_save_frequency

        self.nr_devices = jax.local_device_count()
        nr_devices_str = f"{self.nr_devices} devices" if self.nr_devices > 1 else "device"
        rlx_logger.info(f"Using {nr_devices_str}: {jax.default_backend()}")
        if self.track_wandb:
            wandb.config["runner"]["used_device"] = jax.default_backend()
            wandb.config["runner"]["nr_devices"] = self.nr_devices

        if config.runner.mode == "train":
            assert self.nr_epochs % self.nr_devices == 0, "Number of epochs must be divisible by the number of devices"
        self.nr_epochs //= self.nr_devices
        if self.track_wandb:
            wandb.config["algorithm"]["nr_epochs_per_device"] = self.nr_epochs

        if config.runner.mode == "train":
            assert self.minibatch_size % self.nr_devices == 0, "Minibatch size must be divisible by the number of devices"
        self.minibatch_size //= self.nr_devices
        if self.track_wandb:
            wandb.config["algorithm"]["minibatch_size_per_device"] = self.minibatch_size

        if config.runner.mode == "train":
            assert self.batch_size % self.minibatch_size == 0, "Batch size must be divisible by minibatch size"
        self.nr_minibatches = self.batch_size // self.minibatch_size
        if config.runner.mode == "train":
            assert self.minibatch_size % self.nr_train_robots == 0, "Minibatch size (per device) must be divisible by nr_train_robots"
        self.minibatch_size_per_robot = self.minibatch_size // self.nr_train_robots
        if self.track_wandb:
            wandb.config["algorithm"]["nr_minibatches_per_epoch_per_device"] = self.nr_minibatches
            wandb.config["algorithm"]["nr_minibatches_per_epoch"] = self.nr_minibatches // self.nr_devices
            wandb.config["algorithm"]["minibatch_size_per_robot_per_device"] = self.minibatch_size_per_robot

        self.nr_minibatches_per_update = self.nr_minibatches * self.nr_epochs
        if self.track_wandb:
            wandb.config["algorithm"]["nr_minibatches_per_update"] = self.nr_minibatches_per_update

        self.nr_minibatches_during_training = self.nr_minibatches_per_update * self.nr_updates
        if self.track_wandb:
            wandb.config["algorithm"]["nr_minibatches_during_training"] = self.nr_minibatches_during_training

        # Environment specific values
        self.train_robot_episode_return_keys = [f"rollout/episode_return/{short_name}" for short_name in self.train_env.train_robot_short_names]
        self.train_robot_episode_length_keys = [f"rollout/episode_length/{short_name}" for short_name in self.train_env.train_robot_short_names]
        self.train_robot_xy_vel_diff_abs_keys = [f"env_info/xy_vel_diff_abs/{short_name}" for short_name in self.train_env.train_robot_short_names]
        self.train_robot_env_curriculum_coefficient_keys = [f"env_curriculum/coefficient/{short_name}" for short_name in self.train_env.train_robot_short_names]

        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape

        self.joint_description_size = self.train_env.joint_description_size
        self.joint_observation_size = self.train_env.joint_observation_size
        self.foot_description_size = self.train_env.foot_description_size
        self.foot_observation_size = self.train_env.foot_observation_size
        self.nr_joints_for_robots = tuple(int(j) for j in self.train_env.call("nr_actuator_joints"))
        self.nr_feet_for_robots = tuple(int(f) for f in self.train_env.call("nr_feet"))
        self.policy_general_idx_for_robots = tuple(jnp.asarray(idx, dtype=jnp.int32) for idx in self.train_env.call("policy_general_obs_idx"))
        self.critic_general_idx_for_robots = tuple(jnp.asarray(idx, dtype=jnp.int32) for idx in self.train_env.call("critic_general_obs_idx"))
        self.missing_nr_of_actions_for_robots = tuple(int(m) for m in self.train_env.call("missing_nr_of_actions"))
        self.nr_actuator_joints = self.nr_joints_for_robots[0]
        self.nr_feet = self.nr_feet_for_robots[0]
        self.critic_general_obs_idx = tuple(int(x) for x in self.critic_general_idx_for_robots[0])
        self.softmax_temperature_min = config.algorithm.softmax_temperature_min


        # Setup
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, train_reset_key, _ = jax.random.split(self.key, 5)
        train_reset_key = jax.random.split(train_reset_key, self.nr_envs)

        def linear_schedule(count):
            fraction = 1.0 - (count // self.nr_minibatches_per_update) / self.nr_updates
            learning_rate = self.end_learning_rate + fraction * (self.start_learning_rate - self.end_learning_rate)
            return learning_rate

        def exponential_schedule(count):
            k = -jnp.log(self.end_learning_rate / self.start_learning_rate) / self.nr_minibatches_during_training
            learning_rate = self.start_learning_rate * jnp.exp(-1.0 * k * count)
            return jnp.maximum(learning_rate, self.end_learning_rate)

        if self.learning_rate_schedule == "linear":
            schedule_function = linear_schedule
        elif self.learning_rate_schedule == "exponential":
            schedule_function = exponential_schedule

        self.learning_rate = schedule_function if self.start_learning_rate != self.end_learning_rate else self.start_learning_rate

        self.train_env.init(train_reset_key)

        multi_env_state = self.train_env.train_reset(train_reset_key)

        if self.train_env.factor_dim() == 0:
            raise ValueError("The seen_robot domain randomization of the environment must be active")
        self.embodiment_ndim = int(self.train_env.factor_dim())
        self.co_design_joint_description_indices = tuple(int(x) for x in self.train_env.co_design_joint_description_indices)
        self.co_design_foot_description_indices = tuple(int(x) for x in self.train_env.co_design_foot_description_indices)
        self.co_design_general_obs_indices = tuple(int(x) for x in self.train_env.co_design_general_obs_indices)

        JB = self.joint_description_size + self.joint_observation_size
        FB = self.foot_description_size + self.foot_observation_size
        c0_joints = self.nr_actuator_joints * JB
        f0_feet = c0_joints + self.nr_actuator_joints * JB
        critic_general_obs_idx_arr = np.asarray(self.critic_general_obs_idx, dtype=np.int32)
        joint_obs_indices = [c0_joints + j * JB + k for j in range(self.nr_actuator_joints) for k in self.co_design_joint_description_indices]
        foot_obs_indices = [f0_feet + fi * FB + k for fi in range(self.nr_feet) for k in self.co_design_foot_description_indices]
        general_obs_indices = [int(critic_general_obs_idx_arr[k]) for k in self.co_design_general_obs_indices]
        self.embodiment_observation_indices = jnp.array(joint_obs_indices + foot_obs_indices + general_obs_indices, dtype=jnp.int32)

        self.embodiment_particles = jnp.zeros((self.nr_envs, self.embodiment_ndim), dtype=jnp.float32)
        self.embodiment_pending_particles = self.embodiment_particles.copy()

        multi_env_state = multi_env_state.replace(
            internal_state={**multi_env_state.internal_state, "in_eval_mode": jnp.ones_like(multi_env_state.internal_state["in_eval_mode"])}
        )
        multi_env_state = self.train_env.set_factor_params(multi_env_state, self.embodiment_particles)

        dummy_observation = multi_env_state.next_observation


        self.policy, self.get_processed_action = get_policy(self.config)
        self.critic = get_critic(self.config)

        dummy_obs = dummy_observation[0]
        (dp_jd, dp_jo, dp_gs, dc_jd, dc_jo, dc_fd, dc_fo, dc_gs) = self.decode_train_obs(dummy_obs, 0)

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, dp_jd, dp_jo, dp_gs),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, dc_jd, dc_jo, dc_fd, dc_fo, dc_gs),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=self.learning_rate),
            )
        )

        del dummy_observation, train_reset_key

        nr_policy_params = sum([p.size for p in tree.flatten(self.policy_state.params)])
        nr_critic_params = sum([p.size for p in tree.flatten(self.critic_state.params)])
        rlx_logger.info(f"Policy has {nr_policy_params} parameters")
        rlx_logger.info(f"Critic has {nr_critic_params} parameters")
        if self.track_wandb:
            wandb.config["algorithm"]["nr_policy_params"] = nr_policy_params
            wandb.config["algorithm"]["nr_critic_params"] = nr_critic_params

        if self.save_model:
            os.makedirs(self.save_path)
            self.latest_model_file_name = "latest.model"
            self.latest_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

        if self.track_wandb:
            wandb.config.update({"algorithm": wandb.config["algorithm"]})

        if self.config.algorithm.print_configs:
            rlx_logger.info(f"Config:\n{json.dumps(self.config.to_dict(), indent=4)}")


    def decode_train_obs(self, obs, i):
        J = self.nr_joints_for_robots[i]
        F = self.nr_feet_for_robots[i]
        JD = self.joint_description_size
        JO = self.joint_observation_size
        FD = self.foot_description_size
        FO = self.foot_observation_size
        JB = JD + JO
        FB = FD + FO

        p0 = 0
        p1 = J * JB
        c0 = p1
        c1 = c0 + J * JB
        f0 = c1
        f1 = f0 + F * FB

        p = obs[..., p0:p1].reshape(obs.shape[:-1] + (J, JB))
        policy_joint_descriptions = p[..., :JD]
        policy_joint_observations = p[..., JD:]

        c = obs[..., c0:c1].reshape(obs.shape[:-1] + (J, JB))
        critic_joint_descriptions = c[..., :JD]
        critic_joint_observations = c[..., JD:]

        f = obs[..., f0:f1].reshape(obs.shape[:-1] + (F, FB))
        critic_foot_descriptions = f[..., :FD]
        critic_foot_observations = f[..., FD:]

        policy_general_state = jnp.take(obs, self.policy_general_idx_for_robots[i], axis=-1)
        critic_general_state = jnp.take(obs, self.critic_general_idx_for_robots[i], axis=-1)

        return (
            policy_joint_descriptions,
            policy_joint_observations,
            policy_general_state,
            critic_joint_descriptions,
            critic_joint_observations,
            critic_foot_descriptions,
            critic_foot_observations,
            critic_general_state,
        )


    def decode_policy_obs(self, obs, i):
        J = self.nr_joints_for_robots[i]
        JD = self.joint_description_size
        JO = self.joint_observation_size
        JB = JD + JO

        p = obs[..., :J * JB].reshape(obs.shape[:-1] + (J, JB))
        policy_joint_descriptions = p[..., :JD]
        policy_joint_observations = p[..., JD:]
        policy_general_state = jnp.take(obs, self.policy_general_idx_for_robots[i], axis=-1)

        return policy_joint_descriptions, policy_joint_observations, policy_general_state


    def train(self):
        def jitable_train_function(policy_state, critic_state, embodiment_particles, embodiment_pending_particles, multi_env_state, key, device_id):
            prev_policy_state = policy_state

            multi_env_state = self.train_env.set_factor_params(multi_env_state, embodiment_particles)

            def multi_learning_and_eval_save_iteration(multi_learning_and_eval_save_iteration_carry, multi_learning_iteration_step):
                policy_state, prev_policy_state, critic_state, multi_env_state, embodiment_particles, embodiment_pending_particles, key = multi_learning_and_eval_save_iteration_carry

                def learning_iteration(learning_iteration_carry, learning_iteration_step):
                    policy_state, prev_policy_state, critic_state, multi_env_state, embodiment_particles, embodiment_pending_particles, key  = learning_iteration_carry

                    # Acting
                    def single_rollout(single_rollout_carry, _):
                        policy_state, critic_state, multi_env_state, embodiment_particles, embodiment_pending_particles, key = single_rollout_carry

                        observation = multi_env_state.next_observation
                        obs_r = observation.reshape(self.nr_train_robots, self.nr_envs_per_train_robot, -1)

                        key, subkey = jax.random.split(key)
                        std_dev_keys = jax.random.split(subkey, self.nr_train_robots)

                        processed_actions = []
                        actions_padded = []
                        values_list = []
                        log_probs_list = []
                        for i in range(self.nr_train_robots):
                            obs_i = obs_r[i]
                            (p_jd_i, p_jo_i, p_gs_i, c_jd_i, c_jo_i, c_fd_i, c_fo_i, c_gs_i) = self.decode_train_obs(obs_i, i)

                            action_mean_i, action_logstd_i = self.policy.apply(policy_state.params, p_jd_i, p_jo_i, p_gs_i)
                            action_std_i = jnp.exp(action_logstd_i)
                            action_i = action_mean_i + action_std_i * jax.random.normal(std_dev_keys[i], shape=action_mean_i.shape)
                            log_prob_i = (-0.5 * ((action_i - action_mean_i) / action_std_i) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd_i).sum(axis=-1)

                            action_padded_i = jnp.concatenate([action_i, jnp.zeros((self.nr_envs_per_train_robot, self.missing_nr_of_actions_for_robots[i]))], axis=-1)
                            processed_action_i = self.get_processed_action(action_padded_i)

                            values_i = self.critic.apply(critic_state.params, c_jd_i, c_jo_i, c_fd_i, c_fo_i, c_gs_i)

                            processed_actions.append(processed_action_i)
                            actions_padded.append(action_padded_i)
                            values_list.append(values_i)
                            log_probs_list.append(log_prob_i)

                        processed_action = jnp.concatenate(processed_actions, axis=0)
                        action = jnp.concatenate(actions_padded, axis=0)
                        values = jnp.concatenate(values_list, axis=0)
                        log_prob = jnp.concatenate(log_probs_list, axis=0)

                        multi_env_state = self.train_env.step(multi_env_state, processed_action)
                        actual_next_state = multi_env_state.actual_next_observation
                        transition = (observation, actual_next_state, action, multi_env_state.reward, values, multi_env_state.terminated, log_prob, multi_env_state.info, )

                        done_mask = multi_env_state.terminated | multi_env_state.truncated
                        embodiment_particles = jnp.where(done_mask[:, None], embodiment_pending_particles, embodiment_particles)
                        multi_env_state = self.train_env.set_factor_params(multi_env_state, embodiment_particles)

                        return (policy_state, critic_state, multi_env_state, embodiment_particles, embodiment_pending_particles, key), transition

                    init_carry = (policy_state, critic_state, multi_env_state, embodiment_particles, embodiment_pending_particles, key)
                    single_rollout_carry, batch = jax.lax.scan(single_rollout, init_carry, None, self.nr_steps)
                    policy_state, critic_state, multi_env_state, embodiment_particles, embodiment_pending_particles, key = single_rollout_carry
                    states, next_states, actions, rewards, values, terminations, log_probs, infos = batch


                    # Calculating advantages and returns
                    def calculate_gae_advantages(critic_state, next_states, rewards, values, terminations):
                        next_states_r = next_states.reshape(self.nr_steps, self.nr_train_robots, self.nr_envs_per_train_robot, -1)
                        next_values_list = []
                        for i in range(self.nr_train_robots):
                            next_states_i = next_states_r[:, i]
                            (_, _, _, c_jd_i, c_jo_i, c_fd_i, c_fo_i, c_gs_i) = self.decode_train_obs(next_states_i, i)
                            next_values_i = self.critic.apply(critic_state.params, c_jd_i, c_jo_i, c_fd_i, c_fo_i, c_gs_i)
                            next_values_list.append(next_values_i)
                        next_values = jnp.concatenate(next_values_list, axis=1)

                        delta = rewards[..., None] + self.gamma * next_values * (1.0 - terminations[..., None]) - values

                        def remaining_steps(remaining_next, termination_t):
                            remaining_t = jnp.where(termination_t, 1, remaining_next + 1)
                            return remaining_t, remaining_t

                        _, remaining_steps_reverse = jax.lax.scan(remaining_steps, jnp.zeros((self.nr_envs,), dtype=int), terminations[::-1], unroll=True)
                        remaining_steps = remaining_steps_reverse[::-1]

                        lambda_pow_rem = jnp.power(self.gae_lambda, remaining_steps)
                        lambda_pow_rem_m1 = jnp.power(self.gae_lambda, remaining_steps - 1)

                        denominator = 1 - lambda_pow_rem
                        coef = self.gamma * self.gae_lambda * (1 - lambda_pow_rem_m1) / (denominator + 1e-8)

                        def adv_step(adv_next, inputs):
                            delta_t, coef_t = inputs
                            adv_t = delta_t + coef_t * adv_next
                            return adv_t, adv_t

                        _, advantages_reverse = jax.lax.scan(adv_step, jnp.zeros((self.nr_envs, self.nr_critic_heads)), (delta[::-1], coef[::-1][..., None]), unroll=True)
                        advantages = advantages_reverse[::-1]
                        returns = advantages + values
                        advantages = advantages.mean(axis=-1)
                        if self.mean_critic_returns:
                            returns = returns.mean(axis=-1)
                        return advantages, returns

                    advantages, returns = calculate_gae_advantages(critic_state, next_states, rewards, values, terminations)


                    # Optimizing
                    def loss_fn(policy_params, critic_params, states, actions, log_probs, returns, advantages):
                        losses = []
                        metrics_list = []
                        for i in range(self.nr_train_robots):
                            state_i = states[i]
                            (p_jd_i, p_jo_i, p_gs_i, c_jd_i, c_jo_i, c_fd_i, c_fo_i, c_gs_i) = self.decode_train_obs(state_i, i)

                            J_i = self.nr_joints_for_robots[i]
                            action_i = actions[i, :J_i]
                            log_prob_i = log_probs[i]
                            return_i = returns[i]
                            advantage_i = advantages[i]

                            action_mean_i, action_logstd_i = self.policy.apply(policy_params, p_jd_i, p_jo_i, p_gs_i)
                            action_std_i = jnp.exp(action_logstd_i)
                            action_std_mean = action_std_i.mean()
                            new_log_prob = (-0.5 * ((action_i - action_mean_i) / action_std_i) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd_i).sum(axis=-1)
                            entropy = (action_logstd_i + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(axis=-1)

                            logratio = new_log_prob - log_prob_i
                            ratio = jnp.exp(logratio)
                            approx_kl_div = (ratio - 1) - logratio
                            clip_fraction = jnp.float32((jnp.abs(ratio - 1) > self.clip_range))

                            pg_loss1 = -advantage_i * ratio
                            pg_loss2 = -advantage_i * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                            pg_loss = jnp.maximum(pg_loss1, pg_loss2)

                            new_values_i = self.critic.apply(critic_params, c_jd_i, c_jo_i, c_fd_i, c_fo_i, c_gs_i)
                            critic_loss = 0.5 * (new_values_i - return_i) ** 2

                            loss_i = pg_loss - self.entropy_coef * entropy + self.critic_coef * critic_loss

                            metric_i = {
                                "loss/policy_gradient_loss": pg_loss,
                                "loss/critic_loss": critic_loss,
                                "loss/entropy_loss": entropy,
                                "policy_ratio/approx_kl": approx_kl_div,
                                "policy_ratio/clip_fraction": clip_fraction,
                                "policy/std_dev": action_std_mean,
                            }

                            losses.append(loss_i)
                            metrics_list.append(metric_i)

                        loss = jnp.mean(jnp.array([jnp.mean(l) for l in losses]))
                        metrics = jax.tree.map(lambda *x: jnp.mean(jnp.array(x)), *metrics_list)
                        return loss, metrics


                    batch_per_robot = self.nr_steps * self.nr_envs_per_train_robot
                    batch_states = states.reshape((self.nr_steps, self.nr_train_robots, self.nr_envs_per_train_robot, *self.os_shape)).transpose((0, 2, 1, 3)).reshape((batch_per_robot, self.nr_train_robots, *self.os_shape))
                    batch_actions = actions.reshape((self.nr_steps, self.nr_train_robots, self.nr_envs_per_train_robot, *self.as_shape)).transpose((0, 2, 1, 3)).reshape((batch_per_robot, self.nr_train_robots, *self.as_shape))
                    batch_log_probs = log_probs.reshape((self.nr_steps, self.nr_train_robots, self.nr_envs_per_train_robot)).transpose((0, 2, 1)).reshape((batch_per_robot, self.nr_train_robots))
                    batch_advantages = advantages.reshape((self.nr_steps, self.nr_train_robots, self.nr_envs_per_train_robot)).transpose((0, 2, 1)).reshape((batch_per_robot, self.nr_train_robots))
                    if self.mean_critic_returns:
                        batch_returns = returns.reshape((self.nr_steps, self.nr_train_robots, self.nr_envs_per_train_robot)).transpose((0, 2, 1)).reshape((batch_per_robot, self.nr_train_robots))
                    else:
                        batch_returns = returns.reshape((self.nr_steps, self.nr_train_robots, self.nr_envs_per_train_robot, self.nr_critic_heads)).transpose((0, 2, 1, 3)).reshape((batch_per_robot, self.nr_train_robots, self.nr_critic_heads))

                    vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
                    safe_mean = lambda x: jnp.mean(x) if x is not None else x
                    mean_vmapped_loss_fn = lambda *a, **k: jax.tree.map(safe_mean, vmap_loss_fn(*a, **k))
                    grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

                    key, subkey = jax.random.split(key)
                    batch_indices = jnp.tile(jnp.arange(batch_per_robot), (self.nr_epochs, 1))
                    batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
                    batch_indices = batch_indices.reshape((self.nr_minibatches_per_update, self.minibatch_size_per_robot))

                    def minibatch_update(carry, minibatch_indices):
                        policy_state, critic_state = carry

                        minibatch_advantages = batch_advantages[minibatch_indices]
                        minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

                        (loss, metrics), (policy_grads, critic_grads) = grad_loss_fn(
                            policy_state.params,
                            critic_state.params,
                            batch_states[minibatch_indices],
                            batch_actions[minibatch_indices],
                            batch_log_probs[minibatch_indices],
                            batch_returns[minibatch_indices],
                            minibatch_advantages,
                        )

                        policy_grads = jax.lax.pmean(policy_grads, axis_name="i")
                        critic_grads = jax.lax.pmean(critic_grads, axis_name="i")

                        policy_state = policy_state.apply_gradients(grads=policy_grads)
                        critic_state = critic_state.apply_gradients(grads=critic_grads)

                        metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_grads)
                        metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_grads)

                        return (policy_state, critic_state), (metrics)

                    carry, optimization_metrics = jax.lax.scan(minibatch_update, (policy_state, critic_state), batch_indices)
                    new_policy_state, new_critic_state = carry

                    new_policy_state = jax.lax.cond(
                        (jnp.mean(optimization_metrics["policy_ratio/approx_kl"]) > self.kl_threshold) | (jnp.mean(optimization_metrics["policy_ratio/clip_fraction"]) == 0.0),
                        lambda _: prev_policy_state,
                        lambda _: new_policy_state,
                        operand=None,
                    )
                    prev_policy_state = policy_state

                    optimization_metrics["lr/learning_rate"] = new_policy_state.opt_state[1].hyperparams["learning_rate"]
                    optimization_metrics["policy/softmax_temperature"] = jnp.exp(new_policy_state.params["params"]["joint_log_softmax_temperature"]).squeeze() + self.softmax_temperature_min

                    rollout_episode_return_per_env = jnp.concatenate([jnp.mean(infos[key], axis=0) for key in self.train_robot_episode_return_keys], axis=0)
                    rollout_episode_length_per_env = jnp.concatenate([jnp.mean(infos[key], axis=0) for key in self.train_robot_episode_length_keys], axis=0)
                    xy_vel_diff_abs_per_env = jnp.concatenate([jnp.mean(infos[key], axis=0) for key in self.train_robot_xy_vel_diff_abs_keys], axis=0)
                    curriculum_coefficient_per_env = jnp.concatenate([jnp.mean(infos[key], axis=0) for key in self.train_robot_env_curriculum_coefficient_keys], axis=0)

                    # Resampling of the embodiment particles
                    curriculum_coefficients = self.train_env.get_curriculum_coefficients(multi_env_state)[:, None]
                    key, _, _, local_resample_mask_key, local_particle_key = jax.random.split(key, 5)
                    local_resample_mask = jax.random.bernoulli(local_resample_mask_key, p=self.embodiment_local_resample_probability, shape=(self.nr_envs,))
                    local_perturbation = jax.random.uniform(local_particle_key, shape=(self.nr_envs, self.embodiment_ndim), minval=-1.0, maxval=1.0)
                    local_range = self.embodiment_local_resample_range_fraction * curriculum_coefficients
                    locally_resampled_particles = jnp.clip(embodiment_pending_particles + local_perturbation * local_range, -curriculum_coefficients, curriculum_coefficients)
                    embodiment_pending_particles = jnp.where(local_resample_mask[:, None], locally_resampled_particles, embodiment_pending_particles)

                    # Logging
                    combined_metrics = {**infos, **optimization_metrics}
                    combined_metrics = jax.tree.map(lambda x: jnp.mean(x), combined_metrics)

                    combined_metrics["rollout/episode_return/mean"] = jnp.mean(rollout_episode_return_per_env)
                    combined_metrics["rollout/episode_length/mean"] = jnp.mean(rollout_episode_length_per_env)
                    combined_metrics["env_info/xy_vel_diff_abs/mean"] = jnp.mean(xy_vel_diff_abs_per_env)
                    combined_metrics["env_curriculum/coefficient/mean"] = jnp.mean(curriculum_coefficient_per_env)

                    def callback(carry):
                        metrics, learning_iteration_step, combined_learning_iteration_step, device_id = carry
                        if device_id == 0:
                            current_time = time.time()
                            metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (current_time - self.last_time))
                            self.last_time = current_time
                            global_step = combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs
                            metrics["steps/nr_env_steps"] = global_step
                            metrics["steps/nr_robot_type_steps"] = combined_learning_iteration_step.item() * self.nr_steps * self.nr_envs_per_train_robot
                            metrics["steps/nr_updates"] = combined_learning_iteration_step.item() * self.nr_epochs * self.nr_minibatches
                            self.start_logging(global_step)
                            for key, value in metrics.items():
                                self.log(f"{key}", np.asarray(value), global_step)
                            self.end_logging()

                    combined_learning_iteration_step = (multi_learning_iteration_step * self.nr_updates_per_multi_learning_iteration) + learning_iteration_step + 1
                    jax.debug.callback(callback, (combined_metrics, learning_iteration_step, combined_learning_iteration_step, device_id))

                    return (new_policy_state, prev_policy_state, new_critic_state, multi_env_state, embodiment_particles, embodiment_pending_particles, key), None

                key, subkey = jax.random.split(key)
                learning_iteration_carry, _ = jax.lax.scan(learning_iteration, (policy_state, prev_policy_state, critic_state, multi_env_state, embodiment_particles, embodiment_pending_particles, subkey), jnp.arange(self.nr_updates_per_multi_learning_iteration))
                (policy_state, prev_policy_state, critic_state, multi_env_state, embodiment_particles, embodiment_pending_particles, key) = learning_iteration_carry


                # Saving
                if self.save_model:
                    def save_with_check(policy_state, critic_state, embodiment_particles, device_id):
                        if device_id == 0:
                            self.save(policy_state, critic_state, embodiment_particles)
                    jax.debug.callback(save_with_check, policy_state, critic_state, embodiment_particles, device_id)


                return (policy_state, prev_policy_state, critic_state, multi_env_state, embodiment_particles, embodiment_pending_particles, key), None

            final_carry, _ = jax.lax.scan(multi_learning_and_eval_save_iteration, (policy_state, prev_policy_state, critic_state, multi_env_state, embodiment_particles, embodiment_pending_particles, key), jnp.arange(self.nr_multi_learning_and_eval_save_iterations))
            return final_carry


        train_function = jax.pmap(jitable_train_function, axis_name="i")

        policy_state = jax.device_put_replicated(self.policy_state, jax.local_devices())
        critic_state = jax.device_put_replicated(self.critic_state, jax.local_devices())
        embodiment_particles = jax.device_put_replicated(self.embodiment_particles, jax.local_devices())
        embodiment_pending_particles = jax.device_put_replicated(self.embodiment_pending_particles, jax.local_devices())

        self.key, train_key, reset_key = jax.random.split(self.key, 3)
        train_key = jax.random.split(train_key, self.nr_devices)
        reset_key = jax.random.split(reset_key, (self.nr_devices, self.nr_envs))
        multi_env_state = jax.pmap(self.train_env.train_reset, axis_name="i")(reset_key)

        self.last_time = time.time()
        self.start_time = deepcopy(self.last_time)
        final_carry = train_function(policy_state, critic_state, embodiment_particles, embodiment_pending_particles, multi_env_state, train_key, jnp.arange(self.nr_devices))

        rlx_logger.info(f"Average time: {(time.time() - self.start_time):.2f} s")

        unreplicate = lambda tree: jax.tree_util.tree_map(lambda x: x[0], tree)
        self.policy_state = unreplicate(final_carry[0])
        self.critic_state = unreplicate(final_carry[2])
        self.embodiment_particles = unreplicate(final_carry[4])
        self.embodiment_pending_particles = unreplicate(final_carry[5])

        os.makedirs(self.save_path, exist_ok=True)
        if self.config.algorithm.export_state_bank_at_end:
            self.export_state_bank()


    def log(self, name, value, step):
        if self.track_wandb:
            self.wandb_log_cache[name] = value
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)


    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_wandb:
            self.wandb_log_cache = {"global_step": int(step)}
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_wandb:
            wandb.log(self.wandb_log_cache)
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self, policy_state, critic_state, embodiment_particles):
        checkpoint = {
            "policy": policy_state,
            "critic": critic_state,
            "embodiment_particles": embodiment_particles,
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.latest_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{self.latest_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.latest_model_file_name}.zip", f"{self.save_path}/{self.latest_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.latest_model_file_name}", base_path=self.save_path)


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        unpacked_checkpoint_dir = tempfile.mkdtemp(dir=checkpoint_dir)
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", unpacked_checkpoint_dir, "zip")
        checkpoint_dir = unpacked_checkpoint_dir

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            # The settings of the test mode are not part of the trained model
            if key not in config.algorithm or key.startswith(("export_state_bank_path", "eval_design", "surrogate")):
                continue
            if f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        model = URMAv2ForCodesign(config, train_env, eval_env, run_path, writer)

        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        if config.runner.mode == "test":
            # The embodiment particles are not needed for testing and their shape depends on the training environment
            target = {"policy": model.policy_state, "critic": model.critic_state}
            restore_args = orbax_utils.restore_args_from_target(target)
            checkpoint = checkpointer.restore(checkpoint_dir, args=orbax.checkpoint.args.PyTreeRestore(item=target, restore_args=restore_args, partial_restore=True))
        else:
            target = {
                "policy": model.policy_state,
                "critic": model.critic_state,
                "embodiment_particles": model.embodiment_particles,
            }
            restore_args = orbax_utils.restore_args_from_target(target)
            checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)
            model.embodiment_particles = checkpoint["embodiment_particles"]
            model.embodiment_pending_particles = checkpoint["embodiment_particles"]
        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]

        shutil.rmtree(checkpoint_dir)

        return model


    def test(self, episodes):
        if self.config.algorithm.export_state_bank_path:
            self.export_state_bank(path=self.config.algorithm.export_state_bank_path)
        elif self.config.algorithm.surrogate_search_method:
            self.run_surrogate_search()
        elif self.config.algorithm.eval_design_enabled:
            self.evaluate_design()
        else:
            raise ValueError("Test mode needs one of algorithm.export_state_bank_path, algorithm.surrogate_search_method or algorithm.eval_design_enabled")


    def run_surrogate_search(self):
        self.key, subkey = jax.random.split(self.key)
        reset_keys = jax.random.split(subkey, self.nr_envs)
        base_env_state = self.train_env.train_reset(reset_keys)
        run_surrogate_search_urma(
            cfg=self.config.algorithm,
            save_path=self.save_path,
            critic=self.critic,
            critic_params=self.critic_state.params,
            train_env=self.train_env,
            base_env_state=base_env_state,
            decode_obs_fn=lambda obs: self.decode_train_obs(obs, 0),
        )


    def export_state_bank(self, target_size=4096, path=None):
        @jax.jit
        def policy_apply(observation):
            obs_r = observation.reshape(self.nr_train_robots, self.nr_envs_per_train_robot, -1)
            action_means_padded = []
            for i in range(self.nr_train_robots):
                obs_i = obs_r[i]
                (p_jd_i, p_jo_i, p_gs_i) = self.decode_policy_obs(obs_i, i)
                action_mean_i, _ = self.policy.apply(self.policy_state.params, p_jd_i, p_jo_i, p_gs_i)
                action_mean_padded_i = jnp.concatenate([action_mean_i, jnp.zeros((self.nr_envs_per_train_robot, self.missing_nr_of_actions_for_robots[i]))], axis=-1)
                action_means_padded.append(action_mean_padded_i)
            return jnp.concatenate(action_means_padded, axis=0)

        out_path = path or os.path.join(self.save_path, "surrogate.npz")
        self.key, _ = run_state_bank_export(
            train_env=self.train_env,
            nr_envs=self.nr_envs,
            obs_dim=self.os_shape[0],
            initial_key=self.key,
            policy_apply_fn=policy_apply,
            get_processed_action_fn=self.get_processed_action,
            out_path=out_path,
            embodiment_obs_indices=np.asarray(self.embodiment_observation_indices),
            target_size=target_size,
        )
        if self.track_wandb:
            wandb.save(out_path, base_path=self.save_path)


    def evaluate_design(self):
        def policy_apply_fn(policy_key, observation):
            obs_r = observation.reshape(self.nr_train_robots, self.nr_envs_per_train_robot, -1)
            policy_keys = jax.random.split(policy_key, self.nr_train_robots)
            actions_padded = []
            for i in range(self.nr_train_robots):
                obs_i = obs_r[i]
                (p_jd_i, p_jo_i, p_gs_i) = self.decode_policy_obs(obs_i, i)
                action_mean_i, action_logstd_i = self.policy.apply(self.policy_state.params, p_jd_i, p_jo_i, p_gs_i)
                action_std_i = jnp.exp(action_logstd_i)
                action_i = action_mean_i + action_std_i * jax.random.normal(policy_keys[i], shape=action_mean_i.shape)
                action_padded_i = jnp.concatenate([action_i, jnp.zeros((self.nr_envs_per_train_robot, self.missing_nr_of_actions_for_robots[i]))], axis=-1)
                actions_padded.append(action_padded_i)
            return jnp.concatenate(actions_padded, axis=0)
        self.key, _ = run_design_evaluation(
            cfg=self.config.algorithm,
            train_env=self.train_env,
            nr_envs=self.nr_envs,
            initial_key=self.key,
            policy_apply_fn=policy_apply_fn,
            get_processed_action_fn=self.get_processed_action,
        )


    def general_properties():
        return GeneralProperties
