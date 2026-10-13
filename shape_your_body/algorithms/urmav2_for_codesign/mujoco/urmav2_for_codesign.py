import os
import shutil
import tempfile
import json
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

from shape_your_body.algorithms.urmav2_for_codesign.mujoco.general_properties import GeneralProperties
from shape_your_body.algorithms.urmav2_for_codesign.mujoco.policy import get_policy
from shape_your_body.algorithms.urmav2_for_codesign.mujoco.critic import get_critic
from shape_your_body.algorithms.urmav2_for_codesign.mujoco.batch import Batch
from shape_your_body.algorithms.design_search.eval_utils import load_particle_normalized

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
        if config.runner.mode == "train" and len(config.environment.train_robots) != 1:
            raise ValueError("Training with MuJoCo only supports a single robot")

        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        if self.track_wandb:
            wandb.config["algorithm"]["batch_size"] = self.batch_size

        if config.runner.mode == "train":
            assert self.total_timesteps % self.batch_size == 0, "Total timesteps must be a multiple of batch size"
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        if self.track_wandb:
            wandb.config["algorithm"]["nr_updates"] = self.nr_updates

        if self.evaluation_and_save_frequency == -1:
            self.evaluation_and_save_frequency = self.batch_size * self.nr_updates
        if self.track_wandb:
            wandb.config["algorithm"]["evaluation_and_save_frequency"] = self.evaluation_and_save_frequency
        if config.runner.mode == "train":
            assert self.evaluation_and_save_frequency % self.batch_size == 0, "Evaluation and save frequency must be a multiple of batch size"
            assert self.total_timesteps % self.evaluation_and_save_frequency == 0, "Total timesteps must be a multiple of evaluation and save frequency"

        rlx_logger.info(f"Using: {jax.default_backend()}")

        self.minibatch_size_per_robot = self.minibatch_size
        if self.track_wandb:
            wandb.config["algorithm"]["minibatch_size_per_robot_per_device"] = self.minibatch_size_per_robot

        if config.runner.mode == "train":
            assert self.batch_size % self.minibatch_size == 0, "Batch size must be divisible by minibatch size"
        self.nr_minibatches = self.batch_size // self.minibatch_size
        if self.track_wandb:
            wandb.config["algorithm"]["nr_minibatches_per_epoch_per_device"] = self.nr_minibatches

        self.nr_minibatches_per_update = self.nr_minibatches * self.nr_epochs
        if self.track_wandb:
            wandb.config["algorithm"]["nr_minibatches_per_update"] = self.nr_minibatches_per_update

        self.nr_minibatches_during_training = self.nr_minibatches_per_update * self.nr_updates
        if self.track_wandb:
            wandb.config["algorithm"]["nr_minibatches_during_training"] = self.nr_minibatches_during_training


        # Environment specific values
        self.nr_envs_per_train_robot = self.nr_envs
        self.os_shape = self.train_env.single_observation_space.shape
        self.as_shape = self.train_env.single_action_space.shape

        self.joint_description_size = self.train_env.joint_description_size
        self.joint_observation_size = self.train_env.joint_observation_size
        self.foot_description_size = self.train_env.foot_description_size
        self.foot_observation_size = self.train_env.foot_observation_size
        self.nr_joints_for_robots = tuple(int(j) for j in self.train_env.call("nr_actuator_joints"))
        self.policy_general_idx_for_robots = tuple(self.train_env.call("policy_general_obs_idx"))
        self.nr_actuator_joints = self.train_env.nr_actuator_joints
        self.nr_feet = self.train_env.nr_feet
        self.policy_general_obs_idx = self.train_env.policy_general_obs_idx
        self.critic_general_obs_idx = self.train_env.critic_general_obs_idx
        self.softmax_temperature_min = config.algorithm.softmax_temperature_min


        # Setup
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

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

        state, _ = self.train_env.reset()

        if self.train_env.factor_dim() == 0:
            raise ValueError("The seen_robot domain randomization of the environment must be active")
        self.embodiment_ndim = int(self.train_env.factor_dim())

        self.embodiment_particles = np.zeros((self.nr_envs, self.embodiment_ndim), dtype=np.float32)
        self.train_env.set_factor_params(self.embodiment_particles)

        state_normalized = state.copy()

        self.policy, self.get_processed_action = get_policy(self.config)
        self.critic = get_critic(self.config)

        dummy_obs = state_normalized[0]
        (dp_jd, dp_jo, dp_gs, dc_jd, dc_jo, dc_fd, dc_fo, dc_gs) = self.decode_obs(dummy_obs)

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

        del state, state_normalized

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


    def decode_obs(self, obs):
        jds = self.joint_description_size
        jos = self.joint_observation_size
        fds = self.foot_description_size
        fos = self.foot_observation_size
        nj = self.nr_actuator_joints
        nf = self.nr_feet
        policy_joint_size = (jds + jos) * nj
        critic_joint_size = (jds + jos) * nj
        critic_foot_size = (fds + fos) * nf
        p_joints = obs[..., :policy_joint_size].reshape(obs.shape[:-1] + (nj, jds + jos))
        p_jd = p_joints[..., :jds]
        p_jo = p_joints[..., jds:]
        offset = policy_joint_size
        c_joints = obs[..., offset:offset + critic_joint_size].reshape(obs.shape[:-1] + (nj, jds + jos))
        c_jd = c_joints[..., :jds]
        c_jo = c_joints[..., jds:]
        offset += critic_joint_size
        c_feet = obs[..., offset:offset + critic_foot_size].reshape(obs.shape[:-1] + (nf, fds + fos))
        c_fd = c_feet[..., :fds]
        c_fo = c_feet[..., fds:]
        offset += critic_foot_size
        p_gs = obs[..., self.policy_general_obs_idx]
        c_gs = obs[..., self.critic_general_obs_idx]
        return (p_jd, p_jo, p_gs, c_jd, c_jo, c_fd, c_fo, c_gs)


    def decode_policy_obs(self, obs, i):
        jds = self.joint_description_size
        jos = self.joint_observation_size
        nj = self.nr_joints_for_robots[i]
        policy_joint_size = (jds + jos) * nj
        p_joints = obs[..., :policy_joint_size].reshape(obs.shape[:-1] + (nj, jds + jos))
        p_jd = p_joints[..., :jds]
        p_jo = p_joints[..., jds:]
        p_gs = obs[..., self.policy_general_idx_for_robots[i]]
        return (p_jd, p_jo, p_gs)


    def train(self):
        @jax.jit
        def get_action_and_values(policy_state, critic_state, state_normalized, key):
            key, subkey = jax.random.split(key)
            (p_jd, p_jo, p_gs, c_jd, c_jo, c_fd, c_fo, c_gs) = self.decode_obs(state_normalized)
            action_mean, action_logstd = self.policy.apply(policy_state.params, p_jd, p_jo, p_gs)
            action_std = jnp.exp(action_logstd)
            action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            log_prob = (-0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(axis=-1)
            processed_action = self.get_processed_action(action)
            values = self.critic.apply(critic_state.params, c_jd, c_jo, c_fd, c_fo, c_gs)

            return processed_action, action, values, log_prob, key


        @jax.jit
        def calculate_gae_advantages(critic_state, next_states_normalized, rewards, values, terminations):
            (_, _, _, c_jd, c_jo, c_fd, c_fo, c_gs) = self.decode_obs(next_states_normalized)
            next_values = self.critic.apply(critic_state.params, c_jd, c_jo, c_fd, c_fo, c_gs)

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


        @jax.jit
        def update(policy_state, prev_policy_state, critic_state, states_normalized, actions, advantages, returns, values, log_probs, key):
            def loss_fn(policy_params, critic_params, state_normalized_b, action_b, log_prob_b, returns_b, advantage_b):
                (p_jd, p_jo, p_gs, c_jd, c_jo, c_fd, c_fo, c_gs) = self.decode_obs(state_normalized_b)
                # Policy loss
                action_mean, action_logstd = self.policy.apply(policy_params, p_jd, p_jo, p_gs)
                action_std = jnp.exp(action_logstd)
                action_std_mean = action_std.mean()
                new_log_prob = (-0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd).sum(axis=-1)
                entropy = (action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(axis=-1)

                logratio = new_log_prob - log_prob_b
                ratio = jnp.exp(logratio)
                approx_kl_div = (ratio - 1) - logratio
                clip_fraction = jnp.float32((jnp.abs(ratio - 1) > self.clip_range))

                pg_loss1 = -advantage_b * ratio
                pg_loss2 = -advantage_b * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = jnp.maximum(pg_loss1, pg_loss2)

                # Critic loss
                new_values = self.critic.apply(critic_params, c_jd, c_jo, c_fd, c_fo, c_gs)
                critic_loss = 0.5 * (new_values - returns_b) ** 2

                # Combine losses
                loss = pg_loss - self.entropy_coef * entropy + self.critic_coef * critic_loss

                # Create metrics
                metrics = {
                    "loss/policy_gradient_loss": pg_loss,
                    "loss/critic_loss": critic_loss,
                    "loss/entropy_loss": entropy,
                    "policy_ratio/approx_kl": approx_kl_div,
                    "policy_ratio/clip_fraction": clip_fraction,
                    "policy/std_dev": action_std_mean,
                }

                return loss, (metrics)


            batch_states_normalized = states_normalized.reshape((-1,) + self.os_shape)
            batch_actions = actions.reshape((-1,) + self.as_shape)
            batch_log_probs = log_probs.reshape(-1)
            batch_returns = returns.reshape((-1, 1)) if self.mean_critic_returns else returns.reshape((-1, self.nr_critic_heads))
            batch_advantages = advantages.reshape(-1)

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: jax.tree.map(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

            key, subkey = jax.random.split(key)
            batch_indices = jnp.tile(jnp.arange(self.nr_steps * self.nr_envs_per_train_robot), (self.nr_epochs, 1))
            batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
            batch_indices = batch_indices.reshape((self.nr_minibatches_per_update, self.minibatch_size_per_robot))

            def minibatch_update(carry, minibatch_indices):
                policy_state, critic_state = carry

                minibatch_advantages = batch_advantages[minibatch_indices]
                minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

                (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
                    policy_state.params,
                    critic_state.params,
                    batch_states_normalized[minibatch_indices],
                    batch_actions[minibatch_indices],
                    batch_log_probs[minibatch_indices],
                    batch_returns[minibatch_indices],
                    minibatch_advantages
                )

                policy_state = policy_state.apply_gradients(grads=policy_gradients)
                critic_state = critic_state.apply_gradients(grads=critic_gradients)

                metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
                metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                carry = (policy_state, critic_state)

                return carry, (metrics)

            init_carry = (policy_state, critic_state)
            carry, (optimization_metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
            new_policy_state, new_critic_state = carry
            new_policy_state = jax.lax.cond(
                (jnp.mean(optimization_metrics["policy_ratio/approx_kl"]) > self.kl_threshold) | (jnp.mean(optimization_metrics["policy_ratio/clip_fraction"]) == 0.0),
                lambda x: prev_policy_state,
                lambda x: new_policy_state,
                None
            )
            prev_policy_state = policy_state

            mean_metrics = {key: jnp.mean(optimization_metrics[key]) for key in optimization_metrics}
            mean_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
            mean_metrics["policy/softmax_temperature"] = jnp.exp(new_policy_state.params["params"]["joint_log_softmax_temperature"]).squeeze() + self.softmax_temperature_min

            return new_policy_state, prev_policy_state, new_critic_state, mean_metrics, key


        batch = Batch(
            states_normalized=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            next_states_normalized=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            actions=np.zeros((self.nr_steps, self.nr_envs) + self.as_shape),
            rewards=np.zeros((self.nr_steps, self.nr_envs)),
            values=np.zeros((self.nr_steps, self.nr_envs, self.nr_critic_heads)),
            terminations=np.zeros((self.nr_steps, self.nr_envs), dtype=bool),
            log_probs=np.zeros((self.nr_steps, self.nr_envs)),
            advantages=np.zeros((self.nr_steps, self.nr_envs)),
            returns=np.zeros((self.nr_steps, self.nr_envs, self.nr_critic_heads)),
        )

        state, _ = self.train_env.reset()
        state_normalized = state.copy()
        prev_policy_state = self.policy_state

        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        prev_saving_end_time = None
        logging_time_prev = None

        rng = np.random.default_rng(self.seed + 1)

        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}
            if logging_time_prev:
                time_metrics["time/logging_time_prev"] = logging_time_prev


            # Acting
            dones_this_rollout = 0
            step_info_collection = {}
            for step in range(self.nr_steps):
                processed_action, action, values, log_prob, self.key = get_action_and_values(self.policy_state, self.critic_state, state_normalized, self.key)
                next_state, reward, terminated, truncated, info = self.train_env.step(jax.device_get(processed_action))
                done = terminated | truncated
                actual_next_state = next_state.copy()
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = np.array(self.train_env.get_final_observation_at_index(info, i))
                        dones_this_rollout += 1
                for key, info_value in self.train_env.get_logging_info_dict(info).items():
                    step_info_collection.setdefault(key, []).extend(info_value)

                next_state_normalized = next_state.copy()
                actual_next_state_normalized = actual_next_state.copy()

                batch.states_normalized[step] = state_normalized
                batch.next_states_normalized[step] = actual_next_state_normalized
                batch.actions[step] = action
                batch.rewards[step] = reward
                batch.values[step] = values
                batch.terminations[step] = terminated
                batch.log_probs[step] = log_prob
                state_normalized = next_state_normalized
                global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # Calculating advantages and returns
            batch.advantages, batch.returns = calculate_gae_advantages(self.critic_state, batch.next_states_normalized, batch.rewards, batch.values, batch.terminations)

            calc_adv_return_end_time = time.time()
            time_metrics["time/calc_adv_and_return_time"] = calc_adv_return_end_time - acting_end_time


            # Optimizing
            self.policy_state, prev_policy_state, self.critic_state, optimization_metrics, self.key = update(
                self.policy_state, prev_policy_state, self.critic_state,
                batch.states_normalized, batch.actions, batch.advantages, batch.returns, batch.values, batch.log_probs,
                self.key
            )
            optimization_metrics = {key: value.item() for key, value in optimization_metrics.items()}
            nr_updates += self.nr_epochs * self.nr_minibatches

            # Resampling of the embodiment particles
            curriculum_coefficients = np.asarray(self.train_env.get_curriculum_coefficients(), dtype=np.float32)[:, None]
            local_resample_mask = rng.random(self.nr_envs) < self.embodiment_local_resample_probability
            local_perturbation = rng.uniform(-1.0, 1.0, size=(self.nr_envs, self.embodiment_ndim)).astype(np.float32)
            local_range = self.embodiment_local_resample_range_fraction * curriculum_coefficients
            locally_resampled_particles = np.clip(self.embodiment_particles + local_perturbation * local_range, -curriculum_coefficients, curriculum_coefficients)
            self.embodiment_particles = np.where(local_resample_mask[:, None], locally_resampled_particles, self.embodiment_particles).astype(np.float32)

            self.train_env.set_factor_params(self.embodiment_particles)

            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_adv_return_end_time


            # Saving
            if self.save_model and global_step % self.evaluation_and_save_frequency == 0:
                self.save(self.policy_state, self.critic_state, self.embodiment_particles)

            saving_end_time = time.time()
            if prev_saving_end_time:
                time_metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics["time/saving_time"] = saving_end_time - optimizing_end_time


            # Logging
            self.start_logging(global_step)

            steps_metrics["steps/nr_env_steps"] = global_step
            steps_metrics["steps/nr_updates"] = nr_updates
            steps_metrics["steps/nr_episodes"] = nr_episodes

            rollout_info_metrics = {}
            env_info_metrics = {}
            if step_info_collection:
                info_names = list(step_info_collection.keys())
                for info_name in info_names:
                    metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                    metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                    mean_value = np.mean(step_info_collection[info_name])
                    if mean_value == mean_value:  # Check if mean_value is NaN
                        metric_dict[f"{metric_group}/{info_name}"] = mean_value

            combined_metrics = {**rollout_info_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
            for key, value in combined_metrics.items():
                self.log(f"{key}", value, global_step)

            self.end_logging()

            logging_end_time = time.time()
            logging_time_prev = logging_end_time - saving_end_time


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
            if key not in config.algorithm or key.startswith("eval_design"):
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
            model.embodiment_particles = np.asarray(checkpoint["embodiment_particles"], dtype=np.float32)
        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]

        shutil.rmtree(checkpoint_dir)

        return model


    def test(self, episodes):
        @jax.jit
        def get_action(policy_joint_descriptions, policy_joint_observations, policy_general_state):
            action_mean, action_logstd = self.policy.apply(self.policy_state.params, policy_joint_descriptions, policy_joint_observations, policy_general_state)
            action = action_mean
            processed_action = self.get_processed_action(action)

            return processed_action


        factors = np.zeros((self.nr_envs, self.embodiment_ndim), dtype=np.float32)
        if self.config.algorithm.eval_design_particle_path:
            factors[:] = load_particle_normalized(self.config.algorithm.eval_design_particle_path, self.embodiment_ndim)
        self.train_env.set_factor_params(factors)

        self.train_env.activate_single_env_mode()

        state, _ = self.train_env.reset()
        i = 0
        while True:
            done = False
            episode_return = 0
            while not done:
                (p_jd, p_jo, p_gs) = self.decode_policy_obs(state, self.train_env.env_to_render)
                processed_action = get_action(p_jd, p_jo, p_gs)
                state, reward, terminated, truncated, info = self.train_env.step(jax.device_get(processed_action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
            i += 1


    def general_properties():
        return GeneralProperties
