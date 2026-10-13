import json
import os
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np

rlx_logger = logging.getLogger("rl_x")


def force_max_curriculum(multi_env_state):
    new_env_states = []
    for env_state in multi_env_state.env_states:
        internal_state = dict(env_state.internal_state)
        internal_state["env_curriculum_coeff"] = jnp.ones_like(env_state.internal_state["env_curriculum_coeff"])
        new_env_states.append(env_state.replace(internal_state=internal_state))
    return multi_env_state.replace(env_states=tuple(new_env_states))


def load_particle_normalized(path, expected_dim):
    if not path:
        return None
    particle = np.asarray(np.load(path)["particle"], dtype=np.float32).reshape(-1)
    if particle.shape[0] != expected_dim:
        raise ValueError(f"Particle in {path} has dim {particle.shape[0]}, expected {expected_dim}")
    return particle


def run_state_bank_export(
    train_env,
    nr_envs,
    obs_dim,
    initial_key,
    policy_apply_fn,
    get_processed_action_fn,
    out_path,
    embodiment_obs_indices,
    target_size=4096,
):
    rlx_logger.info(f"Collecting state bank (target_size={target_size})")
    n_steps = max(1, int(np.ceil(target_size / nr_envs)))
    key, subkey = jax.random.split(initial_key)
    reset_keys = jax.random.split(subkey, nr_envs)
    multi_env_state = train_env.train_reset(reset_keys)

    @jax.jit
    def step_and_extract(multi_env_state, key):
        observation = multi_env_state.next_observation
        action_mean = policy_apply_fn(observation)
        processed_action = get_processed_action_fn(action_mean)
        multi_env_state = train_env.step(multi_env_state, processed_action)
        return multi_env_state, multi_env_state.next_observation

    buf = np.zeros((n_steps * nr_envs, obs_dim), dtype=np.float32)
    for step in range(n_steps):
        multi_env_state, obs = step_and_extract(multi_env_state, key)
        obs_np = np.asarray(obs)
        buf[step * nr_envs : (step + 1) * nr_envs] = obs_np

    if buf.shape[0] > target_size:
        idx = np.random.default_rng(0).permutation(buf.shape[0])[:target_size]
        buf = buf[idx]
    state_bank = buf

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        state_bank=np.asarray(state_bank, dtype=np.float32),
        embodiment_obs_indices=np.asarray(embodiment_obs_indices, dtype=np.int32),
    )
    rlx_logger.info(f"Wrote state bank to {out_path} (shape={state_bank.shape})")
    return key, state_bank


def run_design_evaluation(
    cfg,
    train_env,
    nr_envs,
    initial_key,
    policy_apply_fn,
    get_processed_action_fn,
):
    rlx_logger.info("Running design evaluation")

    if not train_env.seen_robot_active:
        raise ValueError("run_design_evaluation requires env's seen_robot to be active")
    eff_dim = int(train_env.factor_dim())
    factor_normalized = load_particle_normalized(cfg.eval_design_particle_path, eff_dim)
    if factor_normalized is None:
        factor_normalized = np.zeros(eff_dim, dtype=np.float32)
        rlx_logger.info("No factor path provided, using nominal (zeros) in factor space")
    else:
        rlx_logger.info(f"Loaded factor from {cfg.eval_design_particle_path}; norm={np.linalg.norm(factor_normalized):.4f}")
    factors_batch = jnp.broadcast_to(jnp.asarray(factor_normalized)[None, :], (nr_envs, eff_dim))
    key, subkey = jax.random.split(initial_key)
    reset_keys = jax.random.split(subkey, nr_envs)
    multi_env_state = train_env.train_reset(reset_keys)
    multi_env_state = multi_env_state.replace(
        internal_state={**multi_env_state.internal_state, "in_eval_mode": jnp.ones_like(multi_env_state.internal_state["in_eval_mode"])}
    )
    multi_env_state = train_env.set_factor_params(multi_env_state, factors_batch)

    force_curriculum = bool(cfg.eval_design_force_max_curriculum)
    if force_curriculum:
        multi_env_state = force_max_curriculum(multi_env_state)

    key, subkey = jax.random.split(key)
    cmd_keys = jax.random.split(subkey, nr_envs)
    multi_env_state = train_env.sample_initial_command(multi_env_state, cmd_keys)

    watched_keys = [k for k in multi_env_state.info.keys() if "rollout/episode_return/" in k or "env_info/xy_vel_diff_abs/" in k]
    rlx_logger.info(f"Watching metric keys: {watched_keys}")

    done_keys = tuple(k for k in watched_keys if k.startswith("rollout/episode_return/") or k.startswith("rollout/episode_length/"))
    continuous_keys = tuple(k for k in watched_keys if k not in done_keys)

    total_steps = int(cfg.eval_design_total_steps)
    window_steps = min(int(cfg.eval_design_window_steps), total_steps)
    convergence_tol = float(cfg.eval_design_convergence_tol)

    key_owner_tasks = {}
    for k in watched_keys:
        key_owner_tasks[k] = [i for i, es in enumerate(multi_env_state.env_states) if k in es.info]

    @jax.jit
    def rollout_all(state, key):
        def step_fn(carry, _):
            state, key = carry
            key, policy_key = jax.random.split(key)
            observation = state.next_observation
            sampled_action = policy_apply_fn(policy_key, observation)
            processed_action = get_processed_action_fn(sampled_action)
            state = train_env.step(state, processed_action)
            if force_curriculum:
                state = force_max_curriculum(state)
            cont_means = {}
            done_sums = {}
            done_counts_per_key = {}
            for k in continuous_keys:
                owners = key_owner_tasks[k]
                vals = jnp.concatenate([state.env_states[i].info[k] for i in owners])
                cont_means[k] = jnp.mean(vals)
            for k in done_keys:
                owners = key_owner_tasks[k]
                vals = jnp.concatenate([state.env_states[i].info[k] for i in owners])
                dones = jnp.concatenate([(jnp.asarray(state.env_states[i].terminated) | jnp.asarray(state.env_states[i].truncated)).astype(jnp.float32) for i in owners])
                done_sums[k] = jnp.sum(vals * dones)
                done_counts_per_key[k] = jnp.sum(dones)
            return (state, key), (cont_means, done_sums, done_counts_per_key)

        (final_state, final_key), traj = jax.lax.scan(step_fn, (state, key), None, length=total_steps)
        return final_state, final_key, traj

    t0 = time.time()
    multi_env_state, key, (cont_traj, done_sum_traj, done_count_traj) = rollout_all(multi_env_state, key)
    jax.block_until_ready(cont_traj[continuous_keys[0]] if continuous_keys else done_count_traj[done_keys[0]])
    rollout_wall_s = time.time() - t0
    rlx_logger.info(f"Rollout complete: {total_steps} steps x {nr_envs} envs in {rollout_wall_s:.1f}s ({total_steps * nr_envs / rollout_wall_s / 1e6:.2f}M env-steps/s)")

    cont_traj_np = {k: np.asarray(v) for k, v in cont_traj.items()}
    done_sum_np = {k: np.asarray(v) for k, v in done_sum_traj.items()}
    done_count_np = {k: np.asarray(v) for k, v in done_count_traj.items()}

    metrics_summary = {}
    for k in continuous_keys:
        values = cont_traj_np[k][-window_steps:]
        final_mean = float(np.mean(values))
        iqr_low, iqr_high = float(np.percentile(values, 25)), float(np.percentile(values, 75))
        if len(cont_traj_np[k]) >= 2 * window_steps:
            prev_mean = float(np.mean(cont_traj_np[k][-2 * window_steps:-window_steps]))
            converged = abs(final_mean - prev_mean) < convergence_tol
        else:
            converged = False
        metrics_summary[k] = {
            "value": final_mean,
            "iqr": [iqr_low, iqr_high],
            "n_samples": int(len(values)),
            "converged": bool(converged),
        }
    for k in done_keys:
        sums = done_sum_np[k]
        counts = done_count_np[k]
        valid = counts > 0
        per_step_mean = np.where(valid, sums / np.maximum(counts, 1), np.nan)
        window_vals = per_step_mean[-window_steps:]
        window_vals = window_vals[~np.isnan(window_vals)]
        if window_vals.size == 0:
            final_mean = 0.0
            iqr_low = iqr_high = 0.0
            converged = False
        else:
            final_mean = float(np.mean(window_vals))
            iqr_low, iqr_high = float(np.percentile(window_vals, 25)), float(np.percentile(window_vals, 75))
            if per_step_mean.size >= 2 * window_steps:
                prev_window = per_step_mean[-2 * window_steps:-window_steps]
                prev_window = prev_window[~np.isnan(prev_window)]
                if prev_window.size > 0:
                    prev_mean = float(np.mean(prev_window))
                    converged = abs(final_mean - prev_mean) < convergence_tol
                else:
                    converged = False
            else:
                converged = False
        metrics_summary[k] = {
            "value": final_mean,
            "iqr": [iqr_low, iqr_high],
            "n_samples": int(window_vals.size),
            "converged": bool(converged),
        }
    summary_line = ", ".join(f"{k.rsplit('/', 1)[-1]}={v['value']:.4f}" for k, v in metrics_summary.items())
    rlx_logger.info(f"  final ({total_steps} steps): {summary_line}")

    result = {
        "particle": factor_normalized.tolist(),
        "force_max_curriculum": force_curriculum,
        "metrics": metrics_summary,
        "config": {
            "total_steps": total_steps,
            "window_steps": window_steps,
            "convergence_tol": convergence_tol,
            "nr_envs": int(nr_envs),
            "rollout_wall_s": rollout_wall_s,
        },
    }

    if cfg.eval_design_output_path:
        os.makedirs(os.path.dirname(os.path.abspath(cfg.eval_design_output_path)) or ".", exist_ok=True)
        with open(cfg.eval_design_output_path, "w") as f:
            json.dump(result, f, indent=2)
        rlx_logger.info(f"Wrote design-evaluation result to {cfg.eval_design_output_path}")
    else:
        rlx_logger.info(f"Design-evaluation result:\n{json.dumps(result, indent=2)}")

    return key, result
