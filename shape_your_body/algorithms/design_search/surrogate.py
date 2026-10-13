import json
import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

rlx_logger = logging.getLogger("rl_x")


def run_random_search(fitness_fn, dim, n_evals, seed=0, curriculum_coeff=1.0):
    rng = np.random.default_rng(seed)
    bound = float(curriculum_coeff)
    best_x = None
    best_y = -np.inf
    trace = []
    for i in range(n_evals):
        x = rng.uniform(low=-bound, high=bound, size=dim).astype(np.float32)
        y = fitness_fn(x)
        if y > best_y:
            best_y = y
            best_x = x
        trace.append({"step": i, "fitness": float(y), "best_fitness": float(best_y)})
    return {"best_particle": best_x, "best_fitness": float(best_y), "trace": trace}


def run_cma_es(fitness_fn, dim, n_iter, popsize=None, sigma0=0.5, seed=0, curriculum_coeff=1.0, uniform_init_seed=-1):
    import cma
    bound = float(curriculum_coeff)
    if dim < 2:
        rlx_logger.warning(f"CMA-ES is degenerate at dim={dim}; falling back to random search.")
        return run_random_search(fitness_fn, dim, n_iter * 4, seed=seed, curriculum_coeff=bound)
    if popsize is None:
        popsize = max(4 + int(3 * np.log(dim)), 4)
    options = {
        "bounds": [[-bound] * dim, [bound] * dim],
        "popsize": popsize,
        "seed": int(seed) + 1,
        "verbose": -9,
    }
    if dim > 100:
        options["CMA_diagonal"] = True
    x0, _ = make_init_particle(dim, uniform_init_seed, bound)
    es = cma.CMAEvolutionStrategy(np.asarray(x0, dtype=np.float64), sigma0 * bound, options)
    trace = []
    best_y = -np.inf
    eval_step = 0
    iters = 0
    while not es.stop() and iters < n_iter:
        candidates = es.ask()
        fitnesses = [fitness_fn(np.asarray(c, dtype=np.float32)) for c in candidates]
        es.tell(candidates, [-f for f in fitnesses])
        iter_mean = float(np.mean(fitnesses))
        iter_best = float(np.max(fitnesses))
        if iter_best > best_y:
            best_y = iter_best
        for f in fitnesses:
            best_y = max(best_y, float(f))
            trace.append({
                "step": eval_step,
                "iter": iters,
                "fitness": float(f),
                "best_fitness": float(best_y),
                "iter_mean_fitness": iter_mean,
                "iter_best_fitness": iter_best,
                "phase": "cma",
            })
            eval_step += 1
        iters += 1
    best_x = np.asarray(es.best.x, dtype=np.float32)
    best_y = float(-es.best.f) if es.best.f is not None else best_y
    return {"best_particle": best_x, "best_fitness": best_y, "trace": trace}


def run_bo(fitness_fn, dim, n_iter, n_init=10, seed=0, curriculum_coeff=1.0, uniform_init_seed=-1):
    bound = float(curriculum_coeff)
    if dim < 2:
        rlx_logger.warning(f"BO is degenerate at dim={dim}; falling back to random search (BoTorch's GP/acquisition optimisation is numerically unstable below dim=2).")
        return run_random_search(fitness_fn, dim, n_iter * 4 + n_init, seed=seed, curriculum_coeff=bound)

    import warnings
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Normalize
    from botorch.models.transforms.outcome import Standardize
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.utils.sampling import draw_sobol_samples
    from botorch.fit import fit_gpytorch_mll
    from botorch.exceptions.warnings import BotorchWarning
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from linear_operator.utils.warnings import NumericalWarning

    torch.manual_seed(seed)
    bounds = torch.tensor([[-bound] * dim, [bound] * dim], dtype=torch.double)

    X_init = draw_sobol_samples(bounds=bounds, n=n_init, q=1, seed=seed).squeeze(-2)
    if int(uniform_init_seed) >= 0:
        u_x, _ = make_init_particle(dim, uniform_init_seed, bound)
        X_init[0] = torch.tensor(u_x, dtype=X_init.dtype)
    Y_init = torch.tensor([[fitness_fn(x.numpy())] for x in X_init], dtype=torch.double)

    X = X_init.clone()
    Y = Y_init.clone()
    trace = []
    best_y = -np.inf
    for i in range(int(X.shape[0])):
        y = float(Y[i, 0])
        best_y = max(best_y, y)
        trace.append({"step": i, "fitness": y, "best_fitness": best_y, "phase": "init"})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=BotorchWarning)
        warnings.simplefilter("ignore", category=NumericalWarning)
        warnings.filterwarnings("ignore", message=".*Optimization failed.*", category=RuntimeWarning)
        for it in range(n_iter):
            try:
                gp = SingleTaskGP(
                    X, Y,
                    input_transform=Normalize(d=dim, bounds=bounds),
                    outcome_transform=Standardize(m=1),
                )
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)
                acq = qLogNoisyExpectedImprovement(gp, X_baseline=X)
                cand, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=8, raw_samples=256)
            except Exception as exc:
                rlx_logger.warning(f"BO step {it} fell back to random due to: {exc}")
                cand = torch.tensor(np.random.uniform(-bound, bound, size=(1, dim)), dtype=torch.double)
            y = float(fitness_fn(cand.squeeze(0).numpy()))
            X = torch.cat([X, cand], dim=0)
            Y = torch.cat([Y, torch.tensor([[y]], dtype=torch.double)], dim=0)
            best_y = max(best_y, y)
            trace.append({"step": int(X.shape[0]) - 1, "fitness": y, "best_fitness": best_y, "phase": "bo"})

    best_idx = int(torch.argmax(Y).item())
    best_x = X[best_idx].numpy().astype(np.float32)
    best_y = float(Y[best_idx, 0].item())
    return {"best_particle": best_x, "best_fitness": best_y, "trace": trace}


def make_init_particle(dim, uniform_init_seed, curriculum_coeff):
    if int(uniform_init_seed) >= 0:
        rng = np.random.default_rng(int(uniform_init_seed))
        x = rng.uniform(low=-float(curriculum_coeff), high=float(curriculum_coeff), size=int(dim)).astype(np.float32)
        return x, "uniform"
    return np.zeros(int(dim), dtype=np.float32), "zero"


def run_nominal(fitness_fn, dim, uniform_init_seed=-1, curriculum_coeff=1.0):
    x, source = make_init_particle(dim, uniform_init_seed, curriculum_coeff)
    y = float(fitness_fn(x))
    trace = [{"step": 0, "fitness": y, "best_fitness": y, "phase": f"nominal[{source}]"}]
    return {"best_particle": x, "best_fitness": y, "trace": trace}


def run_pso_urma(fitness_batch_fn, dim, n_iter, n_particles, seed=0, curriculum_coeff=1.0, uniform_init_seed=-1, c1=0.5, c2=0.3, w=0.9):
    bound = float(curriculum_coeff)
    K = max(2, int(n_particles))
    lower = jnp.full((dim,), -bound, dtype=jnp.float32)
    upper = jnp.full((dim,), bound, dtype=jnp.float32)
    span = upper - lower

    key, position_key, velocity_key = jax.random.split(jax.random.PRNGKey(int(seed)), 3)
    positions = lower + jax.random.uniform(position_key, (K, dim)) * span
    velocities = (jax.random.uniform(velocity_key, (K, dim)) * 2.0 - 1.0) * span
    fitness = fitness_batch_fn(positions)
    personal_best_positions = positions
    personal_best_fitness = fitness
    global_best_idx = jnp.argmax(fitness)
    global_best_position = positions[global_best_idx]
    global_best_fitness = fitness[global_best_idx]

    def pso_iteration(carry, _):
        positions, velocities, personal_best_positions, personal_best_fitness, global_best_position, global_best_fitness, key = carry
        key, r1_key, r2_key = jax.random.split(key, 3)
        r1 = jax.random.uniform(r1_key, (K, dim))
        r2 = jax.random.uniform(r2_key, (K, dim))
        velocities = w * velocities + c1 * r1 * (personal_best_positions - positions) + c2 * r2 * (global_best_position[None, :] - positions)
        positions = jnp.clip(positions + velocities, lower, upper)
        fitness = fitness_batch_fn(positions)
        improved = fitness > personal_best_fitness
        personal_best_positions = jnp.where(improved[:, None], positions, personal_best_positions)
        personal_best_fitness = jnp.where(improved, fitness, personal_best_fitness)
        global_best_idx = jnp.argmax(personal_best_fitness)
        global_best_position = personal_best_positions[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        carry = (positions, velocities, personal_best_positions, personal_best_fitness, global_best_position, global_best_fitness, key)
        return carry, None

    init_carry = (positions, velocities, personal_best_positions, personal_best_fitness, global_best_position, global_best_fitness, key)
    final_carry, _ = jax.lax.scan(pso_iteration, init_carry, None, length=int(n_iter))
    best_x = np.asarray(final_carry[4], dtype=np.float32)
    best_y = float(final_carry[5])
    if int(uniform_init_seed) >= 0:
        u_x, _ = make_init_particle(dim, uniform_init_seed, bound)
        u_y = float(fitness_batch_fn(jnp.asarray(u_x[None, :]))[0])
        if u_y > best_y:
            best_x, best_y = u_x, u_y
    trace = [{"step": 0, "phase": "pso_final", "best_fitness": best_y}]
    return {"best_particle": best_x, "best_fitness": best_y, "trace": trace}


def run_cem(fitness_batch_fn, dim, n_iter, n_particles, elite_frac=0.2, init_sigma=0.5, sigma_floor=0.02, seed=0, curriculum_coeff=1.0, uniform_init_seed=-1):
    bound = float(curriculum_coeff)
    K = max(4, int(n_particles))
    n_elite = max(2, int(round(elite_frac * K)))
    rng = np.random.default_rng(int(seed))
    if int(uniform_init_seed) >= 0:
        mean, _ = make_init_particle(dim, uniform_init_seed, bound)
        mean = np.asarray(mean, dtype=np.float32)
    else:
        mean = np.zeros(dim, dtype=np.float32)
    sigma = float(init_sigma) * np.ones(dim, dtype=np.float32)

    best_x = mean.copy()
    best_y = float(fitness_batch_fn(jnp.asarray(mean[None, :]))[0])
    trace = [{"step": 0, "phase": "init", "best_fitness": best_y}]

    for it in range(int(n_iter)):
        samples = rng.normal(0.0, 1.0, size=(K, dim)).astype(np.float32) * sigma[None, :] + mean[None, :]
        samples = np.clip(samples, -bound, bound)
        ys = np.asarray(fitness_batch_fn(jnp.asarray(samples)))
        elite_idx = np.argsort(ys)[-n_elite:]
        elites = samples[elite_idx]
        mean = elites.mean(axis=0).astype(np.float32)
        sigma = (elites.std(axis=0) + float(sigma_floor)).astype(np.float32)
        gen_best_idx = int(np.argmax(ys))
        if float(ys[gen_best_idx]) > best_y:
            best_y = float(ys[gen_best_idx])
            best_x = samples[gen_best_idx].astype(np.float32)
        trace.append({"step": it + 1, "phase": "cem", "gen_best": float(ys[gen_best_idx]), "best_fitness": best_y, "sigma_mean": float(sigma.mean())})
    return {"best_particle": best_x, "best_fitness": float(best_y), "trace": trace}


def run_de(fitness_batch_fn, dim, n_iter, n_particles, F=0.5, CR=0.9, seed=0, curriculum_coeff=1.0, uniform_init_seed=-1):
    bound = float(curriculum_coeff)
    K = max(4, int(n_particles))
    rng = np.random.default_rng(int(seed))
    population = rng.uniform(-bound, bound, size=(K, dim)).astype(np.float32)
    if int(uniform_init_seed) >= 0:
        u_x, _ = make_init_particle(dim, uniform_init_seed, bound)
        population[0] = u_x.astype(np.float32)
    fitnesses = np.asarray(fitness_batch_fn(jnp.asarray(population)))
    best_idx = int(np.argmax(fitnesses))
    best_x = population[best_idx].copy()
    best_y = float(fitnesses[best_idx])
    trace = [{"step": 0, "phase": "init", "best_fitness": best_y}]

    for it in range(int(n_iter)):
        a_idx = np.empty(K, dtype=np.int32)
        b_idx = np.empty(K, dtype=np.int32)
        c_idx = np.empty(K, dtype=np.int32)
        for i in range(K):
            choices = list(range(K)); choices.remove(i)
            a_idx[i], b_idx[i], c_idx[i] = rng.choice(choices, size=3, replace=False)
        a, b, c = population[a_idx], population[b_idx], population[c_idx]
        mutants = a + float(F) * (b - c)
        mutants = np.clip(mutants, -bound, bound)
        cross_mask = rng.uniform(0, 1, size=(K, dim)) < float(CR)
        forced = rng.integers(0, dim, size=(K,))
        cross_mask[np.arange(K), forced] = True
        trials = np.where(cross_mask, mutants, population)
        trial_fits = np.asarray(fitness_batch_fn(jnp.asarray(trials.astype(np.float32))))
        improved = trial_fits > fitnesses
        population = np.where(improved[:, None], trials, population).astype(np.float32)
        fitnesses = np.where(improved, trial_fits, fitnesses)
        gen_best_idx = int(np.argmax(fitnesses))
        if float(fitnesses[gen_best_idx]) > best_y:
            best_y = float(fitnesses[gen_best_idx])
            best_x = population[gen_best_idx].copy()
        trace.append({"step": it + 1, "phase": "de", "gen_best": float(fitnesses[gen_best_idx]), "best_fitness": best_y, "n_improved": int(improved.sum())})
    return {"best_particle": best_x.astype(np.float32), "best_fitness": float(best_y), "trace": trace}


def run_ars(fitness_batch_fn, dim, n_iter, n_directions, noise_std=0.05, lr=0.02, top_b_frac=1.0, seed=0, curriculum_coeff=1.0, uniform_init_seed=-1):
    bound = float(curriculum_coeff)
    N = max(2, int(n_directions))
    rng = np.random.default_rng(int(seed))
    if int(uniform_init_seed) >= 0:
        x, _ = make_init_particle(dim, uniform_init_seed, bound)
        x = x.astype(np.float32)
    else:
        x = np.zeros(dim, dtype=np.float32)
    best_x = x.copy()
    best_y = float(fitness_batch_fn(jnp.asarray(x[None, :]))[0])
    trace = [{"step": 0, "phase": "init", "best_fitness": best_y}]

    top_b = max(1, int(round(float(top_b_frac) * N)))
    for it in range(int(n_iter)):
        noise = rng.standard_normal(size=(N, dim)).astype(np.float32)
        x_plus = np.clip(x[None, :] + float(noise_std) * noise, -bound, bound).astype(np.float32)
        x_minus = np.clip(x[None, :] - float(noise_std) * noise, -bound, bound).astype(np.float32)
        f_plus = np.asarray(fitness_batch_fn(jnp.asarray(x_plus)))
        f_minus = np.asarray(fitness_batch_fn(jnp.asarray(x_minus)))
        scores = np.abs(f_plus - f_minus)
        if top_b < N:
            keep_idx = np.argsort(scores)[-top_b:]
        else:
            keep_idx = np.arange(N)
        f_p = f_plus[keep_idx]
        f_m = f_minus[keep_idx]
        n_k = noise[keep_idx]
        all_rs = np.concatenate([f_p, f_m])
        scale = max(float(all_rs.std()), 1e-8)
        grad = ((f_p - f_m)[:, None] * n_k).sum(axis=0) / (top_b * scale)
        x = np.clip(x + float(lr) * grad, -bound, bound).astype(np.float32)
        y_now = float(fitness_batch_fn(jnp.asarray(x[None, :]))[0])
        if y_now > best_y:
            best_y = y_now
            best_x = x.copy()
        trace.append({"step": it + 1, "phase": "ars", "fitness": y_now, "best_fitness": best_y, "scale": scale})
    return {"best_particle": best_x.astype(np.float32), "best_fitness": float(best_y), "trace": trace}


def run_vgds_urma(fitness_fn, critic_apply_decoded, critic_params, decode_obs_fn, get_embodiment_patch_fn, base_env_state, state_bank, emb_idx, dim, n_steps, minibatch_size, learning_rate, max_step, seed=0, curriculum_coeff=1.0, l2_prior_weight=0.0, uniform_init_seed=-1, l2_anchor_mode="zero"):
    state_bank_jax = jnp.asarray(state_bank, dtype=jnp.float32)
    emb_idx_jax = jnp.asarray(emb_idx, dtype=jnp.int32)
    bank_size = int(state_bank_jax.shape[0])
    dim = int(dim)
    minibatch_size = max(1, min(int(minibatch_size), bank_size))
    bound = float(curriculum_coeff)

    l2_anchor_mode = str(l2_anchor_mode).lower()
    if l2_anchor_mode not in ("zero", "perturbed_start"):
        raise ValueError(f"l2_anchor_mode must be 'zero'|'perturbed_start', got {l2_anchor_mode!r}")

    tx = optax.adam(learning_rate=learning_rate)
    u_x, _ = make_init_particle(dim, uniform_init_seed, bound)
    particle = jnp.asarray(u_x)
    opt_state = tx.init(particle)

    l2_anchor = jnp.zeros(dim, dtype=jnp.float32) if l2_anchor_mode == "zero" else jnp.asarray(particle)

    def composite_fitness(p, minibatch_states):
        particles_broadcast = jnp.broadcast_to(p[None, :], (1, p.shape[0]))
        description_patch = get_embodiment_patch_fn(base_env_state, particles_broadcast)
        description_patch_single = description_patch[0]
        states = minibatch_states.at[:, emb_idx_jax].set(description_patch_single[None, :])
        decoded = decode_obs_fn(states)
        c_jd, c_jo, c_fd, c_fo, c_gs = decoded[3], decoded[4], decoded[5], decoded[6], decoded[7]
        values = critic_apply_decoded(critic_params, c_jd, c_jo, c_fd, c_fo, c_gs)
        critic_term = values.mean(axis=-1).mean()
        l2_term = jnp.mean((p - l2_anchor) ** 2)
        return critic_term - l2_prior_weight * l2_term

    @jax.jit
    def step_fn(particle, opt_state, minibatch_states):
        grad = jax.grad(lambda p: -composite_fitness(p, minibatch_states))(particle)
        updates, opt_state = tx.update(grad, opt_state, particle)
        updates = jnp.clip(updates, -max_step, max_step)
        particle = optax.apply_updates(particle, updates)
        particle = jnp.clip(particle, -bound, bound)
        return particle, opt_state

    composite_fitness_jit = jax.jit(composite_fitness)

    rng = np.random.default_rng(seed)
    full_bank = state_bank_jax
    y0_composite = float(composite_fitness_jit(particle, full_bank))
    y0_raw = float(fitness_fn(np.asarray(particle)))
    best_x = np.asarray(particle, dtype=np.float32)
    best_y = y0_composite
    trace = [{"step": 0, "fitness": y0_composite, "fitness_raw": y0_raw, "best_fitness": best_y, "phase": "init"}]
    for i in range(int(n_steps)):
        idx = rng.choice(bank_size, size=minibatch_size, replace=False)
        batch = state_bank_jax[idx]
        particle, opt_state = step_fn(particle, opt_state, batch)
        y_composite = float(composite_fitness_jit(particle, full_bank))
        y_raw = float(fitness_fn(np.asarray(particle)))
        if y_composite > best_y:
            best_y = y_composite
            best_x = np.asarray(particle, dtype=np.float32)
        trace.append({"step": i + 1, "fitness": y_composite, "fitness_raw": y_raw, "best_fitness": best_y, "phase": "vgds"})
    return {"best_particle": best_x, "best_fitness": best_y, "trace": trace}


def run_turbo_urma(critic_apply_decoded, critic_params, decode_obs_fn, get_embodiment_patch_fn, base_env_state, state_bank, emb_idx, dim, n_iter, n_candidates, init_radius, min_radius, max_radius, shrink_factor, expand_factor, seed=0, curriculum_coeff=1.0, l2_prior_weight=0.0, uniform_init_seed=-1):
    state_bank_jax = jnp.asarray(state_bank, dtype=jnp.float32)
    emb_idx_jax = jnp.asarray(emb_idx, dtype=jnp.int32)
    dim = int(dim)
    bound = float(curriculum_coeff)

    @jax.jit
    def score_one(p, minibatch_states):
        particles_broadcast = jnp.broadcast_to(p[None, :], (1, p.shape[0]))
        description_patch = get_embodiment_patch_fn(base_env_state, particles_broadcast)
        description_patch_single = description_patch[0]
        states = minibatch_states.at[:, emb_idx_jax].set(description_patch_single[None, :])
        decoded = decode_obs_fn(states)
        c_jd, c_jo, c_fd, c_fo, c_gs = decoded[3], decoded[4], decoded[5], decoded[6], decoded[7]
        values = critic_apply_decoded(critic_params, c_jd, c_jo, c_fd, c_fo, c_gs)
        critic_term = values.mean(axis=-1).mean()
        l2_term = jnp.mean(p ** 2)
        return critic_term - l2_prior_weight * l2_term

    rng = np.random.default_rng(seed)
    init_x, init_source = make_init_particle(dim, uniform_init_seed, bound)
    center = jnp.asarray(init_x, dtype=jnp.float32)
    radius = float(init_radius)
    full_bank = state_bank_jax
    best_x = np.asarray(init_x, dtype=np.float32)
    best_y = float(score_one(center, full_bank))
    trace = [{"step": 0, "fitness": best_y, "best_fitness": best_y, "radius": radius, "phase": "init", "init_source": init_source}]
    rlx_logger.info(f"[turbo] init: best_y={best_y:.4f}, radius={radius:.4f}, n_iter={n_iter}, n_candidates={n_candidates}, init_source={init_source}")

    for it in range(int(n_iter)):
        center_np = np.asarray(best_x, dtype=np.float32)
        candidates = center_np[None, :] + rng.uniform(low=-radius, high=radius, size=(int(n_candidates), dim)).astype(np.float32)
        candidates = np.clip(candidates, -bound, bound)
        ys = np.empty(candidates.shape[0], dtype=np.float32)
        for k in range(candidates.shape[0]):
            ys[k] = float(score_one(jnp.asarray(candidates[k]), full_bank))
        k_best = int(np.argmax(ys))
        y_best = float(ys[k_best])
        improved = y_best > best_y
        if improved:
            best_y = y_best
            best_x = candidates[k_best]
            radius = min(radius * float(expand_factor), float(max_radius))
        else:
            radius = max(radius * float(shrink_factor), float(min_radius))
        for k in range(candidates.shape[0]):
            trace.append({"step": int(len(trace)), "iter": it, "fitness": float(ys[k]), "best_fitness": best_y, "radius": radius, "phase": "turbo"})
        rlx_logger.info(f"[turbo] iter {it}: pop_max={y_best:.4f}, best={best_y:.4f}, radius={radius:.4f}, improved={improved}")
    return {"best_particle": best_x.astype(np.float32), "best_fitness": float(best_y), "trace": trace}


def run_gcpfo_urma(critic_apply_decoded, critic_params, decode_obs_fn, get_embodiment_patch_fn, base_env_state, state_bank, emb_idx, dim, n_iter, minibatch_size, n_regions, particles_per_region, step_size, temperature_init, eps, seed=0, curriculum_coeff=1.0, l2_prior_weight=0.0, uniform_init_seed=-1):
    state_bank_jax = jnp.asarray(state_bank, dtype=jnp.float32)
    emb_idx_jax = jnp.asarray(emb_idx, dtype=jnp.int32)
    bank_size = int(state_bank_jax.shape[0])
    dim = int(dim)
    minibatch_size = max(1, min(int(minibatch_size), bank_size))
    bound = float(curriculum_coeff)
    P = max(2, int(particles_per_region))
    R = max(1, int(n_regions))
    sigma = float(step_size)

    def loss_fn(p, minibatch_states):
        particles_broadcast = jnp.broadcast_to(p[None, :], (1, p.shape[0]))
        description_patch = get_embodiment_patch_fn(base_env_state, particles_broadcast)
        description_patch_single = description_patch[0]
        states = minibatch_states.at[:, emb_idx_jax].set(description_patch_single[None, :])
        decoded = decode_obs_fn(states)
        c_jd, c_jo, c_fd, c_fo, c_gs = decoded[3], decoded[4], decoded[5], decoded[6], decoded[7]
        values = critic_apply_decoded(critic_params, c_jd, c_jo, c_fd, c_fo, c_gs)
        critic_term = values.mean(axis=-1).mean()
        l2_term = jnp.mean(p ** 2)
        return -(critic_term - l2_prior_weight * l2_term)

    loss_and_grad = jax.jit(jax.vmap(jax.value_and_grad(loss_fn), in_axes=(0, None)))

    rng = np.random.default_rng(seed)
    init_rng = np.random.default_rng(int(seed) + 1)
    if int(uniform_init_seed) >= 0:
        x0, _ = make_init_particle(dim, uniform_init_seed, bound)
    else:
        x0 = np.zeros(dim, dtype=np.float32)
    regions = np.zeros((R, P, dim), dtype=np.float32)
    for r in range(R):
        spread = init_rng.normal(0.0, 0.1, size=(P, dim)).astype(np.float32)
        regions[r] = np.clip(x0[None, :] + spread, -bound, bound)

    @jax.jit
    def composite_fitness_full_bank(p):
        return -loss_fn(p, state_bank_jax)
    best_x = x0.astype(np.float32)
    best_y = float(composite_fitness_full_bank(jnp.asarray(x0)))
    trace = [{"step": 0, "best_fitness": best_y, "phase": "init"}]
    rlx_logger.info(f"[gcpfo] init: R={R} P={P} sigma={sigma} tau_init={temperature_init} dim={dim} best_y={best_y:.4f}")

    for t in range(int(n_iter)):
        tau_t = 1.0 / (1.0 + temperature_init * (t / max(1, int(n_iter))))
        idx = rng.choice(bank_size, size=minibatch_size, replace=False)
        batch = state_bank_jax[idx]
        for r in range(R):
            particles = jnp.asarray(regions[r])
            losses, grads = loss_and_grad(particles, batch)
            losses_np = np.asarray(losses, dtype=np.float64)
            grads_np = np.asarray(grads, dtype=np.float64)

            iter_best_idx = int(np.argmin(losses_np))
            cand_x = np.asarray(particles[iter_best_idx], dtype=np.float32)
            cand_y = float(composite_fitness_full_bank(jnp.asarray(cand_x)))
            if cand_y > best_y:
                best_y = cand_y
                best_x = cand_x

            log_w = -losses_np / max(tau_t, 1e-8)
            log_w = log_w - log_w.max()
            w = np.exp(log_w)
            w = w / max(w.sum(), 1e-12)
            chosen = rng.choice(P, size=P, replace=True, p=w)
            new_particles = np.asarray(particles)[chosen]
            new_grads = grads_np[chosen]

            G = new_grads.T / np.sqrt(P)
            C = G @ G.T
            evals, evecs = np.linalg.eigh(C)
            evals = np.maximum(evals, 0.0)
            tr = evals.sum() + eps
            evals_tilde = evals / tr
            beta = evecs * np.sqrt(evals_tilde)[None, :]

            eta = rng.normal(0.0, 1.0, size=(P, dim))
            updates = (beta @ eta.T).T * sigma
            new_particles = new_particles + updates.astype(np.float32)
            new_particles = np.clip(new_particles, -bound, bound)
            regions[r] = new_particles

        trace.append({"step": t + 1, "best_fitness": best_y, "tau": tau_t, "phase": "gcpfo"})
        if (t + 1) % 5 == 0 or t == 0:
            rlx_logger.info(f"[gcpfo] iter {t+1}/{int(n_iter)}: tau={tau_t:.4f} best_y={best_y:.4f}")

    return {"best_particle": best_x.astype(np.float32), "best_fitness": float(best_y), "trace": trace, "n_regions": R, "particles_per_region": P}


def run_surrogate_search_urma(
    cfg,
    save_path,
    critic,
    critic_params,
    train_env,
    base_env_state,
    decode_obs_fn,
):
    bank_path = cfg.surrogate_state_bank_path or os.path.join(save_path, "surrogate.npz")
    bank = np.load(bank_path)
    state_bank = bank["state_bank"]
    emb_idx = np.asarray(bank["embodiment_obs_indices"], dtype=np.int32)
    rlx_logger.info(f"Loaded state bank: shape={state_bank.shape}, embodiment_dim={emb_idx.shape[0]}")

    base_env_state = jax.tree_util.tree_map(
        lambda x: x[:1] if hasattr(x, "shape") and len(x.shape) > 0 and x.shape[0] > 1 else x,
        base_env_state,
    )

    @jax.jit
    def critic_apply_decoded(params, c_jd, c_jo, c_fd, c_fo, c_gs):
        return critic.apply(params, c_jd, c_jo, c_fd, c_fo, c_gs)

    @jax.jit
    def get_patch(state, factor_batch):
        return train_env.get_factor_observation_patch(state, factor_batch)
    dim = int(train_env.factor_dim())
    rlx_logger.info(f"[surrogate] factor-space search in [-1, 1]^{dim} (seen_robot scales per-attribute internally)")

    state_bank_jax = jnp.asarray(state_bank, dtype=jnp.float32)
    emb_idx_jax = jnp.asarray(emb_idx, dtype=jnp.int32)

    @jax.jit
    def fitness_jit(particle):
        particles_broadcast = jnp.broadcast_to(particle[None, :], (1, particle.shape[0]))
        description_patch = get_patch(base_env_state, particles_broadcast)
        description_patch_single = description_patch[0]
        states = state_bank_jax.at[:, emb_idx_jax].set(description_patch_single[None, :])
        decoded = decode_obs_fn(states)
        c_jd, c_jo, c_fd, c_fo, c_gs = decoded[3], decoded[4], decoded[5], decoded[6], decoded[7]
        values = critic_apply_decoded(critic_params, c_jd, c_jo, c_fd, c_fo, c_gs)
        return values.mean()
    fitness_jit(jnp.zeros(dim, dtype=jnp.float32))

    def fitness_fn(particle):
        return float(fitness_jit(jnp.asarray(np.asarray(particle, dtype=np.float32))))

    method = cfg.surrogate_search_method.lower()
    seed = int(cfg.surrogate_search_seed)
    curriculum_coeff = float(cfg.surrogate_search_curriculum_coeff)
    rlx_logger.info(f"Running surrogate search with method={method}, dim={dim}, seed={seed}, curriculum_coeff={curriculum_coeff:.3f}")
    t0 = time.time()
    init_seed = int(cfg.surrogate_init_particle_uniform_seed)

    baseline_l2_w = float(cfg.surrogate_particle_l2_prior_weight)
    baseline_anchor_mode = str(cfg.surrogate_particle_l2_anchor_mode)
    if baseline_anchor_mode == "zero":
        l2_anchor_vec = jnp.zeros(int(dim), dtype=jnp.float32)
    else:
        l2_anchor_vec = jnp.asarray(make_init_particle(dim, init_seed, curriculum_coeff)[0])
    baseline_pen_jit = jax.jit(lambda p: fitness_jit(p) - jnp.float32(baseline_l2_w) * jnp.mean((p - l2_anchor_vec) ** 2))
    baseline_pen_jit(jnp.zeros(int(dim), dtype=jnp.float32))

    def baseline_fitness(particle):
        return float(baseline_pen_jit(jnp.asarray(np.asarray(particle, dtype=np.float32))))
    baseline_fitness_batch_jit = jax.jit(jax.vmap(baseline_pen_jit))
    rlx_logger.info(f"[surrogate] objective: V - {baseline_l2_w} * mean((f-anchor)^2), anchor={baseline_anchor_mode}")

    if method == "random":
        result = run_random_search(baseline_fitness, dim, int(cfg.surrogate_search_n_evals), seed=seed, curriculum_coeff=curriculum_coeff)
    elif method == "cma_es":
        result = run_cma_es(baseline_fitness, dim, int(cfg.surrogate_search_n_iter), popsize=int(cfg.surrogate_search_popsize) or None, seed=seed, curriculum_coeff=curriculum_coeff, uniform_init_seed=init_seed)
    elif method == "bo":
        result = run_bo(baseline_fitness, dim, int(cfg.surrogate_search_n_iter), n_init=int(cfg.surrogate_search_n_init), seed=seed, curriculum_coeff=curriculum_coeff, uniform_init_seed=init_seed)
    elif method == "nominal":
        result = run_nominal(fitness_fn, dim, uniform_init_seed=init_seed, curriculum_coeff=curriculum_coeff)
    elif method == "vgds":
        result = run_vgds_urma(
            fitness_fn=fitness_fn,
            critic_apply_decoded=critic_apply_decoded,
            critic_params=critic_params,
            decode_obs_fn=decode_obs_fn,
            get_embodiment_patch_fn=get_patch,
            base_env_state=base_env_state,
            state_bank=state_bank,
            emb_idx=emb_idx,
            dim=dim,
            n_steps=int(cfg.surrogate_search_n_iter),
            minibatch_size=int(cfg.surrogate_search_minibatch_size),
            learning_rate=float(cfg.surrogate_search_learning_rate),
            max_step=float(cfg.surrogate_search_max_step),
            seed=seed,
            curriculum_coeff=curriculum_coeff,
            l2_prior_weight=float(cfg.surrogate_particle_l2_prior_weight),
            l2_anchor_mode=str(cfg.surrogate_particle_l2_anchor_mode),
            uniform_init_seed=init_seed,
        )
    elif method == "pso":
        result = run_pso_urma(
            fitness_batch_fn=baseline_fitness_batch_jit, dim=dim,
            n_iter=int(cfg.surrogate_search_n_iter),
            n_particles=int(cfg.surrogate_search_popsize) or 32,
            seed=seed, curriculum_coeff=curriculum_coeff, uniform_init_seed=init_seed,
        )
    elif method == "cem":
        result = run_cem(
            fitness_batch_fn=baseline_fitness_batch_jit, dim=dim,
            n_iter=int(cfg.surrogate_search_n_iter),
            n_particles=int(cfg.surrogate_search_popsize) or 32,
            seed=seed, curriculum_coeff=curriculum_coeff, uniform_init_seed=init_seed,
        )
    elif method == "de":
        result = run_de(
            fitness_batch_fn=baseline_fitness_batch_jit, dim=dim,
            n_iter=int(cfg.surrogate_search_n_iter),
            n_particles=int(cfg.surrogate_search_popsize) or 32,
            seed=seed, curriculum_coeff=curriculum_coeff, uniform_init_seed=init_seed,
        )
    elif method == "ars":
        result = run_ars(
            fitness_batch_fn=baseline_fitness_batch_jit, dim=dim,
            n_iter=int(cfg.surrogate_search_n_iter),
            n_directions=int(cfg.surrogate_search_popsize) or 16,
            noise_std=float(cfg.surrogate_search_max_step),
            lr=float(cfg.surrogate_search_learning_rate),
            seed=seed, curriculum_coeff=curriculum_coeff, uniform_init_seed=init_seed,
        )
    elif method == "gcpfo":
        result = run_gcpfo_urma(
            critic_apply_decoded=critic_apply_decoded,
            critic_params=critic_params,
            decode_obs_fn=decode_obs_fn,
            get_embodiment_patch_fn=get_patch,
            base_env_state=base_env_state,
            state_bank=state_bank,
            emb_idx=emb_idx,
            dim=dim,
            n_iter=int(cfg.surrogate_search_n_iter),
            minibatch_size=int(cfg.surrogate_search_minibatch_size),
            n_regions=int(cfg.surrogate_gcpfo_n_regions),
            particles_per_region=int(cfg.surrogate_gcpfo_particles_per_region),
            step_size=float(cfg.surrogate_gcpfo_step_size),
            temperature_init=float(cfg.surrogate_gcpfo_temperature_init),
            eps=float(cfg.surrogate_gcpfo_eps),
            seed=seed,
            curriculum_coeff=curriculum_coeff,
            l2_prior_weight=float(cfg.surrogate_particle_l2_prior_weight),
            uniform_init_seed=init_seed,
        )
    elif method == "turbo":
        result = run_turbo_urma(
            critic_apply_decoded=critic_apply_decoded,
            critic_params=critic_params,
            decode_obs_fn=decode_obs_fn,
            get_embodiment_patch_fn=get_patch,
            base_env_state=base_env_state,
            state_bank=state_bank,
            emb_idx=emb_idx,
            dim=dim,
            n_iter=int(cfg.surrogate_search_n_iter),
            n_candidates=int(cfg.surrogate_turbo_n_candidates_per_iter),
            init_radius=float(cfg.surrogate_turbo_init_radius),
            min_radius=float(cfg.surrogate_turbo_min_radius),
            max_radius=float(cfg.surrogate_turbo_max_radius),
            shrink_factor=float(cfg.surrogate_turbo_shrink_factor),
            expand_factor=float(cfg.surrogate_turbo_expand_factor),
            seed=seed,
            curriculum_coeff=curriculum_coeff,
            l2_prior_weight=float(cfg.surrogate_particle_l2_prior_weight),
            uniform_init_seed=init_seed,
        )
    else:
        raise ValueError(f"Unknown surrogate_search_method: {method!r}; expected one of vgds|random|bo|cma_es|turbo|pso|cem|de|ars|gcpfo|nominal")
    rlx_logger.info(f"Search complete in {time.time() - t0:.1f}s; best_fitness={result['best_fitness']:.4f}")

    out_path = cfg.surrogate_search_output_path or os.path.join(save_path, f"surrogate_search_{method}.npz")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        method=method,
        particle=np.asarray(result["best_particle"], dtype=np.float32),
        best_fitness=np.asarray(result["best_fitness"], dtype=np.float64),
    )
    side_json = os.path.splitext(out_path)[0] + ".json"
    with open(side_json, "w") as f:
        json.dump({
            "method": method,
            "best_particle": result["best_particle"].tolist(),
            "best_fitness": float(result["best_fitness"]),
            "trace": result["trace"],
            "dim": int(dim),
            "state_bank_path": bank_path,
            "wall_time_s": time.time() - t0,
        }, f, indent=2)
    rlx_logger.info(f"Wrote search result to {out_path} and {side_json}")
    return result
