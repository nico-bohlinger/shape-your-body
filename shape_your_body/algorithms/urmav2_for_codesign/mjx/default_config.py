from ml_collections import config_dict


def get_config(algorithm_name):
    config = config_dict.ConfigDict()

    config.name = algorithm_name

    config.device = "gpu"  # cpu, gpu
    config.total_timesteps = 1009254400
    config.start_learning_rate = 1.5e-3
    config.end_learning_rate = 3e-4
    config.learning_rate_schedule = "linear"  # linear, exponential
    config.nr_steps = 64
    config.nr_epochs = 10
    config.minibatch_size = 16384
    config.gamma = 0.99
    config.gae_lambda = 0.7
    config.clip_range = 0.1
    config.entropy_coef = 0.005
    config.critic_coef = 1.0
    config.max_grad_norm = 3.0
    config.kl_threshold = 0.5
    config.softmax_temperature = 1.0
    config.softmax_temperature_min = 0.025
    config.stability_epsilon = 1e-8
    config.std_dev = 1.0
    config.policy_mean_abs_clip = 10.0
    config.policy_std_min_clip = 0.01
    config.policy_std_max_clip = 2.0
    config.nr_critic_heads = 3
    config.mean_critic_returns = True
    config.critic_arch_variant = "design_aware_values"  # original, design_aware_values
    config.critic_latent_state_dim = 32
    config.embodiment_local_resample_probability = 0.2
    config.embodiment_local_resample_range_fraction = 0.1
    config.evaluation_and_save_frequency = 18350080  # -1 to disable
    config.export_state_bank_at_end = True
    config.print_configs = True

    config.export_state_bank_path = ""  # test mode: collects a state bank to this path

    config.eval_design_enabled = False  # test mode: evaluates the design in eval_design_particle_path
    config.eval_design_particle_path = ""  # nominal design if empty
    config.eval_design_force_max_curriculum = True
    config.eval_design_total_steps = 200000
    config.eval_design_window_steps = 20000
    config.eval_design_convergence_tol = 1e-3
    config.eval_design_output_path = ""

    config.surrogate_search_method = ""  # test mode: vgds, random, bo, cma_es, turbo, pso, cem, de, ars, gcpfo, nominal
    config.surrogate_state_bank_path = ""
    config.surrogate_search_output_path = ""
    config.surrogate_search_n_iter = 50
    config.surrogate_search_n_init = 10
    config.surrogate_search_n_evals = 3200
    config.surrogate_search_popsize = 0  # 0 for the default of each method
    config.surrogate_search_seed = 0
    config.surrogate_search_minibatch_size = 64
    config.surrogate_search_learning_rate = 0.05
    config.surrogate_search_max_step = 0.1
    config.surrogate_search_curriculum_coeff = 1.0  # bounds the search space to [-c, c]
    config.surrogate_init_particle_uniform_seed = -1  # -1 to start at the nominal design, otherwise the seed of a uniformly sampled start design
    config.surrogate_particle_l2_prior_weight = 100.0
    config.surrogate_particle_l2_anchor_mode = "zero"  # zero, perturbed_start

    config.surrogate_turbo_init_radius = 0.1
    config.surrogate_turbo_min_radius = 0.005
    config.surrogate_turbo_max_radius = 0.5
    config.surrogate_turbo_shrink_factor = 0.7
    config.surrogate_turbo_expand_factor = 1.3
    config.surrogate_turbo_n_candidates_per_iter = 64

    config.surrogate_gcpfo_n_regions = 2
    config.surrogate_gcpfo_particles_per_region = 32
    config.surrogate_gcpfo_step_size = 0.2
    config.surrogate_gcpfo_temperature_init = 1.0
    config.surrogate_gcpfo_eps = 1e-8

    return config
