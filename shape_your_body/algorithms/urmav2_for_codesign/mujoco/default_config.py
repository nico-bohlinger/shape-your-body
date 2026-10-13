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
    config.print_configs = True

    config.eval_design_particle_path = ""  # test mode: renders this design, nominal design if empty

    return config
