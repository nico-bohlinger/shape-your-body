import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


def get_policy(config):
    return (
        Policy(
            config.algorithm.std_dev,
            config.algorithm.softmax_temperature, config.algorithm.softmax_temperature_min,
            config.algorithm.stability_epsilon,
            config.algorithm.policy_mean_abs_clip,
            config.algorithm.policy_std_min_clip, config.algorithm.policy_std_max_clip
        ),
        get_processed_action_function()
    )


class Policy(nn.Module):
    std_dev: float
    softmax_temperature: float
    softmax_temperature_min: float
    stability_epsilon: float
    policy_mean_abs_clip: float
    policy_std_min_clip: float
    policy_std_max_clip: float

    @nn.compact
    def __call__(self, joint_description, joint_state, general_state):
        joint_log_softmax_temperature = self.param("joint_log_softmax_temperature", constant(jnp.log(self.softmax_temperature - self.softmax_temperature_min)), (1,))
        joint_description_latent = nn.Dense(256)(jnp.concatenate([joint_description, joint_state], axis=-1))
        joint_description_latent = nn.LayerNorm()(joint_description_latent)
        joint_description_latent = nn.elu(joint_description_latent)
        joint_description_latent = nn.Dense(256)(joint_description_latent)
        joint_state_mask = jnp.clip(nn.tanh(joint_description_latent), -1.0 + self.stability_epsilon, 1.0 - self.stability_epsilon)
        latent_joint_state = nn.Dense(4)(joint_state)
        latent_joint_state = nn.elu(latent_joint_state)
        joint_e_x = jnp.exp(joint_state_mask / (jnp.exp(joint_log_softmax_temperature) + self.softmax_temperature_min))
        joint_state_mask = joint_e_x / (joint_e_x.sum(axis=-1, keepdims=True) + self.stability_epsilon)
        joint_state_mask_expanded = jnp.repeat(jnp.expand_dims(joint_state_mask, axis=-1), latent_joint_state.shape[-1], axis=-1)
        masked_joint_state = joint_state_mask_expanded * jnp.expand_dims(latent_joint_state, axis=-2)
        masked_joint_state = jnp.reshape(masked_joint_state, masked_joint_state.shape[:-2] + (masked_joint_state.shape[-2] * masked_joint_state.shape[-1],))
        joint_latent = jnp.sum(masked_joint_state, axis=-2)

        combined_input = jnp.concatenate([joint_latent, general_state], axis=-1)

        action_latent = nn.WeightNorm(nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)), variable_filter={'kernel',})(combined_input)
        action_latent = nn.LayerNorm()(action_latent)
        action_latent = nn.elu(action_latent)
        action_latent = nn.WeightNorm(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)), variable_filter={'kernel',})(action_latent)
        action_latent = nn.elu(action_latent)
        action_latent = nn.WeightNorm(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)), variable_filter={'kernel',})(action_latent)
        action_latent = nn.elu(action_latent)
        action_latent = nn.WeightNorm(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)), variable_filter={'kernel',})(action_latent)
        action_latent = nn.elu(action_latent)
        action_latent = nn.WeightNorm(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)), variable_filter={'kernel',})(action_latent)

        action_latent = jnp.expand_dims(action_latent, axis=-2)
        policy_mean = action_latent * joint_state_mask
        policy_mean = jnp.sum(policy_mean, axis=-1)
        policy_mean = jnp.clip(policy_mean, -self.policy_mean_abs_clip, self.policy_mean_abs_clip)

        policy_logstd = nn.Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(np.log(self.std_dev)))(joint_description_latent)
        policy_logstd = jnp.clip(policy_logstd, np.log(self.policy_std_min_clip), np.log(self.policy_std_max_clip))

        return policy_mean, policy_logstd.squeeze(-1)


def get_processed_action_function():
    def get_clipped_and_scaled_action(action):
        return action
    return jax.jit(get_clipped_and_scaled_action)
