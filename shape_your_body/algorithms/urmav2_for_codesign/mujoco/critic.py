import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal


def get_critic(config):
    return Critic(
        config.algorithm.nr_critic_heads,
        config.algorithm.softmax_temperature,
        config.algorithm.softmax_temperature_min,
        config.algorithm.stability_epsilon,
        config.algorithm.critic_arch_variant,
        config.algorithm.critic_latent_state_dim,
    )


class Critic(nn.Module):
    nr_critic_heads: int
    softmax_temperature: float
    softmax_temperature_min: float
    stability_epsilon: float
    arch_variant: str
    latent_state_dim: int

    @nn.compact
    def __call__(self, joint_description, joint_state, foot_description, foot_state, general_state):
        joint_log_softmax_temperature = self.param("joint_log_softmax_temperature", constant(jnp.log(self.softmax_temperature - self.softmax_temperature_min)), (1,))
        joint_state_mask = nn.Dense(64)(jnp.concatenate([joint_description, joint_state], axis=-1))
        joint_state_mask = nn.LayerNorm()(joint_state_mask)
        joint_state_mask = nn.elu(joint_state_mask)
        joint_state_mask = nn.Dense(64)(joint_state_mask)
        joint_state_mask = jnp.clip(nn.tanh(joint_state_mask), -1.0 + self.stability_epsilon, 1.0 - self.stability_epsilon)
        if self.arch_variant == "design_aware_values":
            latent_joint_state = nn.Dense(self.latent_state_dim)(jnp.concatenate([joint_description, joint_state], axis=-1))
        else:
            latent_joint_state = nn.Dense(self.latent_state_dim)(joint_state)
        latent_joint_state = nn.elu(latent_joint_state)
        joint_e_x = jnp.exp(joint_state_mask / (jnp.exp(joint_log_softmax_temperature) + self.softmax_temperature_min))
        joint_state_mask = joint_e_x / (joint_e_x.sum(axis=-1, keepdims=True) + self.stability_epsilon)
        joint_state_mask = jnp.repeat(jnp.expand_dims(joint_state_mask, axis=-1), latent_joint_state.shape[-1], axis=-1)
        masked_joint_state = joint_state_mask * jnp.expand_dims(latent_joint_state, axis=-2)
        masked_joint_state = jnp.reshape(masked_joint_state, masked_joint_state.shape[:-2] + (masked_joint_state.shape[-2] * masked_joint_state.shape[-1],))
        joint_latent = jnp.sum(masked_joint_state, axis=-2)

        foot_log_softmax_temperature = self.param("foot_log_softmax_temperature", constant(jnp.log(self.softmax_temperature - self.softmax_temperature_min)), (1,))
        foot_state_mask = nn.Dense(32)(jnp.concatenate([foot_description, foot_state], axis=-1))
        foot_state_mask = nn.LayerNorm()(foot_state_mask)
        foot_state_mask = nn.elu(foot_state_mask)
        foot_state_mask = nn.Dense(32)(foot_state_mask)
        foot_state_mask = jnp.clip(nn.tanh(foot_state_mask), -1.0 + self.stability_epsilon, 1.0 - self.stability_epsilon)
        if self.arch_variant == "design_aware_values":
            latent_foot_state = nn.Dense(self.latent_state_dim)(jnp.concatenate([foot_description, foot_state], axis=-1))
        else:
            latent_foot_state = nn.Dense(self.latent_state_dim)(foot_state)
        latent_foot_state = nn.elu(latent_foot_state)
        foot_e_x = jnp.exp(foot_state_mask / (jnp.exp(foot_log_softmax_temperature) + self.softmax_temperature_min))
        foot_state_mask = foot_e_x / (foot_e_x.sum(axis=-1, keepdims=True) + self.stability_epsilon)
        foot_state_mask = jnp.repeat(jnp.expand_dims(foot_state_mask, axis=-1), latent_foot_state.shape[-1], axis=-1)
        masked_foot_state = foot_state_mask * jnp.expand_dims(latent_foot_state, axis=-2)
        masked_foot_state = jnp.reshape(masked_foot_state, masked_foot_state.shape[:-2] + (masked_foot_state.shape[-2] * masked_foot_state.shape[-1],))
        foot_latent = jnp.sum(masked_foot_state, axis=-2)

        combined_input = jnp.concatenate([joint_latent, foot_latent, general_state], axis=-1)

        values = []
        for _ in range(self.nr_critic_heads):
            value = nn.WeightNorm(nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)), variable_filter={'kernel',})(combined_input)
            value = nn.LayerNorm()(value)
            value = nn.elu(value)
            value = nn.WeightNorm(nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)), variable_filter={'kernel',})(value)
            value = nn.elu(value)
            value = nn.WeightNorm(nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)), variable_filter={'kernel',})(value)
            value = nn.elu(value)
            value = nn.WeightNorm(nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)), variable_filter={'kernel',})(value)
            value = nn.elu(value)
            value = nn.WeightNorm(nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)), variable_filter={'kernel',})(value)
            value = nn.elu(value)
            value = nn.WeightNorm(nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0)), variable_filter={'kernel',})(value)
            values.append(value)
        value = jnp.concatenate(values, axis=-1)
        return value
