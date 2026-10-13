from functools import partial
import jax
import jax.numpy as jnp

from shape_your_body.environments.locomotion.urmav2_for_codesign.mjx.state import MultiState


class MultiEnvironment:
    def __init__(self, train_envs, train_robot_short_names):
        self.train_envs = train_envs
        self.train_robot_short_names = train_robot_short_names
        self.nr_train_robots = len(train_envs)
        self.nr_envs_per_train_robot = train_envs[0].nr_envs
        self.nr_envs = self.nr_train_robots * self.nr_envs_per_train_robot
        self.single_action_space = train_envs[0].single_action_space
        self.single_observation_space = train_envs[0].single_observation_space
        self.horizon = train_envs[0].horizon
        self.joint_description_size = train_envs[0].joint_description_size
        self.foot_description_size = train_envs[0].foot_description_size
        self.joint_observation_size = train_envs[0].joint_observation_size
        self.foot_observation_size = train_envs[0].foot_observation_size
        self.nr_actuator_joints = train_envs[0].nr_actuator_joints
        self.nr_feet = train_envs[0].nr_feet
        self.policy_general_obs_idx = train_envs[0].policy_general_obs_idx
        self.critic_general_obs_idx = train_envs[0].critic_general_obs_idx
        self.seen_robot_active = train_envs[0].seen_robot_active
        self.co_design_joint_description_indices = train_envs[0].co_design_joint_description_indices
        self.co_design_foot_description_indices = train_envs[0].co_design_foot_description_indices
        self.co_design_general_obs_indices = train_envs[0].co_design_general_obs_indices
        self.dt = train_envs[0].dt
        self.per_robot_factor_dims = tuple(int(env.factor_dim()) for env in train_envs)
        self.max_factor_dim = max(self.per_robot_factor_dims) if self.per_robot_factor_dims else 0
        self.info_keys = None
        self.internal_state_concat_keys = None


    def init(self, keys):
        per_robot_states = [
            env.train_reset(keys[i * self.nr_envs_per_train_robot:(i + 1) * self.nr_envs_per_train_robot])
            for i, env in enumerate(self.train_envs)
        ]
        info_keys = []
        for s in per_robot_states:
            for k in s.info:
                if k not in info_keys:
                    info_keys.append(k)
        self.info_keys = tuple(info_keys)
        concat_keys = []
        ref = per_robot_states[0].internal_state
        for k, v in ref.items():
            if not isinstance(v, jax.Array):
                continue
            if v.ndim >= 1 and v.shape[0] == self.nr_envs_per_train_robot:
                ok = True
                for s in per_robot_states[1:]:
                    if k not in s.internal_state:
                        ok = False
                        break
                    vi = s.internal_state[k]
                    if not isinstance(vi, jax.Array) or vi.ndim != v.ndim or vi.shape[1:] != v.shape[1:]:
                        ok = False
                        break
                if ok:
                    concat_keys.append(k)
        self.internal_state_concat_keys = tuple(concat_keys)


    def factor_dim(self):
        return self.max_factor_dim


    @partial(jax.jit, static_argnums=(0,))
    def train_reset(self, keys):
        per_robot_states = tuple(
            env.train_reset(keys[i * self.nr_envs_per_train_robot:(i + 1) * self.nr_envs_per_train_robot])
            for i, env in enumerate(self.train_envs)
        )
        return self.create_multi_state(per_robot_states)


    @partial(jax.jit, static_argnums=(0,))
    def step(self, multi_state, action):
        per_robot_states = tuple(
            env.step(
                multi_state.env_states[i],
                action[i * self.nr_envs_per_train_robot:(i + 1) * self.nr_envs_per_train_robot],
            )
            for i, env in enumerate(self.train_envs)
        )
        return self.create_multi_state(per_robot_states)


    @partial(jax.jit, static_argnums=(0,))
    def set_factor_params(self, multi_state, particles):
        per_robot_states = []
        for i, env in enumerate(self.train_envs):
            start = i * self.nr_envs_per_train_robot
            stop = start + self.nr_envs_per_train_robot
            env_state = multi_state.env_states[i]
            updated_internal_state = dict(env_state.internal_state)
            for k in self.internal_state_concat_keys:
                if k in multi_state.internal_state:
                    updated_internal_state[k] = multi_state.internal_state[k][start:stop]
            env_state = env_state.replace(internal_state=updated_internal_state)
            factor_slice = particles[start:stop, :self.per_robot_factor_dims[i]]
            per_robot_states.append(env.set_factor_params(env_state, factor_slice))
        return self.create_multi_state(tuple(per_robot_states))


    def create_multi_state(self, per_robot_states):
        info = {
            k: jnp.concatenate([s.info[k] for s in per_robot_states if k in s.info], axis=0)
            for k in self.info_keys
        }
        internal_state = {
            k: jnp.concatenate([s.internal_state[k] for s in per_robot_states], axis=0)
            for k in self.internal_state_concat_keys
        }
        return MultiState(
            env_states=per_robot_states,
            next_observation=jnp.concatenate([s.next_observation for s in per_robot_states], axis=0),
            actual_next_observation=jnp.concatenate([s.actual_next_observation for s in per_robot_states], axis=0),
            reward=jnp.concatenate([s.reward for s in per_robot_states], axis=0),
            terminated=jnp.concatenate([s.terminated for s in per_robot_states], axis=0),
            truncated=jnp.concatenate([s.truncated for s in per_robot_states], axis=0),
            info=info,
            internal_state=internal_state,
        )


    def get_curriculum_coefficients(self, multi_state):
        return jnp.concatenate(
            [env.get_curriculum_coefficients(s) for env, s in zip(self.train_envs, multi_state.env_states)],
            axis=0,
        )


    @partial(jax.jit, static_argnums=(0,))
    def get_factor_observation_patch(self, multi_state, factor):
        if self.nr_train_robots != 1:
            raise NotImplementedError("get_factor_observation_patch only supported for nr_train_robots=1")
        env_state = multi_state.env_states[0]
        fb = factor.shape[0]
        nr_envs = env_state.data.qpos.shape[0]
        if fb != nr_envs:
            env_state = jax.tree.map(
                lambda x: x[:fb] if isinstance(x, jax.Array) and x.ndim >= 1 and x.shape[0] == nr_envs else x,
                env_state,
            )
        return self.train_envs[0].get_factor_observation_patch(env_state, factor)


    @partial(jax.jit, static_argnums=(0,))
    def sample_initial_command(self, multi_state, keys):
        if self.nr_train_robots != 1:
            raise NotImplementedError("sample_initial_command only supported for nr_train_robots=1")
        new_env_state = self.train_envs[0].sample_initial_command(multi_state.env_states[0], keys)
        new_env_states = (new_env_state,)
        return self.create_multi_state(new_env_states)


    def call(self, attribute_name):
        return [getattr(env, attribute_name) for env in self.train_envs]


    def close(self):
        for env in self.train_envs:
            env.close()
