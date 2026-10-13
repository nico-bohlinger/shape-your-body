from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.initial_state_functions.default import DefaultDRInitialState
from shape_your_body.environments.locomotion.urmav2_for_codesign.mujoco.domain_randomization.initial_state_functions.random import RandomDRInitialState


def get_initial_state_function(name, env, **kwargs):
    if name == "default":
        return DefaultDRInitialState(env, **kwargs)
    elif name == "random":
        return RandomDRInitialState(env, **kwargs)
    else:
        raise NotImplementedError
